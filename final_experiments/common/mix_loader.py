"""
Multi-source token loader for Phase 1+ training.

Each source is a directory of uint16 .npy shards (output of fineweb.py and
prepare_data.py). On every batch we pick a source proportional to its mix
weight, then read B*T+1 contiguous tokens from that source. Long enough runs
hit the requested mix in expectation.

Compatible drop-in for DataLoaderLite: exposes B, T, next_batch(), reset().

Default mix targets the example eval blend from the contest slides:
    FineWeb-Edu 50% / Wikipedia 20% / science 15% / books 15%

If a source directory is missing the loader falls back to FineWeb-Edu only
and prints a warning, so Phase 1 work isn't blocked on full data prep.
"""

import os
from typing import Dict, List

import numpy as np
import torch


def _load_tokens(path: str) -> torch.Tensor:
    arr = np.load(path).astype(np.int32)
    return torch.tensor(arr, dtype=torch.long)


class _SourceShards:
    """Round-robin reader over the .npy shards of a single source."""

    def __init__(self, name: str, shard_paths: List[str], B: int, T: int,
                 process_rank: int, num_processes: int, master: bool):
        assert shard_paths, f"no shards found for source {name!r}"
        self.name = name
        self.shards = sorted(shard_paths)
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        if master:
            print(f"[{name}] {len(self.shards)} shards")
        self.reset()

    def reset(self):
        self.shard_i = 0
        self.tokens = _load_tokens(self.shards[self.shard_i])
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        chunk = B * T
        # Walk forward until we find a shard with enough tokens for one batch.
        # Some data-prep tail shards can be tiny (< B*T+1 tokens); skip them.
        attempts = 0
        while self.pos + chunk + 1 > len(self.tokens):
            attempts += 1
            if attempts > len(self.shards):
                raise RuntimeError(
                    f"[{self.name}] no shard has >= {chunk + 1} tokens "
                    f"(checked all {len(self.shards)})"
                )
            self.shard_i = (self.shard_i + 1) % len(self.shards)
            self.tokens = _load_tokens(self.shards[self.shard_i])
            self.pos = B * T * self.process_rank

        buf = self.tokens[self.pos : self.pos + chunk + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.pos += chunk * self.num_processes
        return x, y

    def state(self):
        return {"shard_i": self.shard_i, "pos": int(self.pos)}

    def load_state(self, s):
        self.shard_i = int(s["shard_i"])
        self.tokens = _load_tokens(self.shards[self.shard_i])
        self.pos = int(s["pos"])


# Set NANOGPT_DATA_ROOT when shards are stored outside the project directory.
# Expected layout:
#   <root>/edu_fineweb10B
#   <root>/data/{wikipedia,science,books}
def _default_source_paths(repo_root: str) -> Dict[str, str]:
    data_root = os.environ.get("NANOGPT_DATA_ROOT", repo_root)
    return {
        "fineweb": os.path.join(data_root, "edu_fineweb10B"),
        "wikipedia": os.path.join(data_root, "data", "wikipedia"),
        "science": os.path.join(data_root, "data", "science"),
        "books": os.path.join(data_root, "data", "books"),
    }


DEFAULT_MIX = {
    "fineweb": 0.56,
    "wikipedia": 0.18,
    "science": 0.13,
    "books": 0.13,
}


def _list_shards(dir_path: str, split: str) -> List[str]:
    """Return shards for the requested split. fineweb-edu uses train/val
    in the filename; the other prep scripts emit no split tag — those go
    entirely into 'train' and val is taken from FineWeb-Edu."""
    if not os.path.isdir(dir_path):
        return []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")]
    if any("_train_" in f or "_val_" in f for f in files):
        # FineWeb-Edu naming: edufineweb_<split>_<idx>.npy
        return [f for f in files if f"_{split}_" in os.path.basename(f)]
    # Other sources: only train shards exist
    return files if split == "train" else []


class MixedDataLoader:
    """Drop-in replacement for DataLoaderLite that mixes multiple sources."""

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int,
                 split: str, mix: Dict[str, float] = None,
                 source_paths: Dict[str, str] = None,
                 repo_root: str = None, seed: int = 1337,
                 master_process: bool = True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        if repo_root is None:
            repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..")
            )
        if source_paths is None:
            source_paths = _default_source_paths(repo_root)
        if mix is None:
            mix = dict(DEFAULT_MIX)

        # collect shards per source; drop sources with no shards for this split
        loaders: Dict[str, _SourceShards] = {}
        for name, weight in mix.items():
            shards = _list_shards(source_paths.get(name, ""), split)
            if not shards:
                if master_process:
                    print(f"[mix] WARNING: no shards for {name!r} ({split}) at "
                          f"{source_paths.get(name)} — dropping from mix")
                continue
            loaders[name] = _SourceShards(
                name, shards, B, T, process_rank, num_processes, master_process,
            )
        if not loaders:
            raise FileNotFoundError("no source has shards for the requested split")

        # renormalize weights over surviving sources
        kept = {n: mix[n] for n in loaders}
        total = sum(kept.values())
        self.weights = {n: w / total for n, w in kept.items()}
        self.loaders = loaders
        self.names = list(self.loaders.keys())
        self.probs = np.array([self.weights[n] for n in self.names], dtype=np.float64)

        if master_process:
            mix_str = ", ".join(f"{n}={self.weights[n]:.0%}" for n in self.names)
            print(f"[mix] active mix ({split}): {mix_str}")

        # Per-rank rng so each DDP rank picks the same source on the same step
        # (so global token mix matches the configured ratio).
        self.rng = np.random.default_rng(seed)

    def reset(self):
        for ld in self.loaders.values():
            ld.reset()

    def next_batch(self):
        idx = self.rng.choice(len(self.names), p=self.probs)
        return self.loaders[self.names[idx]].next_batch()

    def state(self):
        """Snapshot of the loader's resumable state.

        Includes per-source (shard_i, pos) and the numpy RNG state used to
        sample sources. Restoring this on a fresh MixedDataLoader reproduces
        the same source-pick sequence and continues each source's walk
        exactly where we left off.
        """
        return {
            "loaders": {n: l.state() for n, l in self.loaders.items()},
            "rng": self.rng.bit_generator.state,
        }

    def load_state(self, s):
        for n, ls in s.get("loaders", {}).items():
            if n in self.loaders:
                self.loaders[n].load_state(ls)
        if "rng" in s:
            self.rng.bit_generator.state = s["rng"]


__all__ = ["MixedDataLoader", "DEFAULT_MIX"]
