"""Contiguous-token loader for one pretraining source.

Supports either a directory of split-tagged uint16 ``.npy`` shards (the
FineWeb-Edu layout) or a directory containing uint16 ``train.bin`` and
``val.bin`` files (the OpenWebText layout).
"""

import os
from typing import List

import numpy as np
import torch


def _discover_files(data_dir: str, split: str, file_format: str) -> List[str]:
    if file_format == "npy":
        if not os.path.isdir(data_dir):
            return []
        files = [
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.endswith(".npy") and f"_{split}_" in name
        ]
        return sorted(files)
    if file_format == "bin":
        path = os.path.join(data_dir, f"{split}.bin")
        return [path] if os.path.isfile(path) else []
    raise ValueError(f"unsupported file format: {file_format!r}")


def _load_tokens(path: str, file_format: str):
    if file_format == "npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.uint16, mode="r")


class SingleSourceDataLoader:
    """Round-robin loader over one source's token files."""

    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_dir: str,
        file_format: str,
        source_name: str,
        master_process: bool = True,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported split: {split!r}")
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_dir = os.path.abspath(data_dir)
        self.file_format = file_format
        self.source_name = source_name
        self.files = _discover_files(self.data_dir, split, file_format)
        if not self.files:
            expected = (
                f"split-tagged .npy shards in {self.data_dir}"
                if file_format == "npy"
                else os.path.join(self.data_dir, f"{split}.bin")
            )
            raise FileNotFoundError(
                f"no {source_name} {split} data found; expected {expected}"
            )
        if master_process:
            print(
                f"[{source_name}] {split}: {len(self.files)} token file(s) "
                f"from {self.data_dir}"
            )
        self.reset()

    def reset(self):
        self.file_i = 0
        self.tokens = _load_tokens(self.files[self.file_i], self.file_format)
        self.pos = self.B * self.T * self.process_rank

    def next_batch(self):
        chunk = self.B * self.T
        attempts = 0
        while self.pos + chunk + 1 > len(self.tokens):
            attempts += 1
            if attempts > len(self.files):
                raise RuntimeError(
                    f"[{self.source_name}] no {self.split} token file has at "
                    f"least {chunk + 1} tokens"
                )
            self.file_i = (self.file_i + 1) % len(self.files)
            self.tokens = _load_tokens(
                self.files[self.file_i], self.file_format
            )
            self.pos = chunk * self.process_rank

        buf = np.asarray(self.tokens[self.pos : self.pos + chunk + 1])
        batch = torch.from_numpy(buf.astype(np.int64))
        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)
        self.pos += chunk * self.num_processes
        return x, y


__all__ = ["SingleSourceDataLoader"]
