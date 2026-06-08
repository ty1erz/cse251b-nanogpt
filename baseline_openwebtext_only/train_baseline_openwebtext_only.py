"""Train the GPT baseline using OpenWebText data only."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "common"))

from baseline_mix_50_20_15_15.model import GPT, GPTConfig  # noqa: E402
from baseline_training import train_baseline  # noqa: E402


if __name__ == "__main__":
    data_root = os.environ.get(
        "OPENWEBTEXT_DATA_ROOT",
        os.path.join(
            os.environ.get("NANOGPT_DATA_ROOT", REPO_ROOT),
            "data",
            "openwebtext",
        ),
    )
    train_baseline(
        experiment_name="baseline_openwebtext_only",
        source_name="OpenWebText",
        data_dir=data_root,
        file_format="bin",
        output_root=HERE,
        model_class=GPT,
        config_class=GPTConfig,
    )
