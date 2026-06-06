from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CSV = SCRIPT_DIR / "hotpot_eval_base_num_rollouts.csv"
RLSC_CSV = SCRIPT_DIR / "hotpot_eval_RLSC_num_rollouts.csv"

KS = [4, 8, 16]
METHODS = ["Base", "CSR (Ours)"]
COLORS = {"Base": "#2563eb", "CSR (Ours)": "#dc2626"}


def load_metrics(csv_path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["k", "n", "accuracy_mean", "ece_soft", "brier_soft", "auroc_conf"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "validation"].copy()
    df = df.dropna(subset=["k"]).copy()
    df["method"] = method_name
    return df.sort_values("k").reset_index(drop=True)


def build_plot_df() -> pd.DataFrame:
    base = load_metrics(BASE_CSV, "Base")
    rlsc = load_metrics(RLSC_CSV, "CSR (Ours)")
    df = pd.concat([base, rlsc], ignore_index=True)
    df = df[df["k"].isin(KS)].copy()
    df["k"] = df["k"].astype(int)
    return df


def draw_grouped_bars(ax, df: pd.DataFrame, metric: str, ylabel: str, ylim=None) -> None:
    x = np.arange(len(KS), dtype=float)
    width = 0.34

    for idx, method in enumerate(METHODS):
        sub = df[df["method"] == method].set_index("k")
        y = [float(sub.loc[k, metric]) if k in sub.index else np.nan for k in KS]
        offset = (-width / 2) if idx == 0 else (width / 2)
        ax.bar(
            x + offset,
            y,
            width=width,
            color=COLORS[method],
            edgecolor="#111111",
            linewidth=1.6,
            label=method,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in KS], fontsize=13)
    ax.set_xlabel("Num rollouts (k)", fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13, width=1.6)
    ax.grid(axis="y", linestyle=":", color="#c9ced6", linewidth=1.0, alpha=0.9, zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(2.1)


def plot_single_metric(df: pd.DataFrame, metric: str, ylabel: str, out_name: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    draw_grouped_bars(ax, df, metric=metric, ylabel=ylabel, ylim=ylim)
    ax.legend(loc="best", frameon=False, fontsize=16)
    fig.tight_layout()
    out_path = SCRIPT_DIR / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    df = build_plot_df()
    plot_single_metric(df, metric="accuracy_mean", ylabel="Accuracy", out_name="hotpot_num_rollouts_accuracy.png", ylim=(0.0, 0.4))
    plot_single_metric(df, metric="ece_soft", ylabel="ECE", out_name="hotpot_num_rollouts_ece.png", ylim=(0.0, 0.35))
    plot_single_metric(df, metric="auroc_conf", ylabel="AUROC", out_name="hotpot_num_rollouts_auroc.png", ylim=(0.0, 1.1))


if __name__ == "__main__":
    main()
