#!/usr/bin/env python3
"""Generate report figures directly from the experiment logs."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
REPO_DIR = REPORT_DIR.parent
LOG_DIR = REPO_DIR / "build-nanogpt" / "log"

BLUE = "#2563eb"
RED = "#dc2626"
INK = "#111111"
GRID = "#c9ced6"
LIGHT_BLUE = "#dbeafe"
LIGHT_RED = "#fee2e2"
LIGHT_GRAY = "#f3f4f6"
LIGHT_GREEN = "#ecf8df"
LIGHT_PURPLE = "#f1eafe"
LIGHT_CREAM = "#fff7e6"


def parse_eval(path: Path, kinds=("evaluate",)) -> list[tuple[int, float]]:
    pattern = re.compile(
        r"^(\d+)\s+(evaluate|quick_eval)\s+ppl\s+([0-9.]+)"
    )
    points = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if match and match.group(2) in kinds:
                points.append((int(match.group(1)), float(match.group(3))))
    return points


def selected_continuation_points() -> list[tuple[int, float]]:
    """Follow the checkpoint lineage that produced the 69.7k best model."""
    paths_and_ranges = [
        (
            LOG_DIR / "v16/log_20260525_alvin_v16_v13_continue.txt",
            38_500,
            43_000,
            ("evaluate",),
        ),
        (
            LOG_DIR / "v16/log_20260526_alvin_v16_v13_continue.txt",
            43_500,
            54_000,
            ("evaluate", "quick_eval"),
        ),
        (
            LOG_DIR / "v16/log_20260527_alvin_v16_v13_continue.txt",
            54_200,
            62_000,
            ("evaluate", "quick_eval"),
        ),
        (
            LOG_DIR / "v16/log_20260528_alvin_v16_from62000_lr15e5_save400_eval100.txt",
            62_100,
            63_600,
            ("evaluate", "quick_eval"),
        ),
        (
            LOG_DIR / "v16/log_20260528_alvin_v16_from63600_lr1e4_save200_eval100_t70000.txt",
            63_700,
            69_600,
            ("evaluate", "quick_eval"),
        ),
        (
            LOG_DIR / "v16/log_20260529_alvin_v16_from69600_lr6e5_save100_eval50_t72000.txt",
            69_650,
            71_800,
            ("evaluate", "quick_eval"),
        ),
    ]
    by_step = {}
    for path, start, end, kinds in paths_and_ranges:
        for step, ppl in parse_eval(path, kinds=kinds):
            if start <= step <= end:
                by_step[step] = ppl
    return sorted(by_step.items())


def style_axis(ax) -> None:
    ax.grid(axis="y", linestyle=":", color=GRID, linewidth=0.9, alpha=0.95, zorder=0)
    ax.tick_params(axis="both", labelsize=8, width=1.2)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.25)


def plot_training_curve() -> None:
    # Team-reported Model 1 checkpoint lineage from presentation slide 5.
    model1_boundary = (38_000, 21.6089)
    model1_continuation = [
        (42_000, 20.23),
        (44_000, 19.95),
        (45_000, 19.70),
        (45_750, 19.41),
        (46_750, 19.34),
        (47_750, 19.28),
        (49_500, 19.13),
        (50_750, 19.09),
    ]

    main = parse_eval(
        LOG_DIR / "v13/log_20260520_alvin_v13_zzw_mix_muon.txt",
        kinds=("evaluate",),
    )
    continuation = selected_continuation_points()

    main_steps = np.asarray([x for x, _ in main]) / 1000
    main_ppl = np.asarray([y for _, y in main])
    cont_steps = np.asarray([x for x, _ in continuation]) / 1000
    cont_ppl = np.asarray([y for _, y in continuation])

    # Prepend the first-epoch endpoint so the two epochs form one continuous
    # checkpoint trajectory rather than two visually disconnected segments.
    model2_cont_steps = np.concatenate(([38.0], cont_steps))
    model2_cont_ppl = np.concatenate(([21.1630], cont_ppl))
    model1_cont_steps = np.asarray(
        [model1_boundary[0]] + [x for x, _ in model1_continuation]
    ) / 1000
    model1_cont_ppl = np.asarray(
        [model1_boundary[1]] + [y for _, y in model1_continuation]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.48),
        gridspec_kw={"width_ratios": [0.82, 1.55], "wspace": 0.25},
    )

    ax = axes[0]
    ax.scatter(
        [38],
        [model1_boundary[1]],
        s=30,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.7,
        label="1st epoch",
        zorder=5,
    )
    ax.plot(
        model1_cont_steps,
        model1_cont_ppl,
        color=RED,
        marker="o",
        markersize=3.0,
        markeredgewidth=0,
        linewidth=1.8,
        label="2nd epoch",
        zorder=3,
    )
    ax.scatter(
        [50.75],
        [19.09],
        s=36,
        color=RED,
        edgecolor=INK,
        linewidth=0.8,
        zorder=6,
    )
    ax.annotate(
        "best 19.09",
        xy=(50.75, 19.09),
        xytext=(46.0, 19.63),
        fontsize=7.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.8},
    )
    ax.axvline(38, color=INK, linestyle="--", linewidth=0.9, alpha=0.55, zorder=1)
    ax.set_title("(a) Model 1", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Training step (thousands)", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Validation perplexity", fontsize=8.5, fontweight="bold")
    ax.set_xlim(37.4, 51.4)
    ax.set_ylim(18.9, 21.85)
    style_axis(ax)

    ax = axes[1]
    ax.plot(
        main_steps,
        main_ppl,
        color=BLUE,
        marker="o",
        markersize=3.4,
        linewidth=2.0,
        label="1st epoch",
        zorder=3,
    )
    ax.plot(
        model2_cont_steps,
        model2_cont_ppl,
        color=RED,
        marker="o",
        markersize=2.1,
        markeredgewidth=0,
        linewidth=1.7,
        label="2nd epoch",
        zorder=3,
    )
    ax.scatter([38], [21.1630], s=34, color=BLUE, edgecolor=INK, linewidth=0.8, zorder=5)
    ax.annotate(
        "epoch 1: 21.163",
        xy=(38, 21.1630),
        xytext=(27.0, 24.1),
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9},
    )
    ax.scatter([69.7], [18.7681], s=42, color=RED, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.annotate(
        "best 18.768",
        xy=(69.7, 18.7681),
        xytext=(58.5, 20.8),
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 0.9},
    )
    ax.axvline(38, color=INK, linestyle="--", linewidth=1.0, alpha=0.55, zorder=1)
    ax.text(
        39.2,
        31.5,
        "epoch boundary",
        fontsize=7.0,
        rotation=90,
        va="center",
        color="#4b5563",
    )
    ax.set_title("(b) Model 2", fontsize=9.2, fontweight="bold")
    ax.set_xlabel("Training step (thousands)", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 75)
    ax.set_ylim(17.5, 37.3)
    style_axis(ax)

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7.8,
        handlelength=1.9,
        ncol=1,
        labelspacing=0.35,
    )

    fig.savefig(REPORT_DIR / "training_curve.pdf", bbox_inches="tight")
    fig.savefig(REPORT_DIR / "training_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor,
    fontsize=8.2,
    linewidth=1.2,
    edgecolor=INK,
):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.18,
    )


def arrow(ax, start, end, color=BLUE):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.1,
            "color": color,
            "mutation_scale": 8,
        },
    )


def plot_architecture(
    filename: str,
    blocks: int,
    attention: str,
    attention_detail: str,
    width: int,
    mlp_width: int,
) -> None:
    """Draw one slide-aligned architecture panel with non-overlapping modules."""
    fig, ax = plt.subplots(figsize=(2.05, 3.52))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.20, 0.005), 0.60, 0.052, "Input tokens", LIGHT_GREEN, fontsize=7.2)
    add_box(
        ax,
        (0.13, 0.120),
        0.74,
        0.075,
        f"Token embedding\n50,257 x {width}",
        LIGHT_BLUE,
        fontsize=7.1,
        edgecolor=BLUE,
    )
    arrow(ax, (0.50, 0.075), (0.50, 0.095))

    outer = FancyBboxPatch(
        (0.07, 0.270),
        0.86,
        0.360,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=LIGHT_CREAM,
        edgecolor=BLUE,
        linewidth=1.7,
    )
    ax.add_patch(outer)
    ax.text(
        0.50,
        0.600,
        f"{blocks} x Transformer blocks",
        ha="center",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=BLUE,
    )
    add_box(
        ax,
        (0.13, 0.288),
        0.74,
        0.122,
        f"RMSNorm + {attention}\n{attention_detail}",
        LIGHT_BLUE,
        fontsize=6.55,
        linewidth=1.0,
        edgecolor=BLUE,
    )
    add_box(
        ax,
        (0.13, 0.490),
        0.74,
        0.088,
        f"RMSNorm + SwiGLU\n{width} -> {mlp_width} -> {width}",
        LIGHT_RED,
        fontsize=6.9,
        linewidth=1.0,
        edgecolor=RED,
    )
    arrow(ax, (0.50, 0.215), (0.50, 0.245))
    arrow(ax, (0.50, 0.430), (0.50, 0.468))

    add_box(
        ax,
        (0.18, 0.700),
        0.64,
        0.062,
        "Final RMSNorm",
        LIGHT_BLUE,
        fontsize=7.1,
        edgecolor=BLUE,
    )
    arrow(ax, (0.50, 0.650), (0.50, 0.678))
    add_box(
        ax,
        (0.15, 0.820),
        0.70,
        0.068,
        "LM head (tied)",
        LIGHT_PURPLE,
        fontsize=7.2,
        edgecolor="#7c3aed",
    )
    arrow(ax, (0.50, 0.782), (0.50, 0.798))
    add_box(ax, (0.20, 0.940), 0.60, 0.052, "Output logits", LIGHT_GREEN, fontsize=7.2)
    arrow(ax, (0.50, 0.908), (0.50, 0.918))

    fig.savefig(REPORT_DIR / f"{filename}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(REPORT_DIR / f"{filename}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_training_curve()
    plot_architecture(
        "model1_architecture",
        blocks=13,
        attention="MHA",
        attention_detail="10 query / 10 KV\nRoPE + QK norm",
        width=640,
        mlp_width=1728,
    )
    plot_architecture(
        "model2_architecture",
        blocks=16,
        attention="GQA",
        attention_detail="10 query / 5 KV\nRoPE",
        width=640,
        mlp_width=1536,
    )
    plot_architecture(
        "model3_architecture",
        blocks=20,
        attention="GQA",
        attention_detail="9 query / 3 KV\nRoPE + QK norm",
        width=576,
        mlp_width=1536,
    )
    print("Generated one training curve and three architecture panels.")


if __name__ == "__main__":
    main()
