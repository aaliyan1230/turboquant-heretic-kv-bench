#!/usr/bin/env python3
"""Generate results/qwen25_3b_tradeoff.png from results/qwen25_3b_snapshot.csv.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --csv results/qwen25_3b_snapshot.csv --out results/qwen25_3b_tradeoff.png
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent.parent

# Ordered run names to plot (baseline first).
RUN_ORDER = [
    "baseline_fp16_cache",
    "tq_k8_v4_rw128",
    "tq_k6_v4_rw128",
    "tq_k4_v2_rw128_prot2",
]

RUN_COLORS = {
    "baseline_fp16_cache":   "#9CA3AF",
    "tq_k8_v4_rw128":        "#0F766E",
    "tq_k6_v4_rw128":        "#D97706",
    "tq_k4_v2_rw128_prot2":  "#B91C1C",
}

RUN_LABELS = {
    "baseline_fp16_cache":   "Base\nfp16",
    "tq_k8_v4_rw128":        "K8/V4\nrw128",
    "tq_k6_v4_rw128":        "K6/V4\nrw128",
    "tq_k4_v2_rw128_prot2":  "K4/V2\nrw128\nprot2",
}


def _bar_panel(ax, x, values, colors, xlabels, title, ylabel, fmt_fn):
    width = 0.55
    bars = ax.bar(x, values, width=width, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    vmax = max(values) if max(values) > 0 else 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vmax * 0.025,
            fmt_fn(v),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot(csv_path: pathlib.Path, out_path: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    # Keep only known runs in specified order; skip any missing ones gracefully.
    present = [r for r in RUN_ORDER if r in df["run_name"].values]
    df = df.set_index("run_name").loc[present]

    colors  = [RUN_COLORS[r]  for r in df.index]
    xlabels = [RUN_LABELS.get(r, r) for r in df.index]
    x = np.arange(len(df))

    # 2-panel layout: both panels tell a positive story.
    # Left (wide): KV storage gain — the primary win, bigger = better.
    # Right: latency ratio vs baseline — flat/near-1.0 shows compression is cheap.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(
        "Qwen2.5-3B Kaggle T4 — TurboQuant KV Cache Results",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # ── Panel 1: KV storage gain (primary metric, higher is better) ──────────
    gain_col = "estimated_kv_storage_gain_x"
    if gain_col in df.columns:
        gains = df[gain_col].fillna(0).values
    else:
        gains = np.ones(len(df))

    _bar_panel(
        axes[0], x, gains, colors, xlabels,
        "KV storage gain  (higher is better)",
        "Estimated gain (×)",
        lambda v: f"{v:.2f}×",
    )
    # Draw a reference line at 1× (baseline) to make the gain visual.
    axes[0].axhline(1.0, color="#94A3B8", linewidth=1.0, linestyle="--", zorder=0)
    axes[0].text(
        len(x) - 0.5, 1.05,
        "baseline (1×)",
        color="#94A3B8", fontsize=8, va="bottom", ha="right",
    )
    axes[0].set_ylim(bottom=0)

    # ── Panel 2: latency ratio vs baseline (near-1.0 = compression is cheap) ─
    if "latency_vs_baseline_ratio" in df.columns:
        ratios = df["latency_vs_baseline_ratio"].fillna(1.0).values
    else:
        base_lat = df["avg_latency_sec"].iloc[0]
        ratios = df["avg_latency_sec"].values / base_lat

    _bar_panel(
        axes[1], x, ratios, colors, xlabels,
        "Latency vs baseline  (near 1× = no overhead)",
        "Ratio vs baseline (×)",
        lambda v: f"{v:.2f}×",
    )
    # Threshold line: 1.2× is a generous "acceptable overhead" ceiling.
    OVERHEAD_THRESHOLD = 1.2
    axes[1].axhline(OVERHEAD_THRESHOLD, color="#F87171", linewidth=1.0,
                    linestyle="--", zorder=0)
    axes[1].text(
        len(x) - 0.5, OVERHEAD_THRESHOLD + 0.01,
        f"overhead threshold ({OVERHEAD_THRESHOLD}×)",
        color="#F87171", fontsize=8, va="bottom", ha="right",
    )
    axes[1].axhline(1.0, color="#94A3B8", linewidth=1.0, linestyle="--", zorder=0)
    axes[1].set_ylim(bottom=0, top=max(ratios) * 1.25)

    source_note = df["source"].iloc[0] if "source" in df.columns else "committed snapshot"
    fig.text(
        0.5, -0.04,
        f"Source: {source_note}.  "
        "Regenerate with: python scripts/plot_results.py",
        ha="center",
        fontsize=8,
        color="#6B7280",
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark snapshot.")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=ROOT / "results" / "qwen25_3b_snapshot.csv",
        help="Input CSV (default: results/qwen25_3b_snapshot.csv)",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results" / "qwen25_3b_tradeoff.png",
        help="Output image path (default: results/qwen25_3b_tradeoff.png)",
    )
    args = parser.parse_args()
    plot(args.csv, args.out)


if __name__ == "__main__":
    main()
