"""Plot train/active_loss_tokens vs training step for each variant from a W&B CSV export."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_repo = Path(__file__).parent.parent
_candidates = sorted(_repo.glob("wandb_export*.csv"))
CSV_PATH = _candidates[0] if _candidates else _repo / "wandb_export_2026-05-31T00_41_10.858-07_00.csv"
OUT_PATH = Path(__file__).parent.parent / "assets" / "plot_active_tokens.png"

# Map run config slug → short display label
VARIANT_LABELS = {
    "lastsent_topk2":    "lastsent_topk2",
    "firstsent_topk2":   "firstsent_topk2",
    "parafirsttok_topk1":"parafirsttok_topk1",
    "lastsent_topk1":    "lastsent_topk1",
    "middlesent_topk1":  "middlesent_topk1",
    "firstsent_topk1":   "firstsent_topk1",
    "baseline":          "baseline",
}

# Colors consistent with the existing eval plots
COLORS = {
    "baseline":           "#1f77b4",
    "firstsent_topk1":    "#ff7f0e",
    "firstsent_topk2":    "#d62728",
    "lastsent_topk1":     "#9467bd",
    "lastsent_topk2":     "#8c564b",
    "middlesent_topk1":   "#e377c2",
    "parafirsttok_topk1": "#17becf",
}


def identify_variant(run_name: str) -> str:
    """Return a canonical variant key from a W&B run name column header."""
    for key in ["lastsent_topk2", "firstsent_topk2", "parafirsttok_topk1",
                "lastsent_topk1", "middlesent_topk1", "firstsent_topk1"]:
        if key in run_name:
            return key
    # The full-token baseline has no position keyword
    if re.search(r"clip005_lr", run_name):
        return "baseline"
    return None


def main():
    df = pd.read_csv(CSV_PATH)
    step_col = "train/global_step"

    # Collect one active_loss_tokens column per variant (skip __MIN / __MAX)
    series = {}
    for col in df.columns:
        if "active_loss_tokens" not in col:
            continue
        if col.endswith("__MIN") or col.endswith("__MAX"):
            continue
        run_part = col.split(" - ")[0].strip()
        variant = identify_variant(run_part)
        if variant is None or variant in series:
            continue
        series[variant] = df[[step_col, col]].dropna().rename(columns={col: "tokens"})

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    for variant, label in VARIANT_LABELS.items():
        if variant not in series:
            continue
        d = series[variant].sort_values(step_col)
        ax.plot(
            d[step_col],
            d["tokens"],
            label=label,
            color=COLORS[variant],
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Active Loss Tokens")
    ax.set_title("Active Loss Tokens vs Training Step", fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6, color="gray")
    ax.set_axisbelow(True)

    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fontsize=11,
        handlelength=2.5,
        columnspacing=1.5,
    )
    ax.get_legend().remove() if ax.get_legend() else None
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
