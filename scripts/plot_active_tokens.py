"""Plot train/active_loss_tokens vs training step for each variant from a W&B CSV export."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).parent.parent / "wandb_export_2026-05-31T00_41_10.858-07_00.csv"
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

    fig, ax = plt.subplots(figsize=(9, 5))

    for variant, label in VARIANT_LABELS.items():
        if variant not in series:
            continue
        data = series[variant].sort_values(step_col)
        ax.plot(
            data[step_col],
            data["tokens"],
            label=label,
            color=COLORS[variant],
            linewidth=1.8,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Active Loss Tokens", fontsize=12)
    ax.set_title("Active Loss Tokens vs Training Step", fontsize=13)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
