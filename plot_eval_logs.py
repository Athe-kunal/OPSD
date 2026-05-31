import re
import os
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

METRICS = ["Pass@8", "Avg@8", "MajVote@8"]
BENCHMARKS = ["AIME24", "AIME25", "HMMT25"]

METRIC_KEYS = {
    "Pass@8": r"Pass@8:\s*([\d.]+)%",
    "Avg@8":  r"Avg@8:\s*([\d.]+)%",
    "MajVote@8": r"MajVote@8:\s*([\d.]+)%",
}


def parse_log(path: Path) -> dict:
    """Return {benchmark: {metric: value}} from a single log file."""
    text = path.read_text()
    results = {}
    # Find the summary block
    block_match = re.search(
        r"ALL EVALUATIONS COMPLETE\n={10,}(.*?)={10,}", text, re.DOTALL
    )
    if not block_match:
        return results
    block = block_match.group(1)
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        bench_match = re.match(r"(\w+)\s+", line)
        if not bench_match:
            continue
        bench = bench_match.group(1)
        metrics = {}
        for metric, pattern in METRIC_KEYS.items():
            m = re.search(pattern, line)
            if m:
                metrics[metric] = float(m.group(1))
        if metrics:
            results[bench] = metrics
    return results


EXCLUDE_VARIANTS = {"firstsent_topk2", "lastsent_topk2"}


def shorten_name(name: str) -> str:
    """Create a readable label from the experiment folder name."""
    prefix = "qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"
    label = name.replace(prefix, "").lstrip("_")
    return label if label else "baseline"


def collect_data(logs_dir: Path) -> dict:
    """
    Returns:
        data[algo][benchmark][metric] = sorted list of (step, value)
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        algo = shorten_name(exp_dir.name)
        if algo in EXCLUDE_VARIANTS:
            continue
        for log_file in sorted(exp_dir.glob("step*-thinking.log")):
            step_match = re.search(r"step(\d+)", log_file.name)
            if not step_match:
                continue
            step = int(step_match.group(1))
            parsed = parse_log(log_file)
            for bench, metrics in parsed.items():
                for metric, value in metrics.items():
                    data[algo][bench][metric].append((step, value))

    # Sort by step
    for algo in data:
        for bench in data[algo]:
            for metric in data[algo][bench]:
                data[algo][bench][metric].sort(key=lambda x: x[0])

    return data


LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_metric(data: dict, metric: str, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    algos = list(data.keys())
    palette = cm.get_cmap("tab10")
    colors = [palette(i / max(len(algos) - 1, 1)) for i in range(len(algos))]

    fig, axes = plt.subplots(
        1, len(BENCHMARKS),
        figsize=(7 * len(BENCHMARKS), 6),
        sharey=False,
    )
    if len(BENCHMARKS) == 1:
        axes = [axes]

    for ax, bench in zip(axes, BENCHMARKS):
        for i, (algo, color) in enumerate(zip(algos, colors)):
            points = data[algo].get(bench, {}).get(metric, [])
            if not points:
                continue
            steps, values = zip(*points)
            ls = LINE_STYLES[i % len(LINE_STYLES)]
            mk = MARKERS[i % len(MARKERS)]
            ax.plot(
                steps, values,
                linestyle=ls,
                marker=mk,
                markersize=7,
                linewidth=2.0,
                color=color,
                label=algo,
            )

        all_steps = sorted({s for algo in algos for s, _ in data[algo].get(bench, {}).get(metric, [])})
        ax.set_xticks(all_steps)
        ax.set_title(bench, fontweight="bold", pad=8)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(f"{metric} (%)")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6, color="gray")
        ax.set_axisbelow(True)

    # Gather legend entries from all axes (some variants may be missing from axes[0])
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)

    n_cols = len(handles)  # one row spanning full width
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=n_cols,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="lightgray",
        fontsize=13,
        handlelength=2.5,
        handletextpad=0.6,
        columnspacing=1.2,
        borderpad=0.8,
        markerscale=1.4,
    )

    fig.suptitle(metric, fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0.13, 1, 0.97])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot eval logs by step.")
    parser.add_argument("logs_dir", type=Path, help="Path to the logs folder")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for plots (default: logs_dir)")
    args = parser.parse_args()

    out_dir = args.out_dir or args.logs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect_data(args.logs_dir)

    if not data:
        print("No data found. Check that log files contain the expected summary block.")
        return

    for metric in METRICS:
        safe_name = metric.replace("@", "at").replace("/", "_")
        plot_metric(data, metric, out_dir / f"plot_{safe_name}.png")


if __name__ == "__main__":
    main()
