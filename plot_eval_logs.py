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


EXCLUDE_VARIANTS = set()


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


def plot_metric(data: dict, metric: str, out_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    algos = list(data.keys())
    n_algos = len(algos)
    palette = cm.get_cmap("tab10")
    colors = [palette(i / 10) for i in range(n_algos)]

    all_steps = sorted({
        s
        for algo in algos
        for bench in BENCHMARKS
        for s, _ in data[algo].get(bench, {}).get(metric, [])
    })
    n_steps = len(all_steps)
    step_index = {s: i for i, s in enumerate(all_steps)}

    # bar layout: for each step group, n_algos bars packed together
    bar_width = 0.8 / n_algos
    group_centers = np.arange(n_steps)

    fig, axes = plt.subplots(
        1, len(BENCHMARKS),
        figsize=(5 * len(BENCHMARKS), 6),
        sharey=False,
    )
    if len(BENCHMARKS) == 1:
        axes = [axes]

    bar_handles = []
    for ax, bench in zip(axes, BENCHMARKS):
        for i, (algo, color) in enumerate(zip(algos, colors)):
            points = data[algo].get(bench, {}).get(metric, [])
            if not points:
                continue
            x_pos = []
            heights = []
            for step, value in points:
                gi = step_index[step]
                x_pos.append(group_centers[gi] + (i - n_algos / 2 + 0.5) * bar_width)
                heights.append(value)
            bars = ax.bar(
                x_pos, heights,
                width=bar_width * 0.9,
                color=color,
                label=algo,
                edgecolor="white",
                linewidth=0.4,
            )
            if ax is axes[0]:
                bar_handles.append(bars[0])

        ax.set_xticks(group_centers)
        ax.set_xticklabels([f"Step {s}" for s in all_steps], rotation=30, ha="right")
        ax.set_title(bench, fontweight="bold", pad=8)
        ax.set_ylabel(f"{metric} (%)")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.6, color="gray")
        ax.set_axisbelow(True)

    # Collect legend handles/labels across all axes
    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)

    n_cols = min(n_algos, 4)
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=n_cols,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="lightgray",
        fontsize=11,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.7,
    )

    fig.suptitle(metric, fontsize=15, fontweight="bold")
    n_legend_rows = (n_algos + n_cols - 1) // n_cols
    bottom_margin = 0.10 + 0.06 * n_legend_rows
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_stepwise_metrics(exp_dir: Path):
    """Print a table of stepwise metrics for a single experiment folder."""
    exp_dir = exp_dir.resolve()
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory.")
        return

    rows = []
    for log_file in sorted(exp_dir.glob("step*-thinking.log")):
        step_match = re.search(r"step(\d+)", log_file.name)
        if not step_match:
            continue
        step = int(step_match.group(1))
        parsed = parse_log(log_file)
        if parsed:
            rows.append((step, parsed))

    rows.sort(key=lambda x: x[0])

    if not rows:
        print(f"No parsed results found in {exp_dir}")
        return

    col_w = 10
    bench_col = 8
    header = f"{'Step':>6}  {'Bench':<{bench_col}}" + "".join(
        f"  {m:>{col_w}}" for m in METRICS
    )
    sep = "-" * len(header)
    print(f"\nExperiment: {exp_dir.name}")
    print(sep)
    print(header)
    print(sep)
    for step, parsed in rows:
        first = True
        for bench in BENCHMARKS:
            metrics = parsed.get(bench, {})
            if not metrics:
                continue
            step_str = str(step) if first else ""
            first = False
            row = f"{step_str:>6}  {bench:<{bench_col}}"
            for m in METRICS:
                val = metrics.get(m)
                cell = f"{val:.1f}%" if val is not None else "N/A"
                row += f"  {cell:>{col_w}}"
            print(row)
        if not first:
            print()
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Plot eval logs by step.")
    parser.add_argument("logs_dir", type=Path, help="Path to the logs folder")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for plots (default: logs_dir)")
    parser.add_argument("--print_metrics", type=Path, default=None,
                        help="Print stepwise metrics for a specific experiment folder and exit")
    args = parser.parse_args()

    if args.print_metrics:
        print_stepwise_metrics(args.print_metrics)
        return

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
