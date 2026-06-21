from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PathLike = Union[str, Path]

ACCESSIBLE_RCPARAMS = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "lines.linewidth": 2.0,
    "font.size": 11,
}


def write_alt_text_md(png_path: PathLike, metadata: Dict[str, Any]) -> Path:
    """Write a sibling markdown file documenting plot accessibility metadata."""
    png_path = Path(png_path)
    md_path = png_path.with_suffix(".md")

    lines = [
        f"# Alt Text: {png_path.name}",
        "",
        "## Visual Attributes",
    ]
    for key, value in metadata.get("visual_attributes", {}).items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(["", "## Statistical Parameters"])
    for key, value in metadata.get("statistics", {}).items():
        lines.append(f"- **{key}**: {value}")

    conclusion = metadata.get("conclusion", "No conclusion provided.")
    lines.extend(["", "## Qualitative Conclusion", "", str(conclusion), ""])

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def plot_latency_cliff(results: Dict[int, Dict[str, Any]], out_path: PathLike) -> Path:
    """Plot TTFT and mean decode latency vs prompt length."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_lengths = sorted(results.keys())
    ttft_values = [results[length]["ttft"] for length in prompt_lengths]
    decode_means = [
        float(np.mean(results[length]["latencies"])) if results[length]["latencies"] else 0.0
        for length in prompt_lengths
    ]

    with plt.rc_context(ACCESSIBLE_RCPARAMS):
        palette = sns.color_palette("colorblind", 2)
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            prompt_lengths,
            ttft_values,
            marker="s",
            linestyle="-",
            color=palette[0],
            label="TTFT (prefill)",
        )
        ax.plot(
            prompt_lengths,
            decode_means,
            marker="o",
            linestyle="--",
            color=palette[1],
            label="Mean decode latency",
        )

        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("Latency (seconds)")
        ax.set_title("Latency Profile vs Prompt Length")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    write_alt_text_md(
        out_path,
        {
            "visual_attributes": {
                "x_axis": "Prompt length (tokens, log2 scale)",
                "y_axis": "Latency in seconds",
                "markers": "squares (TTFT), circles (decode mean)",
            },
            "statistics": {
                "prompt_lengths": prompt_lengths,
                "ttft_values": [round(v, 4) for v in ttft_values],
                "decode_means": [round(v, 4) for v in decode_means],
            },
            "conclusion": metadata_conclusion_for_latency(ttft_values, decode_means),
        },
    )
    return out_path


def plot_metric_bars(
    metrics: Dict[str, float],
    ci_bounds: Optional[Dict[str, Dict[str, float]]] = None,
    out_path: PathLike = "results/plots/metrics.png",
    title: str = "Evaluation Metrics",
) -> Path:
    """Plot bar chart with optional bootstrap confidence error bars."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    errors: List[float] = []

    if ci_bounds:
        for label in labels:
            bounds = ci_bounds.get(label, {})
            mean = bounds.get("mean", metrics[label])
            lower = bounds.get("lower", mean)
            upper = bounds.get("upper", mean)
            errors.append(max(mean - lower, upper - mean))
    else:
        errors = [0.0] * len(labels)

    with plt.rc_context(ACCESSIBLE_RCPARAMS):
        palette = sns.color_palette("colorblind", len(labels))
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=4, color=palette, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    write_alt_text_md(
        out_path,
        {
            "visual_attributes": {
                "chart_type": "bar chart with error bars",
                "labels": labels,
            },
            "statistics": {"metrics": metrics, "ci_bounds": ci_bounds or {}},
            "conclusion": f"Bar chart comparing {len(labels)} metrics with bootstrap bounds.",
        },
    )
    return out_path


def metadata_conclusion_for_latency(ttft_values: List[float], decode_means: List[float]) -> str:
    if len(ttft_values) < 2:
        return "Insufficient data to assess latency cliffs."

    ttft_growth = ttft_values[-1] / max(ttft_values[0], 1e-9)
    decode_growth = decode_means[-1] / max(decode_means[0], 1e-9)

    if ttft_growth > 2.0:
        return (
            f"TTFT increases {ttft_growth:.1f}x across prompt lengths, "
            "suggesting a prefill latency cliff."
        )
    if decode_growth > 1.5:
        return f"Decode latency grows {decode_growth:.1f}x, indicating decode-stage pressure."
    return "Latency scales relatively smoothly across tested prompt lengths."
