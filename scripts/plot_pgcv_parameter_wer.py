#!/usr/bin/env python3
"""Plot PGCV development WER over beta and tau for both ASR backbones."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/teemi/a01_only/pgcv"
MODEL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("whisper_base", "Whisper-base"),
    ("qwen3_asr_0_6b", "Qwen3-ASR-0.6B"),
)
COLORS: Dict[float, str] = {
    0.00: "#4C78A8",
    0.25: "#E45756",
    0.50: "#72B7B2",
    0.75: "#F2CF5B",
    1.00: "#B279A2",
}


def load_curve(results_root: Path, model_slug: str) -> pd.DataFrame:
    path = results_root / model_slug / "lambda_curve_pgcv.csv"
    frame = pd.read_csv(path)
    required = {"beta", "lambda", "wer"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    return frame.sort_values(["beta", "lambda"]).reset_index(drop=True)


def plot_parameter_wer(results_root: Path, output_stem: Path) -> Tuple[Path, Path]:
    frames = {slug: load_curve(results_root, slug) for slug, _ in MODEL_SPECS}
    fig = plt.figure(figsize=(8.0, 3.7))
    outer = fig.add_gridspec(1, 2, wspace=0.20)
    axes = []
    for column in range(2):
        inner = outer[column].subgridspec(
            2, 1, height_ratios=(5.0, 0.55), hspace=0.08
        )
        upper = fig.add_subplot(inner[0])
        lower = fig.add_subplot(inner[1], sharex=upper)
        axes.append((upper, lower))

    y_axis_specs = {
        "whisper_base": ((11.5, 22.5), [12, 14, 16, 18, 20, 22]),
        "qwen3_asr_0_6b": ((8.0, 10.0), [8.0, 8.5, 9.0, 9.5, 10.0]),
    }

    for (ax, lower_ax), (model_slug, model_label) in zip(axes, MODEL_SPECS):
        frame = frames[model_slug]
        best = frame.loc[frame["wer"].idxmin()]
        lambda_values = sorted(float(value) for value in frame["lambda"].unique())
        lambda_positions = {value: index for index, value in enumerate(lambda_values)}
        for beta in sorted(frame["beta"].unique()):
            sub = frame[frame["beta"] == beta]
            beta_value = float(beta)
            selected = beta_value == 0.25
            global_only = beta_value == 0.0
            x_positions = [lambda_positions[float(value)] for value in sub["lambda"]]
            ax.plot(
                x_positions,
                sub["wer"],
                marker="o",
                markersize=4.2 if selected else 3.4,
                linewidth=2.2 if selected else 1.25,
                linestyle="--" if global_only else "-",
                color=COLORS.get(beta_value, "#777777"),
                alpha=1.0 if selected or global_only else 0.78,
                label=f"{beta_value:g}",
                zorder=3 if selected else 2,
            )

        best_x = lambda_positions[float(best["lambda"])]
        ax.scatter(
            [best_x],
            [float(best["wer"])],
            marker="*",
            s=115,
            color="#111111",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        panel_label = "(a)" if model_slug == "whisper_base" else "(b)"
        ax.set_title(f"{panel_label} {model_label}")
        ax.grid(True, alpha=0.25, linewidth=0.7)
        y_limits, y_ticks = y_axis_specs[model_slug]
        ax.set_ylim(*y_limits)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="both", labelsize=8.5)

        lower_ax.set_ylim(0.0, 0.5)
        lower_ax.set_yticks([0])
        lower_ax.set_xticks(range(len(lambda_values)))
        lower_ax.set_xticklabels([f"{value:.2f}" for value in lambda_values])
        lower_ax.tick_params(axis="both", labelsize=8.5)
        lower_ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        break_size = 0.014
        break_style = {"color": "black", "clip_on": False, "linewidth": 0.9}
        ax.plot(
            (-break_size, +break_size),
            (-break_size, +break_size),
            transform=ax.transAxes,
            **break_style,
        )
        ax.plot(
            (1 - break_size, 1 + break_size),
            (-break_size, +break_size),
            transform=ax.transAxes,
            **break_style,
        )
        lower_ax.plot(
            (-break_size, +break_size),
            (1 - break_size, 1 + break_size),
            transform=lower_ax.transAxes,
            **break_style,
        )
        lower_ax.plot(
            (1 - break_size, 1 + break_size),
            (1 - break_size, 1 + break_size),
            transform=lower_ax.transAxes,
            **break_style,
        )

    fig.supxlabel(r"Correction strength $\tau$", y=0.035, fontsize=11)
    fig.supylabel("A01 dev WER (%)", x=0.018, fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.25,
        handletextpad=0.45,
        title=r"Composition weight $\beta$",
        title_fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.80)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot A01 development WER over the PGCV beta/tau grid."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing whisper_base and qwen3_asr_0_6b results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "figures/pgcv_parameter_wer",
        help="Output path without an extension.",
    )
    args = parser.parse_args()
    results_root = args.results_root
    output = args.output
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    pdf_path, png_path = plot_parameter_wer(results_root, output)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
