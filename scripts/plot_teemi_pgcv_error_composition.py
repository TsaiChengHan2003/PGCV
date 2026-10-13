#!/usr/bin/env python3
"""Plot TEEMI PGCV substitution/deletion/insertion stacked bars."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PAPER_IMAGE_DIR = PROJECT_ROOT / "analysis"

LEVELS = ["pre-A", "A1", "A1A2", "A2", "A2B1", "B1", "B1B2", "B2"]
DISPLAY = {
    "pre-A": "pre-A",
    "A1": "A1",
    "A1A2": "A1+",
    "A2": "A2",
    "A2B1": "A2+",
    "B1": "B1",
    "B1B2": "B1+",
    "B2": "B2",
}
PATHS = {
    "whisper_base": PROJECT_ROOT
    / "results/teemi/a01_only/pgcv/whisper_base/eval_summary.json",
    "qwen3_asr_0_6b": PROJECT_ROOT
    / "results/teemi/a01_only/pgcv/qwen3_asr_0_6b/eval_summary.json",
}
SPLITS = {
    "test": "A01",
    "test_a02_external": "A02",
}
SUB_COLOR = "#65c2a6"
DEL_COLOR = "#fc8e62"
INS_COLOR = "#8da0cb"
METHODS = [("theta_ft", "Full fine-tuning"), ("pgcv", "PGCV")]
SEGMENT_GAP = 0.28


def configure_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "TeX Gyre Termes",
                "Liberation Serif",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows() -> pd.DataFrame:
    rows = []
    for slug, path in PATHS.items():
        payload = json.loads(path.read_text())
        for split_key, booklet in SPLITS.items():
            for method_key, method_name in METHODS:
                by_level = payload[split_key][method_key]["by_level"]
                for level in LEVELS:
                    metrics = by_level[level]
                    rows.append(
                        {
                            "model": slug,
                            "booklet": booklet,
                            "method": method_name,
                            "level": level,
                            "display": DISPLAY[level],
                            "sub": float(metrics["sub"]),
                            "del": float(metrics["del"]),
                            "ins": float(metrics["ins"]),
                            "wer": float(metrics["wer"]),
                            "n_ref_words": int(metrics["n_ref_words"]),
                        }
                    )
    return pd.DataFrame(rows)


def add_stack(ax, x, frame: pd.DataFrame, width: float, labeled: bool) -> None:
    sub = frame["sub"].to_numpy()
    deletion = frame["del"].to_numpy()
    insertion = frame["ins"].to_numpy()
    ax.bar(
        x,
        sub,
        width=width,
        color=SUB_COLOR,
        edgecolor="0.18",
        linewidth=0.35,
        label="Substitution" if labeled else None,
        zorder=2,
    )
    ax.bar(
        x,
        deletion,
        width=width,
        bottom=sub + SEGMENT_GAP,
        color=DEL_COLOR,
        edgecolor="0.18",
        linewidth=0.35,
        label="Deletion" if labeled else None,
        zorder=2,
    )
    ax.bar(
        x,
        insertion,
        width=width,
        bottom=sub + deletion + 2 * SEGMENT_GAP,
        color=INS_COLOR,
        edgecolor="0.18",
        linewidth=0.35,
        label="Insertion" if labeled else None,
        zorder=2,
    )


def annotate_segments(ax, x, frame: pd.DataFrame) -> None:
    sub = frame["sub"].to_numpy()
    deletion = frame["del"].to_numpy()
    insertion = frame["ins"].to_numpy()
    wer = frame["wer"].to_numpy()
    bottoms = [
        np.zeros_like(sub),
        sub + SEGMENT_GAP,
        sub + deletion + 2 * SEGMENT_GAP,
    ]
    values = [sub, deletion, insertion]
    for bottom, vals in zip(bottoms, values):
        for xi, base, val in zip(x, bottom, vals):
            ax.text(
                xi,
                base + 0.5 * val,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=4.0,
                color="0.08",
                zorder=3,
            )
    for xi, total in zip(x, wer):
        ax.text(
            xi,
            total + 2 * SEGMENT_GAP + 0.45,
            f"{total:.1f}%",
            ha="center",
            va="bottom",
            fontsize=4.6,
            color="0.18",
            zorder=5,
        )


def method_positions(n_levels: int, group_gap: float) -> list[np.ndarray]:
    return [
        np.arange(n_levels, dtype=np.float64),
        np.arange(n_levels, dtype=np.float64) + n_levels + group_gap,
    ]


def plot_a01_column(frame: pd.DataFrame, stems: list[Path]) -> None:
    configure_pub_style()
    block = frame[(frame["model"] == "whisper_base") & (frame["booklet"] == "A01")].copy()
    methods = ["Full fine-tuning", "PGCV"]
    n_levels = len(LEVELS)
    group_gap = 1.55
    width = 0.86
    xs = method_positions(n_levels, group_gap)
    fig, ax = plt.subplots(figsize=(3.28, 2.78), facecolor="white")
    ax.set_facecolor("white")
    for i, (method, x) in enumerate(zip(methods, xs)):
        panel = block[block["method"] == method].set_index("level").loc[LEVELS].reset_index()
        add_stack(ax, x, panel, width, labeled=(i == 0))
        annotate_segments(ax, x, panel)
    all_x = np.concatenate(xs)
    ax.set_xticks(all_x)
    ax.set_xticklabels(
        [DISPLAY[level] for _ in methods for level in LEVELS],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontsize=5.2,
    )
    ax.tick_params(axis="both", length=1.5, width=0.4, pad=0.5, color="0.25", labelsize=5.8)
    ax.set_ylabel("WER (%)", fontsize=7.0, labelpad=1.4)
    ax.set_xlim(xs[0][0] - 0.70, xs[1][-1] + 0.70)
    ax.set_ylim(0.0, 38.2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.35, color="0.88")
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.18")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for x, name in zip(xs, methods):
        ax.text(
            float(np.mean(x)),
            -0.20,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.1,
            color="0.12",
            clip_on=False,
        )
    legend = ax.legend(
        title="Error Type",
        loc="upper center",
        bbox_to_anchor=(0.52, 1.16),
        ncol=3,
        frameon=False,
        fontsize=5.0,
        title_fontsize=5.5,
        handlelength=0.95,
        handleheight=0.65,
        handletextpad=0.28,
        columnspacing=1.20,
        borderaxespad=0.0,
        borderpad=0.0,
        labelspacing=0.18,
    )
    legend.get_title().set_fontweight("normal")
    legend.get_title().set_color("0.15")
    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.26)
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    frame = load_rows()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    a01 = frame[(frame["model"] == "whisper_base") & (frame["booklet"] == "A01")].copy()
    a01.to_csv(ANALYSIS_DIR / "teemi_a01_pgcv_error_composition.csv", index=False)
    stems = [
        ANALYSIS_DIR / "teemi_a01_pgcv_error_composition",
        PAPER_IMAGE_DIR / "teemi_a01_pgcv_error_composition",
    ]
    plot_a01_column(frame, stems)
    print("Wrote TEEMI A01 Full FT vs PGCV grouped error-composition figure.")


if __name__ == "__main__":
    main()
