#!/usr/bin/env python3
"""Plot TEEMI / S&I mean proficiency-affinity heatmaps."""

from __future__ import annotations

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

TEEMI = {
    "levels": ["pre-A", "A1", "A1A2", "A2", "A2B1", "B1", "B1B2", "B2"],
    "display": {
        "pre-A": "pre-A",
        "A1": "A1",
        "A1A2": "A1+",
        "A2": "A2",
        "A2B1": "A2+",
        "B1": "B1",
        "B1B2": "B1+",
        "B2": "B2",
    },
    "alpha": [
        "alpha_pre_a",
        "alpha_a1",
        "alpha_a1a2",
        "alpha_a2",
        "alpha_a2b1",
        "alpha_b1",
        "alpha_b1b2",
        "alpha_b2",
    ],
    "paths": {
        "whisper_base": PROJECT_ROOT
        / "data/teemi/a01_only/pgcv/whisper_base/assignment_test.csv",
        "qwen3_asr_0_6b": PROJECT_ROOT
        / "data/teemi/a01_only/pgcv/qwen3_asr_0_6b/assignment_test.csv",
    },
}

SANDI = {
    "levels": ["A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+"],
    "display": {level: level for level in ["A2", "A2+", "B1", "B1+", "B2", "B2+", "C1", "C1+"]},
    "alpha": [
        "alpha_a2",
        "alpha_a2_plus",
        "alpha_b1",
        "alpha_b1_plus",
        "alpha_b2",
        "alpha_b2_plus",
        "alpha_c1",
        "alpha_c1_plus",
    ],
    "paths": {
        "whisper_base": PROJECT_ROOT
        / "data/sandi/overall_cefr/pgcv/whisper_base/assignment_eval.csv",
        "qwen3_asr_0_6b": PROJECT_ROOT
        / "data/sandi/overall_cefr/pgcv/qwen3_asr_0_6b/assignment_eval.csv",
    },
}

MODEL_LABEL = {
    "whisper_base": "Whisper-base",
    "qwen3_asr_0_6b": "Qwen3-ASR-0.6B",
}


def mean_affinity_matrix(path: Path, spec: dict) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"proficiency_level", *spec["alpha"]} - set(frame.columns)
    if missing:
        raise KeyError(f"{path} missing columns {sorted(missing)}")
    display_order = [spec["display"][level] for level in spec["levels"]]
    gold = frame["proficiency_level"].map(spec["display"])
    rows = []
    for level in spec["levels"]:
        mask = gold == spec["display"][level]
        block = frame.loc[mask, spec["alpha"]]
        if len(block) == 0:
            rows.append([np.nan] * len(spec["levels"]))
        else:
            rows.append(block.mean(axis=0).to_numpy(dtype=np.float64))
    return pd.DataFrame(rows, index=display_order, columns=display_order)


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


def cell_text_color(value: float, vmax: float, cmap) -> str:
    red, green, blue, _ = cmap(value / max(vmax, 1e-8))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "0.08"


def annotate_cells(ax, values: np.ndarray, fontsize: float, vmax: float | None = None, cmap=None) -> None:
    if vmax is None:
        vmax = 0.45
        cmap = plt.cm.Blues
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            color = cell_text_color(val, vmax, cmap) if cmap is not None else ("white" if val >= 0.45 else "black")
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                fontweight="normal",
            )


def draw_panel(ax, matrix: pd.DataFrame, title: str, vmax: float, fontsize: float):
    values = matrix.to_numpy(dtype=np.float64)
    labels = list(matrix.columns)
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Proficiency-specific correction")
    ax.set_ylabel("Gold proficiency")
    ax.set_title(title)
    annotate_cells(ax, values, fontsize, vmax=vmax, cmap=plt.cm.Blues)
    return image


def style_heatmap_axes(
    ax,
    labels: list[str],
    show_yticks: bool,
    show_xticks: bool,
    ticksize: float,
) -> None:
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if show_xticks:
        ax.set_xticklabels(
            labels,
            rotation=38,
            ha="right",
            rotation_mode="anchor",
            fontsize=ticksize,
        )
        ax.tick_params(axis="x", which="major", length=1.5, width=0.4, pad=0.7, color="0.25")
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="major", length=0, labelbottom=False)
    if show_yticks:
        ax.set_yticklabels(labels, fontsize=ticksize)
        ax.tick_params(axis="y", which="major", length=1.5, width=0.4, pad=0.7, color="0.25")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="major", length=1.5, width=0.4, labelleft=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.18")


def draw_paper_panel(
    ax,
    matrix: pd.DataFrame,
    vmax: float,
    cmap,
    show_yticks: bool,
    show_xticks: bool,
):
    values = matrix.to_numpy(dtype=np.float64)
    labels = list(matrix.columns)
    n = len(labels)
    image = ax.pcolormesh(
        np.arange(n + 1) - 0.5,
        np.arange(n + 1) - 0.5,
        values,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
    )
    ax.set_aspect("auto")
    style_heatmap_axes(
        ax,
        labels,
        show_yticks=show_yticks,
        show_xticks=show_xticks,
        ticksize=5.4,
    )
    annotate_cells(ax, values, fontsize=5.5, vmax=vmax, cmap=cmap)
    return image


def save_figure(fig, stems: list[Path], pad_inches: float = 0.04) -> None:
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )
        fig.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=pad_inches,
        )


def plot_grid(panels: list[tuple[str, pd.DataFrame]], nrows: int, ncols: int, stems: list[Path]) -> None:
    vmax = max(0.5, max(float(np.nanmax(matrix.to_numpy())) for _, matrix in panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.5 * nrows))
    fig.patch.set_facecolor("white")
    axes_flat = np.atleast_1d(axes).ravel()
    image = None
    fontsize = 7.0 if len(panels) <= 2 else 6.2
    for ax, (title, matrix) in zip(axes_flat, panels):
        ax.set_facecolor("white")
        image = draw_panel(ax, matrix, title, vmax, fontsize)
    fig.subplots_adjust(right=0.90, wspace=0.28, hspace=0.38)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    fig.colorbar(image, cax=cax)
    save_figure(fig, stems)
    plt.close(fig)


def plot_paper_four(
    teemi: dict[str, pd.DataFrame],
    sandi: dict[str, pd.DataFrame],
    stems: list[Path],
) -> None:
    configure_pub_style()
    grids = [
        [teemi["whisper_base"], teemi["qwen3_asr_0_6b"]],
        [sandi["whisper_base"], sandi["qwen3_asr_0_6b"]],
    ]
    col_headers = ["Whisper-base", "Qwen3-ASR-0.6B"]
    row_headers = ["TEEMI", "Speak &\nImprove\nCorpus"]
    vmax = max(
        0.5,
        max(float(np.nanmax(matrix.to_numpy())) for row in grids for matrix in row),
    )
    cmap = plt.cm.Blues

    fig = plt.figure(figsize=(6.90, 4.32), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.142,
        right=0.932,
        top=0.962,
        bottom=0.118,
        wspace=0.024,
        hspace=0.128,
    )
    axes = np.empty((2, 2), dtype=object)
    image = None
    for i, row in enumerate(grids):
        for j, matrix in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            ax.set_facecolor("white")
            image = draw_paper_panel(
                ax,
                matrix,
                vmax=vmax,
                cmap=cmap,
                show_yticks=(j == 0),
                show_xticks=True,
            )
            ax.set_title("")
            axes[i, j] = ax

    top_left = axes[0, 0].get_position()
    top_right = axes[0, 1].get_position()
    bot_left = axes[1, 0].get_position()
    bot_right = axes[1, 1].get_position()

    for j, name in enumerate(col_headers):
        pos = axes[0, j].get_position()
        fig.text(
            0.5 * (pos.x0 + pos.x1),
            pos.y1 + 0.004,
            name,
            ha="center",
            va="bottom",
            fontsize=7.4,
        )

    for i, name in enumerate(row_headers):
        pos = axes[i, 0].get_position()
        fig.text(
            pos.x0 - 0.050,
            0.5 * (pos.y0 + pos.y1),
            name,
            ha="center",
            va="center",
            fontsize=6.6,
            color="0.28",
            linespacing=1.12,
        )

    block_height = top_right.y1 - bot_right.y0
    cbar_inset = 0.06 * block_height
    cax = fig.add_axes(
        [
            top_right.x1 + 0.006,
            bot_right.y0 + cbar_inset,
            0.006,
            block_height - 2.0 * cbar_inset,
        ]
    )
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Mean affinity", fontsize=6.8, labelpad=1.4)
    cbar.ax.tick_params(labelsize=5.6, length=1.4, width=0.35, pad=0.8, color="0.25")
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("0.18")

    fig.text(
        0.5 * (bot_left.x0 + bot_right.x1),
        0.016,
        "Proficiency-specific correction",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    fig.text(
        0.014,
        0.5 * (bot_left.y0 + top_left.y1),
        "Gold proficiency",
        ha="center",
        va="center",
        rotation=90,
        fontsize=7.2,
    )

    save_figure(fig, stems, pad_inches=0.006)
    plt.close(fig)


def main() -> None:
    teemi = {
        slug: mean_affinity_matrix(path, TEEMI) for slug, path in TEEMI["paths"].items()
    }
    sandi = {
        slug: mean_affinity_matrix(path, SANDI) for slug, path in SANDI["paths"].items()
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for slug, matrix in sandi.items():
        matrix.to_csv(ANALYSIS_DIR / f"affinity_matrix_sandi_{slug}.csv", index_label="gold_proficiency")

    plot_grid(
        [
            (f"(a) {MODEL_LABEL['whisper_base']}", sandi["whisper_base"]),
            (f"(b) {MODEL_LABEL['qwen3_asr_0_6b']}", sandi["qwen3_asr_0_6b"]),
        ],
        nrows=1,
        ncols=2,
        stems=[
            ANALYSIS_DIR / "affinity_heatmap_sandi",
            PAPER_IMAGE_DIR / "affinity_heatmap_sandi",
        ],
    )
    paper_stems = [
        ANALYSIS_DIR / "affinity_heatmap_teemi_sandi",
        PAPER_IMAGE_DIR / "affinity_heatmap_teemi_sandi",
        PAPER_IMAGE_DIR / "affinity_heatmap",
        ANALYSIS_DIR / "affinity_heatmap_compact",
        PAPER_IMAGE_DIR / "affinity_heatmap_compact",
    ]
    plot_paper_four(teemi, sandi, paper_stems)
    print("Wrote S&I 2-panel and paper 4-panel affinity heatmaps.")


if __name__ == "__main__":
    main()
