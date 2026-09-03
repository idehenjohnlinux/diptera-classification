#!/usr/bin/env python3
"""
Plot family and genus confusion matrices.

Each cell displays:
    count (row percentage)

Example:
    24 (3.1%)

Rows represent true labels.
Columns represent predicted labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

FAMILY_CSV = Path(
    "results/production/hierarchical/evaluation/"
    "family_confusion_matrix.csv"
)

GENUS_CSV = Path(
    "results/production/hierarchical/evaluation/"
    "genus_confusion_matrix.csv"
)

OUTPUT_DIR = Path(
    "results/figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load matrix
# ============================================================

def load_confusion_matrix(csv_path: Path) -> pd.DataFrame:
    """Load a confusion matrix from a CSV file."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Confusion matrix file not found: {csv_path}"
        )

    matrix = pd.read_csv(csv_path, index_col=0)

    # Clean possible prefixes in row and column names.
    matrix.index = (
        matrix.index.astype(str)
        .str.replace("true:", "", regex=False)
        .str.replace("True:", "", regex=False)
        .str.strip()
    )

    matrix.columns = (
        matrix.columns.astype(str)
        .str.replace("pred:", "", regex=False)
        .str.replace("Pred:", "", regex=False)
        .str.strip()
    )

    # Ensure all matrix values are numeric.
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)

    return matrix.astype(int)


# ============================================================
# Plot matrix
# ============================================================

def plot_confusion_matrix(
    matrix: pd.DataFrame,
    title: str,
    output_name: str,
    figsize: tuple[float, float],
    tick_fontsize: float,
    annotation_fontsize: float,
    annotate_zeros: bool = True,
) -> None:
    """
    Plot a blue confusion matrix with counts and row percentages.
    """

    counts = matrix.to_numpy(dtype=int)

    # Percentage of each true class assigned to each predicted class.
    row_totals = counts.sum(axis=1, keepdims=True)

    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    ) * 100

    fig, ax = plt.subplots(figsize=figsize)

    # Light grey figure background similar to the reference image.
    fig.patch.set_facecolor("#d9dce1")
    ax.set_facecolor("#d9dce1")

    image = ax.imshow(
        percentages,
        cmap="Blues",
        interpolation="nearest",
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    # --------------------------------------------------------
    # Axis configuration
    # --------------------------------------------------------

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels(
        matrix.columns,
        rotation=90,
        fontsize=tick_fontsize,
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=tick_fontsize,
    )

    ax.set_xlabel(
        "Rótulo previsto",
        fontsize=tick_fontsize + 2,
        labelpad=12,
    )

    ax.set_ylabel(
        "Rótulo verdadeiro",
        fontsize=tick_fontsize + 2,
        labelpad=12,
    )

    ax.set_title(
        title,
        fontsize=tick_fontsize + 4,
        pad=14,
    )

    # --------------------------------------------------------
    # Grid lines
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, counts.shape[1], 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, counts.shape[0], 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.6,
        alpha=0.45,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Cell annotations
    # --------------------------------------------------------

    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = counts[row, column]
            percentage = percentages[row, column]

            if not annotate_zeros and count == 0:
                continue

            label = f"{count} ({percentage:.1f}%)"

            # White text in dark-blue cells.
            text_colour = (
                "white"
                if percentage >= 50
                else "#222222"
            )

            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color=text_colour,
            )

    # Keep complete matrix visible.
    ax.set_ylim(counts.shape[0] - 0.5, -0.5)

    # --------------------------------------------------------
    # Colour bar
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    for extension in ("png", "pdf", "svg"):
        output_path = OUTPUT_DIR / f"{output_name}.{extension}"

        save_parameters = {
            "bbox_inches": "tight",
            "facecolor": fig.get_facecolor(),
        }

        if extension == "png":
            save_parameters["dpi"] = 600

        fig.savefig(
            output_path,
            **save_parameters,
        )

    plt.close(fig)

    print(f"Saved: {OUTPUT_DIR / output_name}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    family_matrix = load_confusion_matrix(FAMILY_CSV)
    genus_matrix = load_confusion_matrix(GENUS_CSV)

    print(f"Family matrix shape: {family_matrix.shape}")
    print(f"Genus matrix shape: {genus_matrix.shape}")

    # Family matrix: show all cells.
    plot_confusion_matrix(
        matrix=family_matrix,
        title="Matriz de Confusão — Classificação por Família",
        output_name="family_confusion_matrix_blue",
        figsize=(13, 11),
        tick_fontsize=11,
        annotation_fontsize=9,
        annotate_zeros=False,
    )

    # Genus matrix: hide zero annotations to reduce visual clutter.
    plot_confusion_matrix(
        matrix=genus_matrix,
        title="Matriz de Confusão — Classificação por Género",
        output_name="genus_confusion_matrix_blue",
        figsize=(30,28 ),
        tick_fontsize=8,
        annotation_fontsize=6,
        annotate_zeros=False,
    )

    print("Confusion matrices generated successfully.")


if __name__ == "__main__":
    main()
