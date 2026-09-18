#!/usr/bin/env python3
"""
Create one combined heatmap containing family- and genus-level
per-class evaluation metrics.

Inputs:
    results/production/hierarchical/evaluation/
        family_per_class_metrics.csv
        genus_per_class_metrics.csv

Output:
    results/figures/
        combined_family_genus_metrics_heatmap.png
        combined_family_genus_metrics_heatmap.pdf
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
    "family_per_class_metrics.csv"
)

GENUS_CSV = Path(
    "results/production/hierarchical/evaluation/"
    "genus_per_class_metrics.csv"
)

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load and standardise files
# ============================================================

def load_metrics(csv_path: Path, taxonomic_level: str) -> pd.DataFrame:
    """
    Load a per-class metrics CSV and standardise its column names.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardise column names.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    print(f"\nColumns detected in {csv_path.name}:")
    print(df.columns.tolist())

    # Possible names for the taxon/class column.
    class_candidates = [
        "class",
        "label",
        "taxon",
        "family",
        "genus",
        "class_name",
        "taxon_name",
        "true_label",
    ]

    class_column = next(
        (
            column
            for column in class_candidates
            if column in df.columns
        ),
        None,
    )

    # Fall back to the first column if no known class column exists.
    if class_column is None:
        class_column = df.columns[0]

    # Accept common alternatives for F1.
    if "f1_score" in df.columns:
        f1_column = "f1_score"
    elif "f1" in df.columns:
        f1_column = "f1"
    else:
        raise KeyError(
            f"No F1-score column found in {csv_path}. "
            f"Detected columns: {df.columns.tolist()}"
        )

    required_columns = ["precision", "recall", "support"]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns in {csv_path}: {missing}. "
            f"Detected columns: {df.columns.tolist()}"
        )

    result = pd.DataFrame(
        {
            "Taxon": df[class_column].astype(str),
            "Precision": pd.to_numeric(
                df["precision"], errors="coerce"
            ),
            "Recall": pd.to_numeric(
                df["recall"], errors="coerce"
            ),
            "F1-score": pd.to_numeric(
                df[f1_column], errors="coerce"
            ),
            "Support": pd.to_numeric(
                df["support"], errors="coerce"
            ).fillna(0).astype(int),
            "Level": taxonomic_level,
        }
    )

    result = result.dropna(
        subset=["Precision", "Recall", "F1-score"]
    )

    return result


# ============================================================
# Plot combined heatmap
# ============================================================

def plot_combined_heatmap(
    family_df: pd.DataFrame,
    genus_df: pd.DataFrame,
) -> None:
    """
    Produce one heatmap containing family and genus metrics.
    """

    # Sort taxa alphabetically inside each taxonomic level.
    family_df = family_df.sort_values("Taxon").reset_index(drop=True)
    genus_df = genus_df.sort_values("Taxon").reset_index(drop=True)

    combined = pd.concat(
        [family_df, genus_df],
        ignore_index=True,
    )

    metric_columns = [
        "Precision",
        "Recall",
        "F1-score",
    ]

    metric_values = combined[metric_columns].to_numpy(dtype=float)

    # Labels show taxonomic level and support.
    row_labels = [
        f"{row.Level}: {row.Taxon}  (n={row.Support})"
        for row in combined.itertuples()
    ]

    number_of_rows = len(combined)

    # Dynamic figure height.
    figure_height = max(10, number_of_rows * 0.42)

    fig, ax = plt.subplots(
        figsize=(10, figure_height)
    )

    image = ax.imshow(
        metric_values,
        cmap="Blues",
        aspect="auto",
        interpolation="none",
        vmin=0,
        vmax=1,
    )

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_xticklabels(
        metric_columns,
        fontsize=11,
        fontweight="bold",
    )

    ax.set_yticks(np.arange(number_of_rows))
    ax.set_yticklabels(
        row_labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "Métricas de avaliação",
        fontsize=12,
        labelpad=12,
    )

    ax.set_ylabel(
        "Taxonomic class",
        fontsize=12,
        labelpad=12,
    )

    ax.set_title(
        "Desempenho por classe nos níveis de família e gênero",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    # --------------------------------------------------------
    # Cell values
    # --------------------------------------------------------

    for row_index in range(metric_values.shape[0]):
        for column_index in range(metric_values.shape[1]):
            value = metric_values[row_index, column_index]

            text_colour = (
                "white"
                if value >= 0.65
                else "black"
            )

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_colour,
            )

    # --------------------------------------------------------
    # Cell borders
    # --------------------------------------------------------

    ax.set_xticks(
        np.arange(-0.5, len(metric_columns), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, number_of_rows, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=0.4,
        alpha=0.45,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Divide families and genera
    # --------------------------------------------------------

    family_count = len(family_df)

    ax.axhline(
        family_count - 0.5,
        linewidth=2.2,
        color="black",
    )

    # Add section labels on the right side.
    family_middle = (family_count - 1) / 2
    genus_middle = family_count + (len(genus_df) - 1) / 2

    ax.text(
        3.05,
        family_middle,
        "FAMÍLIA",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transData,
        clip_on=False,
    )

    ax.text(
        3.05,
        genus_middle,
        "GÊNERO",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transData,
        clip_on=False,
    )

    # Italicise genus labels.
    for index, label in enumerate(ax.get_yticklabels()):
        if index >= family_count:
            label.set_fontstyle("italic")

   

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png_path = (
        OUTPUT_DIR
        / "combined_family_genus_metrics_heatmap.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / "combined_family_genus_metrics_heatmap.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    family_metrics = load_metrics(
        FAMILY_CSV,
        taxonomic_level="Família",
    )

    genus_metrics = load_metrics(
        GENUS_CSV,
        taxonomic_level="Gênero",
    )

    print(f"\nFamily classes: {len(family_metrics)}")
    print(f"Genus classes: {len(genus_metrics)}")

    plot_combined_heatmap(
        family_metrics,
        genus_metrics,
    )

    print("\nCombined heatmap generated successfully.")


if __name__ == "__main__":
    main()
