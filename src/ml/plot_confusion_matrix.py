"""
Plot Confusion Matrices
=======================

Creates publication-quality confusion matrix figures.

For each confusion_matrix.csv:
- confusion_matrix_counts.png
- confusion_matrix_normalized.png

Also creates average confusion matrices per model/view.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """Plot a confusion matrix."""
    plt.figure(figsize=(9, 7))

    values = matrix.values.astype(float)

    plt.imshow(values, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    labels = matrix.index.tolist()

    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(labels)),
        labels=labels,
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = values.max() / 2 if values.max() > 0 else 0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if normalized:
                text = f"{values[i, j]:.2f}"
            else:
                text = str(int(values[i, j]))

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if values[i, j] > threshold else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Normalize confusion matrix by true label rows."""
    values = matrix.values.astype(float)

    row_sums = values.sum(axis=1, keepdims=True)

    normalized = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values),
        where=row_sums != 0,
    )

    return pd.DataFrame(
        normalized,
        index=matrix.index,
        columns=matrix.columns,
    )


def process_single_confusion_matrix(csv_file: Path) -> None:
    """Create count and normalized plots for one confusion matrix."""
    matrix = pd.read_csv(csv_file, index_col=0)

    output_dir = csv_file.parent

    plot_matrix(
        matrix=matrix,
        output_path=output_dir / "confusion_matrix_counts.png",
        title="Confusion matrix — counts",
        normalized=False,
    )

    normalized = normalize_matrix(matrix)

    normalized.to_csv(output_dir / "confusion_matrix_normalized.csv")

    plot_matrix(
        matrix=normalized,
        output_path=output_dir / "confusion_matrix_normalized.png",
        title="Confusion matrix — normalized by true label",
        normalized=True,
    )


def create_average_matrices() -> None:
    """
    Create average confusion matrices across folds.

    Expected structure:
    reports/evaluation/<mode>/<level>/<view>/<model>/fold_X/confusion_matrix.csv
    """
    base_dir = Path("reports/evaluation")

    model_dirs = [
        path
        for path in base_dir.rglob("*")
        if path.is_dir()
        and any(path.glob("fold_*/confusion_matrix.csv"))
    ]

    for model_dir in model_dirs:
        matrices = []

        for csv_file in sorted(model_dir.glob("fold_*/confusion_matrix.csv")):
            matrix = pd.read_csv(csv_file, index_col=0)
            matrices.append(matrix)

        if not matrices:
            continue

        reference_index = matrices[0].index
        reference_columns = matrices[0].columns

        aligned = [
            matrix.reindex(
                index=reference_index,
                columns=reference_columns,
                fill_value=0,
            )
            for matrix in matrices
        ]

        sum_matrix = sum(aligned)

        average_matrix = sum_matrix / len(aligned)

        average_matrix.to_csv(model_dir / "average_confusion_matrix_counts.csv")

        plot_matrix(
            matrix=average_matrix,
            output_path=model_dir / "average_confusion_matrix_counts.png",
            title=f"Average confusion matrix — {model_dir.name}",
            normalized=False,
        )

        normalized_average = normalize_matrix(sum_matrix)

        normalized_average.to_csv(
            model_dir / "average_confusion_matrix_normalized.csv"
        )

        plot_matrix(
            matrix=normalized_average,
            output_path=model_dir / "average_confusion_matrix_normalized.png",
            title=f"Average normalized confusion matrix — {model_dir.name}",
            normalized=True,
        )


def main() -> None:
    confusion_files = sorted(
        Path("reports/evaluation").rglob("confusion_matrix.csv")
    )

    if not confusion_files:
        raise FileNotFoundError(
            "No confusion_matrix.csv files found under reports/evaluation/"
        )

    for csv_file in confusion_files:
        process_single_confusion_matrix(csv_file)

    create_average_matrices()


if __name__ == "__main__":
    main()
