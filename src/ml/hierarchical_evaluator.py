"""Evaluate hierarchical Brachycera family-genus predictions.


"""
# import modules 
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for hierarchical evaluation. "
        "Install it with: pip install scikit-learn"
    ) from error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IDENTIFICATION_CSV = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
    / "identification"
    / "specimen_identifications.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
    / "evaluation"
)

UNKNOWN_VALUES = {
    "",
    "unknown",
    "unlabelled",
    "unlabeled",
    "none",
    "nan",
    "na",
    "n/a",
    "null",
    "not identified",
    "not_identified",
    "não identificado",
    "nao identificado",
    "-",
}


def normalize_label(value: Any) -> str | None:
    """Return a clean label or ``None`` for missing taxonomy."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    label = str(value).strip()

    if label.casefold() in UNKNOWN_VALUES:
        return None

    return label


def parse_boolean(value: Any) -> bool:
    """Interpret common boolean representations."""

    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
        "y",
        "sim",
    }


def safe_float(value: Any) -> float:
    """Convert a value to float and return NaN when unavailable."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def percentage(value: float | None) -> str:
    """Format a proportion as a percentage."""

    if value is None or np.isnan(value):
        return "Not available"

    return f"{value * 100:.2f}%"


class HierarchicalEvaluator:
    """Evaluate family and genus predictions against collection metadata."""

    def __init__(
        self,
        identification_csv: str | Path = DEFAULT_IDENTIFICATION_CSV,
        output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
        minimum_support: int = 1,
    ) -> None:
        self.identification_csv = Path(identification_csv)
        self.output_directory = Path(output_directory)
        self.minimum_support = int(minimum_support)

        if self.minimum_support < 1:
            raise ValueError("minimum_support must be at least 1.")

        self.dataframe = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and normalize the identification output."""

        if not self.identification_csv.exists():
            raise FileNotFoundError(
                f"Identification CSV not found: {self.identification_csv}"
            )

        dataframe = pd.read_csv(self.identification_csv)

        required_columns = {
            "specimen_id",
            "metadata_family",
            "metadata_genus",
            "predicted_family",
            "predicted_genus",
            "family_confidence",
            "genus_confidence",
        }

        missing_columns = required_columns.difference(dataframe.columns)

        if missing_columns:
            raise KeyError(
                "The identification CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for column in (
            "metadata_family",
            "metadata_genus",
            "predicted_family",
            "predicted_genus",
            "raw_genus",
            "raw_genus_family",
        ):
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].apply(normalize_label)

        dataframe["family_confidence"] = dataframe[
            "family_confidence"
        ].apply(safe_float)

        dataframe["genus_confidence"] = dataframe[
            "genus_confidence"
        ].apply(safe_float)

        if "raw_genus_confidence" in dataframe.columns:
            dataframe["raw_genus_confidence"] = dataframe[
                "raw_genus_confidence"
            ].apply(safe_float)

        if "requires_expert_verification" in dataframe.columns:
            dataframe["requires_expert_verification"] = dataframe[
                "requires_expert_verification"
            ].apply(parse_boolean)

        dataframe["family_correct"] = (
            dataframe["metadata_family"].notna()
            & dataframe["predicted_family"].notna()
            & (
                dataframe["metadata_family"].str.casefold()
                == dataframe["predicted_family"].str.casefold()
            )
        )

        dataframe["genus_correct"] = (
            dataframe["metadata_genus"].notna()
            & dataframe["predicted_genus"].notna()
            & (
                dataframe["metadata_genus"].str.casefold()
                == dataframe["predicted_genus"].str.casefold()
            )
        )

        dataframe["hierarchical_correct"] = (
            dataframe["metadata_family"].notna()
            & dataframe["metadata_genus"].notna()
            & dataframe["family_correct"]
            & dataframe["genus_correct"]
        )

        return dataframe

    @staticmethod
    def _classification_metrics(
        true_labels: pd.Series,
        predicted_labels: pd.Series,
    ) -> dict[str, Any]:
        """Calculate overall multiclass metrics."""

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        )

        return {
            "sample_count": int(len(true_labels)),
            "accuracy": float(
                accuracy_score(true_labels, predicted_labels)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    true_labels,
                    predicted_labels,
                )
            ),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1": float(f1_weighted),
        }

    def _evaluate_level(
        self,
        level: str,
    ) -> tuple[
        dict[str, Any],
        pd.DataFrame,
        pd.DataFrame,
        list[str],
    ]:
        """Evaluate one taxonomic level."""

        metadata_column = f"metadata_{level}"
        predicted_column = f"predicted_{level}"

        subset = self.dataframe[
            self.dataframe[metadata_column].notna()
            & self.dataframe[predicted_column].notna()
        ].copy()

        if subset.empty:
            return {}, pd.DataFrame(), pd.DataFrame(), []

        true_labels = subset[metadata_column].astype(str)
        predicted_labels = subset[predicted_column].astype(str)

        labels = sorted(
            set(true_labels).union(set(predicted_labels)),
            key=str.casefold,
        )

        overall_metrics = self._classification_metrics(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
        )

        report = classification_report(
            true_labels,
            predicted_labels,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        per_class_rows = []

        for label in labels:
            values = report.get(label, {})
            support = int(values.get("support", 0))

            if support < self.minimum_support:
                continue

            per_class_rows.append(
                {
                    level: label,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1_score": float(values.get("f1-score", 0.0)),
                    "support": support,
                }
            )

        per_class = pd.DataFrame(per_class_rows)

        matrix = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=labels,
        )

        confusion = pd.DataFrame(
            matrix,
            index=[f"true:{label}" for label in labels],
            columns=[f"pred:{label}" for label in labels],
        )

        return overall_metrics, per_class, confusion, labels

    def _hierarchical_metrics(self) -> dict[str, Any]:
        """Calculate joint family-genus agreement statistics."""

        subset = self.dataframe[
            self.dataframe["metadata_family"].notna()
            & self.dataframe["metadata_genus"].notna()
        ].copy()

        if subset.empty:
            return {
                "sample_count": 0,
                "exact_match_accuracy": None,
                "family_accuracy": None,
                "genus_accuracy": None,
            }

        return {
            "sample_count": int(len(subset)),
            "exact_match_accuracy": float(
                subset["hierarchical_correct"].mean()
            ),
            "family_accuracy": float(
                subset["family_correct"].mean()
            ),
            "genus_accuracy": float(
                subset["genus_correct"].mean()
            ),
        }

    def _confidence_metrics(self) -> dict[str, Any]:
        """Summarize confidence and confidence-conditioned agreement."""

        results: dict[str, Any] = {}

        for level in ("family", "genus"):
            confidence_column = f"{level}_confidence"
            correct_column = f"{level}_correct"
            metadata_column = f"metadata_{level}"

            subset = self.dataframe[
                self.dataframe[metadata_column].notna()
                & self.dataframe[confidence_column].notna()
            ].copy()

            if subset.empty:
                results[level] = {}
                continue

            confidence = subset[confidence_column]

            threshold_metrics = {}

            for threshold in (0.50, 0.60, 0.75, 0.85, 0.95):
                threshold_subset = subset[
                    subset[confidence_column] >= threshold
                ]

                threshold_metrics[str(threshold)] = {
                    "count": int(len(threshold_subset)),
                    "coverage": float(
                        len(threshold_subset) / len(subset)
                    ),
                    "agreement": (
                        float(
                            threshold_subset[correct_column].mean()
                        )
                        if not threshold_subset.empty
                        else None
                    ),
                }

            correct_confidence = subset.loc[
                subset[correct_column],
                confidence_column,
            ]

            incorrect_confidence = subset.loc[
                ~subset[correct_column],
                confidence_column,
            ]

            results[level] = {
                "count": int(len(subset)),
                "mean": float(confidence.mean()),
                "median": float(confidence.median()),
                "minimum": float(confidence.min()),
                "maximum": float(confidence.max()),
                "mean_when_correct": (
                    float(correct_confidence.mean())
                    if not correct_confidence.empty
                    else None
                ),
                "mean_when_incorrect": (
                    float(incorrect_confidence.mean())
                    if not incorrect_confidence.empty
                    else None
                ),
                "threshold_analysis": threshold_metrics,
            }

        return results

    def _review_priority(self) -> pd.DataFrame:
        """Create a curator review queue sorted by urgency."""

        dataframe = self.dataframe.copy()

        dataframe["review_priority_score"] = 0

        dataframe.loc[
            dataframe["metadata_family"].notna()
            & ~dataframe["family_correct"],
            "review_priority_score",
        ] += 5

        dataframe.loc[
            dataframe["metadata_genus"].notna()
            & ~dataframe["genus_correct"],
            "review_priority_score",
        ] += 3

        dataframe.loc[
            dataframe["family_confidence"] < 0.60,
            "review_priority_score",
        ] += 2

        dataframe.loc[
            dataframe["genus_confidence"] < 0.60,
            "review_priority_score",
        ] += 2

        if "raw_prediction_is_consistent" in dataframe.columns:
            inconsistent = ~dataframe[
                "raw_prediction_is_consistent"
            ].apply(parse_boolean)

            dataframe.loc[
                inconsistent,
                "review_priority_score",
            ] += 2

        dataframe["priority"] = pd.cut(
            dataframe["review_priority_score"],
            bins=[-1, 1, 4, 7, np.inf],
            labels=["Low", "Moderate", "High", "Critical"],
        ).astype(str)

        selected_columns = [
            "specimen_id",
            "metadata_family",
            "metadata_genus",
            "predicted_family",
            "family_confidence",
            "predicted_genus",
            "genus_confidence",
            "family_correct",
            "genus_correct",
            "review_priority_score",
            "priority",
        ]

        for optional_column in (
            "raw_genus",
            "raw_genus_confidence",
            "raw_genus_family",
            "raw_prediction_is_consistent",
            "top_genus_candidates",
            "image_path",
        ):
            if optional_column in dataframe.columns:
                selected_columns.append(optional_column)

        return dataframe[selected_columns].sort_values(
            by=[
                "review_priority_score",
                "family_confidence",
                "genus_confidence",
            ],
            ascending=[False, True, True],
        )

    @staticmethod
    def _plot_confusion_matrix(
        confusion: pd.DataFrame,
        title: str,
        output_path: Path,
    ) -> None:
        """Save one confusion matrix figure."""

        if confusion.empty:
            return

        matrix = confusion.to_numpy()

        figure_size = max(8, min(20, len(confusion) * 0.55))
        plt.figure(figsize=(figure_size, figure_size))
        plt.imshow(matrix, interpolation="nearest", aspect="auto")
        plt.title(title)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks(
            np.arange(len(confusion.columns)),
            [column.replace("pred:", "") for column in confusion.columns],
            rotation=90,
        )
        plt.yticks(
            np.arange(len(confusion.index)),
            [index.replace("true:", "") for index in confusion.index],
        )
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_confidence_histogram(
        self,
        level: str,
        output_path: Path,
    ) -> None:
        """Plot confidence distributions for correct and incorrect rows."""

        confidence_column = f"{level}_confidence"
        correct_column = f"{level}_correct"
        metadata_column = f"metadata_{level}"

        subset = self.dataframe[
            self.dataframe[metadata_column].notna()
            & self.dataframe[confidence_column].notna()
        ].copy()

        if subset.empty:
            return

        correct_values = subset.loc[
            subset[correct_column],
            confidence_column,
        ]

        incorrect_values = subset.loc[
            ~subset[correct_column],
            confidence_column,
        ]

        plt.figure(figsize=(9, 6))

        bins = np.linspace(0, 1, 21)

        if not correct_values.empty:
            plt.hist(
                correct_values,
                bins=bins,
                alpha=0.6,
                label="Agreement",
            )

        if not incorrect_values.empty:
            plt.hist(
                incorrect_values,
                bins=bins,
                alpha=0.6,
                label="Disagreement",
            )

        plt.title(f"{level.title()} confidence distribution")
        plt.xlabel("Model confidence")
        plt.ylabel("Number of specimens")
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_class_support(
        self,
        level: str,
        per_class: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Plot per-class F1 score and sample support."""

        if per_class.empty:
            return

        label_column = level
        ordered = per_class.sort_values(
            by="f1_score",
            ascending=True,
        )

        plt.figure(
            figsize=(10, max(6, len(ordered) * 0.35))
        )
        plt.barh(
            ordered[label_column],
            ordered["f1_score"],
        )
        plt.xlabel("F1 score")
        plt.ylabel(level.title())
        plt.xlim(0, 1)
        plt.title(f"Per-{level} F1 score")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def evaluate(self) -> dict[str, Any]:
        """Run the complete evaluation workflow."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        family_metrics, family_per_class, family_confusion, _ = (
            self._evaluate_level("family")
        )

        genus_metrics, genus_per_class, genus_confusion, _ = (
            self._evaluate_level("genus")
        )

        hierarchical_metrics = self._hierarchical_metrics()
        confidence_metrics = self._confidence_metrics()
        review_priority = self._review_priority()

        summary = {
            "generated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "source_file": str(self.identification_csv),
            "total_identification_records": int(
                len(self.dataframe)
            ),
            "family": family_metrics,
            "genus": genus_metrics,
            "hierarchical": hierarchical_metrics,
            "confidence": confidence_metrics,
            "scientific_interpretation": (
                "These values measure agreement with available collection "
                "metadata. They should be described as independent test "
                "performance only when the evaluated specimens were not "
                "used during model training, model selection or threshold "
                "tuning."
            ),
        }

        family_per_class.to_csv(
            self.output_directory / "family_per_class_metrics.csv",
            index=False,
        )

        genus_per_class.to_csv(
            self.output_directory / "genus_per_class_metrics.csv",
            index=False,
        )

        family_confusion.to_csv(
            self.output_directory / "family_confusion_matrix.csv"
        )

        genus_confusion.to_csv(
            self.output_directory / "genus_confusion_matrix.csv"
        )

        review_priority.to_csv(
            self.output_directory / "review_priority.csv",
            index=False,
        )

        family_disagreements = self.dataframe[
            self.dataframe["metadata_family"].notna()
            & ~self.dataframe["family_correct"]
        ].copy()

        genus_disagreements = self.dataframe[
            self.dataframe["metadata_genus"].notna()
            & ~self.dataframe["genus_correct"]
        ].copy()

        family_disagreements.to_csv(
            self.output_directory / "family_disagreements.csv",
            index=False,
        )

        genus_disagreements.to_csv(
            self.output_directory / "genus_disagreements.csv",
            index=False,
        )

        with (
            self.output_directory / "evaluation_summary.json"
        ).open("w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

        self._write_text_summary(summary)

        self._plot_confusion_matrix(
            family_confusion,
            "Family confusion matrix",
            self.output_directory
            / "family_confusion_matrix.png",
        )

        self._plot_confusion_matrix(
            genus_confusion,
            "Genus confusion matrix",
            self.output_directory
            / "genus_confusion_matrix.png",
        )

        self._plot_confidence_histogram(
            "family",
            self.output_directory
            / "family_confidence_histogram.png",
        )

        self._plot_confidence_histogram(
            "genus",
            self.output_directory
            / "genus_confidence_histogram.png",
        )

        self._plot_class_support(
            "family",
            family_per_class,
            self.output_directory
            / "family_f1_scores.png",
        )

        self._plot_class_support(
            "genus",
            genus_per_class,
            self.output_directory
            / "genus_f1_scores.png",
        )

        self._print_summary(summary)

        return summary

    def _write_text_summary(
        self,
        summary: dict[str, Any],
    ) -> None:
        """Write a readable evaluation report."""

        family = summary.get("family", {})
        genus = summary.get("genus", {})
        hierarchy = summary.get("hierarchical", {})

        lines = [
            "=" * 78,
            "HIERARCHICAL BRACHYCERA EVALUATION REPORT",
            "=" * 78,
            "",
            f"Generated at: {summary['generated_at']}",
            f"Source: {summary['source_file']}",
            (
                "Total prediction records: "
                f"{summary['total_identification_records']}"
            ),
            "",
            "Family-level evaluation",
            "-" * 78,
            f"Evaluated specimens: {family.get('sample_count', 0)}",
            f"Accuracy: {percentage(family.get('accuracy'))}",
            (
                "Balanced accuracy: "
                f"{percentage(family.get('balanced_accuracy'))}"
            ),
            f"Macro F1: {percentage(family.get('macro_f1'))}",
            (
                "Weighted F1: "
                f"{percentage(family.get('weighted_f1'))}"
            ),
            "",
            "Genus-level evaluation",
            "-" * 78,
            f"Evaluated specimens: {genus.get('sample_count', 0)}",
            f"Accuracy: {percentage(genus.get('accuracy'))}",
            (
                "Balanced accuracy: "
                f"{percentage(genus.get('balanced_accuracy'))}"
            ),
            f"Macro F1: {percentage(genus.get('macro_f1'))}",
            (
                "Weighted F1: "
                f"{percentage(genus.get('weighted_f1'))}"
            ),
            "",
            "Joint hierarchical evaluation",
            "-" * 78,
            (
                "Specimens with both metadata labels: "
                f"{hierarchy.get('sample_count', 0)}"
            ),
            (
                "Exact family-genus agreement: "
                f"{percentage(hierarchy.get('exact_match_accuracy'))}"
            ),
            (
                "Family agreement in joint subset: "
                f"{percentage(hierarchy.get('family_accuracy'))}"
            ),
            (
                "Genus agreement in joint subset: "
                f"{percentage(hierarchy.get('genus_accuracy'))}"
            ),
            "",
            "Interpretation",
            "-" * 78,
            summary["scientific_interpretation"],
        ]

        (
            self.output_directory / "evaluation_summary.txt"
        ).write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    @staticmethod
    def _print_summary(
        summary: dict[str, Any],
    ) -> None:
        """Print concise evaluation results."""

        family = summary.get("family", {})
        genus = summary.get("genus", {})
        hierarchy = summary.get("hierarchical", {})

        print("\nHierarchical evaluation complete")
        print("=" * 72)
        print(
            "Family accuracy: "
            f"{percentage(family.get('accuracy'))}"
        )
        print(
            "Family macro F1: "
            f"{percentage(family.get('macro_f1'))}"
        )
        print(
            "Genus accuracy: "
            f"{percentage(genus.get('accuracy'))}"
        )
        print(
            "Genus macro F1: "
            f"{percentage(genus.get('macro_f1'))}"
        )
        print(
            "Exact family-genus agreement: "
            f"{percentage(hierarchy.get('exact_match_accuracy'))}"
        )
        print(
            f"Reports saved to: {DEFAULT_OUTPUT_DIRECTORY}"
        )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hierarchical Brachycera predictions against "
            "available collection metadata."
        )
    )

    parser.add_argument(
        "--identifications",
        type=Path,
        default=DEFAULT_IDENTIFICATION_CSV,
        help="Path to specimen_identifications.csv.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--minimum-support",
        type=int,
        default=1,
        help=(
            "Minimum metadata support required to include a class "
            "in per-class metric tables."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run hierarchical evaluation."""

    arguments = parse_arguments()

    evaluator = HierarchicalEvaluator(
        identification_csv=arguments.identifications,
        output_directory=arguments.output_directory,
        minimum_support=arguments.minimum_support,
    )

    evaluator.evaluate()


if __name__ == "__main__":
    main()

