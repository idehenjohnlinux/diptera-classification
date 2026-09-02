"""Aggregate Brachycera CNN evaluation results across all folds.


"""
# import module
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_ROOT: Final[Path] = (
    PROJECT_ROOT / "results" / "training"
)

DEFAULT_OUTPUT_DIRECTORY: Final[Path] = (
    PROJECT_ROOT / "results" / "evaluation"
)

EXPECTED_FOLDS: Final[int] = 5

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
)

EXPERIMENT_COLUMNS: Final[tuple[str, ...]] = (
    "architecture",
    "taxonomic_level",
    "view_code",
)

IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "architecture",
    "taxonomic_level",
    "view_code",
    "validation_fold",
    "evaluation_level",
)


# ============================================================
# FILE UTILITIES
# ============================================================

# Load and validate an evaluation summary stored in JSON format.
def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"but received {type(data).__name__}."
        )

    return data

# Save aggregated evaluation information in a structured JSON file.
def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a dictionary as formatted JSON."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

# Locate all evaluation-summary files produced by the CNN experiments.
def find_evaluation_summaries(
    training_root: Path,
) -> list[Path]:
    """Find all evaluation-summary files."""

    if not training_root.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: "
            f"{training_root}"
        )

    return sorted(
        training_root.rglob(
            "evaluation_summary.json"
        )
    )


# ============================================================
# SUMMARY PARSING
# ============================================================
# Extract the classification metrics required for model comparison.
def extract_metrics(
    metric_data: dict[str, Any],
) -> dict[str, float]:
    """Extract supported metrics from one evaluation level."""

    extracted: dict[str, float] = {}

    for metric in METRIC_COLUMNS:
        value = metric_data.get(metric)

        if value is None:
            extracted[metric] = np.nan
        else:
            extracted[metric] = float(value)

    return extracted

# Convert the evaluation results of one experiment into a structured table row.
def create_result_row(
    summary: dict[str, Any],
    summary_path: Path,
    evaluation_level: str,
) -> dict[str, Any]:
    """Convert one evaluation level into one table row."""

    level_data = summary.get(
        f"{evaluation_level}_level"
    )

    if not isinstance(level_data, dict):
        raise KeyError(
            f"Missing '{evaluation_level}_level' "
            f"in {summary_path}"
        )

    row: dict[str, Any] = {
        "architecture": str(
            summary["architecture"]
        ),
        "taxonomic_level": str(
            summary["taxonomic_level"]
        ),
        "view_code": str(
            summary["view_code"]
        ),
        "validation_fold": int(
            summary["validation_fold"]
        ),
        "evaluation_level": evaluation_level,
        "number_of_classes": int(
            summary.get("number_of_classes", 0)
        ),
        "checkpoint_epoch": int(
            summary.get("checkpoint_epoch", -1)
        ),
        "checkpoint_path": str(
            summary.get("checkpoint_path", "")
        ),
        "evaluation_summary_path": str(
            summary_path.resolve()
        ),
    }

    if evaluation_level == "image":
        row["number_of_samples"] = int(
            level_data.get("number_of_images", 0)
        )
    else:
        row["number_of_samples"] = int(
            level_data.get(
                "number_of_specimens",
                0,
            )
        )

    row.update(
        extract_metrics(level_data)
    )

    return row

# Extract both image-level and specimen-level results from one evaluation summary.
def parse_evaluation_summary(
    summary_path: Path,
) -> list[dict[str, Any]]:
    """Parse image-level and specimen-level result rows."""

    summary = load_json(summary_path)

    required_keys = {
        "architecture",
        "taxonomic_level",
        "view_code",
        "validation_fold",
    }

    missing_keys = required_keys - set(summary)

    if missing_keys:
        raise KeyError(
            f"{summary_path} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    rows: list[dict[str, Any]] = []

    for evaluation_level in (
        "image",
        "specimen",
    ):
        level_key = f"{evaluation_level}_level"

        if level_key in summary:
            rows.append(
                create_result_row(
                    summary=summary,
                    summary_path=summary_path,
                    evaluation_level=(
                        evaluation_level
                    ),
                )
            )

    if not rows:
        raise KeyError(
            f"No image-level or specimen-level "
            f"metrics were found in {summary_path}."
        )

    return rows


# ============================================================
# DATA VALIDATION
# ============================================================


# Check that each architecture, taxonomic level, view, fold, and
# evaluation level is represented only once in the results.
def validate_duplicate_results(
    results: pd.DataFrame,
) -> None:
    """Ensure an experiment fold is not represented twice."""

    duplicated = results.duplicated(
        subset=list(IDENTITY_COLUMNS),
        keep=False,
    )

    if duplicated.any():
        duplicate_rows = results.loc[
            duplicated,
            list(IDENTITY_COLUMNS),
        ]

        raise RuntimeError(
            "Duplicate evaluation results were found:\n"
            f"{duplicate_rows.to_string(index=False)}"
        )

# Convert multi-level aggregation column names into simple column names.
def create_fold_completeness_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how many folds exist per configuration."""

    completeness = (
        results
        .groupby(
            [
                *EXPERIMENT_COLUMNS,
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(
            folds_present=(
                "validation_fold",
                "nunique",
            ),
            fold_numbers=(
                "validation_fold",
                lambda values: ",".join(
                    str(value)
                    for value in sorted(
                        set(values)
                    )
                ),
            ),
        )
    )

    completeness["expected_folds"] = (
        EXPECTED_FOLDS
    )

    completeness["complete"] = (
        completeness["folds_present"]
        == EXPECTED_FOLDS
    )

    return completeness


# ============================================================
# AGGREGATION
# ============================================================

# Calculate the mean, standard deviation, minimum, and maximum
# performance across the five cross-validation folds.
def flatten_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten columns created by multiple aggregations."""

    flattened = dataframe.copy()

    flattened.columns = [
        (
            "_".join(
                str(part)
                for part in column
                if str(part)
            )
            if isinstance(column, tuple)
            else str(column)
        )
        for column in flattened.columns
    ]

    return flattened

# Compare the overall classification performance of the three CNN architectures.
def aggregate_across_folds(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean and standard deviation across folds."""

    aggregation_rules: dict[str, list[str]] = {
        metric: [
            "mean",
            "std",
            "min",
            "max",
        ]
        for metric in METRIC_COLUMNS
    }

    aggregation_rules.update(
        {
            "validation_fold": [
                "nunique",
            ],
            "number_of_samples": [
                "sum",
                "mean",
            ],
        }
    )

    summary = (
        results
        .groupby(
            [
                *EXPERIMENT_COLUMNS,
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(aggregation_rules)
    )

    summary = flatten_columns(summary)

    summary = summary.rename(
        columns={
            "validation_fold_nunique": (
                "folds_evaluated"
            ),
            "number_of_samples_sum": (
                "samples_across_folds"
            ),
            "number_of_samples_mean": (
                "mean_samples_per_fold"
            ),
        }
    )

    for metric in METRIC_COLUMNS:
        standard_deviation_column = (
            f"{metric}_std"
        )

        if standard_deviation_column in summary:
            summary[
                standard_deviation_column
            ] = summary[
                standard_deviation_column
            ].fillna(0.0)

    return summary

# Compare the overall classification performance of the three CNN architectures.
def create_architecture_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare architectures across views and levels."""

    comparison = (
        results
        .groupby(
            [
                "architecture",
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(
            experiments=(
                "accuracy",
                "size",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            std_accuracy=(
                "accuracy",
                "std",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_f1_macro=(
                "f1_macro",
                "mean",
            ),
            std_f1_macro=(
                "f1_macro",
                "std",
            ),
            mean_precision_macro=(
                "precision_macro",
                "mean",
            ),
            mean_recall_macro=(
                "recall_macro",
                "mean",
            ),
        )
    )

    return comparison.fillna(0.0)

# Compare classification performance across the four anatomical views
def create_view_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare anatomical views across architectures and levels."""

    comparison = (
        results
        .groupby(
            [
                "view_code",
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(
            experiments=(
                "accuracy",
                "size",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            std_accuracy=(
                "accuracy",
                "std",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_f1_macro=(
                "f1_macro",
                "mean",
            ),
            std_f1_macro=(
                "f1_macro",
                "std",
            ),
            mean_precision_macro=(
                "precision_macro",
                "mean",
            ),
            mean_recall_macro=(
                "recall_macro",
                "mean",
            ),
        )
    )

    return comparison.fillna(0.0)

# Compare model performance between family- and genus-level classification.
def create_taxonomic_level_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare family and genus classification."""

    comparison = (
        results
        .groupby(
            [
                "taxonomic_level",
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(
            experiments=(
                "accuracy",
                "size",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            std_accuracy=(
                "accuracy",
                "std",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_f1_macro=(
                "f1_macro",
                "mean",
            ),
            std_f1_macro=(
                "f1_macro",
                "std",
            ),
            mean_precision_macro=(
                "precision_macro",
                "mean",
            ),
            mean_recall_macro=(
                "recall_macro",
                "mean",
            ),
        )
    )

    return comparison.fillna(0.0)


def create_detailed_comparison(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare architecture, level and view combinations."""

    comparison = (
        results
        .groupby(
            [
                "architecture",
                "taxonomic_level",
                "view_code",
                "evaluation_level",
            ],
            as_index=False,
        )
        .agg(
            folds_evaluated=(
                "validation_fold",
                "nunique",
            ),
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            std_accuracy=(
                "accuracy",
                "std",
            ),
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean",
            ),
            std_balanced_accuracy=(
                "balanced_accuracy",
                "std",
            ),
            mean_precision_macro=(
                "precision_macro",
                "mean",
            ),
            std_precision_macro=(
                "precision_macro",
                "std",
            ),
            mean_recall_macro=(
                "recall_macro",
                "mean",
            ),
            std_recall_macro=(
                "recall_macro",
                "std",
            ),
            mean_f1_macro=(
                "f1_macro",
                "mean",
            ),
            std_f1_macro=(
                "f1_macro",
                "std",
            ),
            mean_f1_weighted=(
                "f1_weighted",
                "mean",
            ),
            std_f1_weighted=(
                "f1_weighted",
                "std",
            ),
        )
    )

    return comparison.fillna(0.0)


# ============================================================
# RANKING AND BEST MODEL SELECTION
# ============================================================

# Rank model configurations using accuracy, balanced accuracy,
# macro F1-score, and a combined performance score.
def create_rankings(
    detailed_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Rank experiment configurations within each evaluation group."""

    rankings = detailed_comparison.copy()

    rankings["accuracy_rank"] = (
        rankings
        .groupby(
            [
                "taxonomic_level",
                "evaluation_level",
            ]
        )["mean_accuracy"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    rankings["balanced_accuracy_rank"] = (
        rankings
        .groupby(
            [
                "taxonomic_level",
                "evaluation_level",
            ]
        )["mean_balanced_accuracy"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    rankings["f1_macro_rank"] = (
        rankings
        .groupby(
            [
                "taxonomic_level",
                "evaluation_level",
            ]
        )["mean_f1_macro"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    rankings["combined_score"] = (
        rankings["mean_balanced_accuracy"]
        + rankings["mean_f1_macro"]
    ) / 2.0

    rankings["combined_rank"] = (
        rankings
        .groupby(
            [
                "taxonomic_level",
                "evaluation_level",
            ]
        )["combined_score"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    rankings = rankings.sort_values(
        by=[
            "taxonomic_level",
            "evaluation_level",
            "combined_rank",
            "mean_f1_macro",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
    )

    return rankings.reset_index(drop=True)

# Select the highest-ranked model configuration for each
# taxonomic and evaluation level.
def create_best_models_table(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Select the highest-ranked configuration per level."""

    best_models = rankings.loc[
        rankings["combined_rank"] == 1
    ].copy()

    best_models = best_models.sort_values(
        by=[
            "taxonomic_level",
            "evaluation_level",
        ]
    )

    return best_models.reset_index(drop=True)


# ============================================================
# MAIN AGGREGATION
# ============================================================

# Aggregate all CNN evaluation results, generate comparison tables,
# rank the models, and save the final evaluation outputs.
def aggregate_results(
    training_root: Path = DEFAULT_TRAINING_ROOT,
    output_directory: Path = (
        DEFAULT_OUTPUT_DIRECTORY
    ),
    strict: bool = False,
) -> dict[str, Any]:
    """Aggregate every available evaluation result."""

    training_root = training_root.resolve()
    output_directory = output_directory.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_paths = find_evaluation_summaries(
        training_root=training_root
    )

    if not summary_paths:
        raise FileNotFoundError(
            "No evaluation_summary.json files were found "
            f"under {training_root}."
        )

    print("=" * 72)
    print("BRACHYCERA CNN RESULT AGGREGATION")
    print("=" * 72)
    print(f"Training root: {training_root}")
    print(
        f"Evaluation files found: "
        f"{len(summary_paths)}"
    )

    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for position, summary_path in enumerate(
        summary_paths,
        start=1,
    ):
        print(
            f"[{position}/{len(summary_paths)}] "
            f"{summary_path}"
        )

        try:
            result_rows.extend(
                parse_evaluation_summary(
                    summary_path
                )
            )
        except Exception as error:
            failure = {
                "path": str(summary_path),
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            failures.append(failure)

            if strict:
                raise

            print(
                "  Failed: "
                f"{failure['error']}"
            )

    if not result_rows:
        raise RuntimeError(
            "No valid evaluation results were parsed."
        )

    all_results = pd.DataFrame(result_rows)

    all_results = all_results.sort_values(
        by=[
            "evaluation_level",
            "taxonomic_level",
            "architecture",
            "view_code",
            "validation_fold",
        ]
    ).reset_index(drop=True)

    validate_duplicate_results(all_results)

    completeness = (
        create_fold_completeness_table(
            results=all_results
        )
    )

    fold_summary = aggregate_across_folds(
        results=all_results
    )

    architecture_comparison = (
        create_architecture_comparison(
            results=all_results
        )
    )

    view_comparison = (
        create_view_comparison(
            results=all_results
        )
    )

    taxonomic_comparison = (
        create_taxonomic_level_comparison(
            results=all_results
        )
    )

    detailed_comparison = (
        create_detailed_comparison(
            results=all_results
        )
    )

    rankings = create_rankings(
        detailed_comparison=detailed_comparison
    )

    best_models = create_best_models_table(
        rankings=rankings
    )

    # --------------------------------------------------------
    # Save CSV outputs
    # --------------------------------------------------------

    all_results.to_csv(
        output_directory / "all_results.csv",
        index=False,
    )

    completeness.to_csv(
        output_directory
        / "fold_completeness.csv",
        index=False,
    )

    fold_summary.to_csv(
        output_directory
        / "cross_validation_summary.csv",
        index=False,
    )

    architecture_comparison.to_csv(
        output_directory
        / "architecture_comparison.csv",
        index=False,
    )

    view_comparison.to_csv(
        output_directory
        / "view_comparison.csv",
        index=False,
    )

    taxonomic_comparison.to_csv(
        output_directory
        / "taxonomic_level_comparison.csv",
        index=False,
    )

    detailed_comparison.to_csv(
        output_directory
        / "detailed_model_comparison.csv",
        index=False,
    )

    rankings.to_csv(
        output_directory
        / "model_rankings.csv",
        index=False,
    )

    best_models.to_csv(
        output_directory
        / "best_models.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            output_directory
            / "aggregation_failures.csv",
            index=False,
        )

    incomplete_experiments = completeness.loc[
        ~completeness["complete"]
    ]

    aggregation_summary: dict[str, Any] = {
        "training_root": str(training_root),
        "output_directory": str(
            output_directory
        ),
        "evaluation_summary_files_found": int(
            len(summary_paths)
        ),
        "evaluation_rows_created": int(
            len(all_results)
        ),
        "image_level_rows": int(
            (
                all_results["evaluation_level"]
                == "image"
            ).sum()
        ),
        "specimen_level_rows": int(
            (
                all_results["evaluation_level"]
                == "specimen"
            ).sum()
        ),
        "unique_configurations": int(
            all_results[
                [
                    *EXPERIMENT_COLUMNS,
                    "evaluation_level",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),
        "complete_configurations": int(
            completeness["complete"].sum()
        ),
        "incomplete_configurations": int(
            (~completeness["complete"]).sum()
        ),
        "failed_files": int(len(failures)),
        "expected_folds_per_configuration": (
            EXPECTED_FOLDS
        ),
    }

    save_json(
        data=aggregation_summary,
        output_path=(
            output_directory
            / "aggregation_summary.json"
        ),
    )

    print("\n" + "=" * 72)
    print("AGGREGATION COMPLETE")
    print("=" * 72)
    print(
        f"Valid result rows: "
        f"{len(all_results)}"
    )
    print(
        f"Failed summaries:  "
        f"{len(failures)}"
    )
    print(
        f"Complete configurations: "
        f"{completeness['complete'].sum()}"
    )
    print(
        f"Incomplete configurations: "
        f"{len(incomplete_experiments)}"
    )
    print(f"Outputs: {output_directory}")

    if not incomplete_experiments.empty:
        print("\nIncomplete configurations:")
        print(
            incomplete_experiments[
                [
                    "architecture",
                    "taxonomic_level",
                    "view_code",
                    "evaluation_level",
                    "folds_present",
                    "fold_numbers",
                ]
            ].to_string(index=False)
        )

    print("\nBest configurations:")
    display_columns = [
        "architecture",
        "taxonomic_level",
        "view_code",
        "evaluation_level",
        "folds_evaluated",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "mean_f1_macro",
        "combined_score",
    ]

    print(
        best_models[
            display_columns
        ].to_string(index=False)
    )

    return aggregation_summary


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Brachycera CNN evaluation results "
            "across all cross-validation folds."
        )
    )

    parser.add_argument(
        "--training-root",
        type=Path,
        default=DEFAULT_TRAINING_ROOT,
        help=(
            "Directory containing trained experiment "
            "folders. Default: results/training."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory where aggregated tables will be "
            "saved. Default: results/evaluation."
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Stop immediately when an invalid evaluation "
            "summary is encountered."
        ),
    )

    return parser

# Execute the complete result-aggregation workflow from the command line.
def main() -> None:
    """Run result aggregation from the command line."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    aggregate_results(
        training_root=arguments.training_root,
        output_directory=(
            arguments.output_directory
        ),
        strict=arguments.strict,
    )


if __name__ == "__main__":
    main()
