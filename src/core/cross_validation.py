"""Create specimen-level cross-validation folds for family and genus models.

The script reads the supervised dataset generated during preprocessing.
Fold assignments are created at specimen level, ensuring that images from
the same specimen never appear in both training and validation data.

Classes with fewer specimens than the number of folds are excluded from
cross-validation but retained in separate audit CSV files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = Path("metadata/supervised_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("metadata/cross_validation")

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42

VALID_LEVELS = ("family", "genus")
MISSING_LABELS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unidentified",
    "unlabelled",
}


def setup_logging() -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize known metadata column names."""

    dataframe = dataframe.copy()

    aliases = {
        "numcol": "numCol",
        "num_col": "numCol",
        "specimen_id": "numCol",
        "specimenid": "numCol",
        "family": "family",
        "genus": "genus",
        "view": "view_code",
        "viewcode": "view_code",
        "view_code": "view_code",
        "imagepath": "image_path",
        "image_path": "image_path",
        "processedimagepath": "processed_image_path",
        "processed_image_path": "processed_image_path",
    }

    renamed_columns: dict[str, str] = {}

    for column in dataframe.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        compact = normalized.replace("_", "")

        if normalized in aliases:
            renamed_columns[column] = aliases[normalized]
        elif compact in aliases:
            renamed_columns[column] = aliases[compact]

    return dataframe.rename(columns=renamed_columns)


def clean_text(value: Any) -> str:
    """Convert a metadata value into cleaned text."""

    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_taxonomy_label(value: Any) -> str:
    """Standardize family or genus labels."""

    text = clean_text(value)

    if text.lower() in MISSING_LABELS:
        return ""

    normalized = text.capitalize()

    corrections = {
        # Confirmed metadata correction.
        "Fannidae": "Fanniidae",
    }

    return corrections.get(normalized, normalized)


def validate_input_columns(dataframe: pd.DataFrame) -> None:
    """Ensure the supervised dataset contains required columns."""

    required_columns = {"numCol", "family", "genus"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The supervised dataset is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )


def load_supervised_dataset(input_path: Path) -> pd.DataFrame:
    """Load and clean the supervised image-level metadata."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Supervised dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)
    dataframe = normalize_column_names(dataframe)

    validate_input_columns(dataframe)

    dataframe["numCol"] = dataframe["numCol"].map(clean_text)

    for level in VALID_LEVELS:
        dataframe[level] = dataframe[level].map(clean_taxonomy_label)

    if "view_code" in dataframe.columns:
        dataframe["view_code"] = (
            dataframe["view_code"]
            .map(clean_text)
            .str.upper()
        )

    dataframe = dataframe[dataframe["numCol"] != ""].copy()

    LOGGER.info(
        "Loaded %s image records from %s",
        len(dataframe),
        input_path,
    )
    LOGGER.info(
        "Unique specimens in input: %s",
        dataframe["numCol"].nunique(),
    )

    return dataframe


def validate_specimen_labels(
    dataframe: pd.DataFrame,
    label_column: str,
) -> None:
    """Check that one specimen does not have multiple labels."""

    specimen_label_counts = (
        dataframe[dataframe[label_column] != ""]
        .groupby("numCol")[label_column]
        .nunique()
    )

    conflicting_specimens = specimen_label_counts[
        specimen_label_counts > 1
    ]

    if not conflicting_specimens.empty:
        examples = conflicting_specimens.index.tolist()[:10]

        raise ValueError(
            f"Some specimens have multiple {label_column} labels. "
            f"Examples: {examples}"
        )


def build_specimen_table(
    dataframe: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Create one row per labelled specimen."""

    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid level '{level}'. Choose from {VALID_LEVELS}."
        )

    validate_specimen_labels(dataframe, level)

    labelled = dataframe[dataframe[level] != ""].copy()

    specimen_table = (
        labelled[["numCol", level]]
        .drop_duplicates()
        .sort_values(["numCol", level])
        .reset_index(drop=True)
    )

    return specimen_table


def separate_eligible_and_rare_classes(
    specimen_table: pd.DataFrame,
    level: str,
    minimum_specimens: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate classes eligible for CV from underrepresented classes."""

    class_counts = (
        specimen_table.groupby(level)["numCol"]
        .nunique()
        .rename("specimen_count")
        .reset_index()
        .sort_values(
            ["specimen_count", level],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    class_counts["eligible_for_cross_validation"] = (
        class_counts["specimen_count"] >= minimum_specimens
    )

    eligible_labels = set(
        class_counts.loc[
            class_counts["eligible_for_cross_validation"],
            level,
        ]
    )

    eligible_specimens = specimen_table[
        specimen_table[level].isin(eligible_labels)
    ].copy()

    rare_specimens = specimen_table[
        ~specimen_table[level].isin(eligible_labels)
    ].copy()

    if not rare_specimens.empty:
        rare_specimens = rare_specimens.merge(
            class_counts[[level, "specimen_count"]],
            on=level,
            how="left",
        )

        rare_specimens["reason"] = (
            "Class has fewer than "
            + str(minimum_specimens)
            + " specimens required for cross-validation"
        )

    return eligible_specimens, rare_specimens, class_counts


def assign_stratified_folds(
    eligible_specimens: pd.DataFrame,
    level: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Assign stratified folds to one-row-per-specimen metadata."""

    if eligible_specimens.empty:
        raise ValueError(
            f"No {level} classes have enough specimens for "
            f"{n_splits}-fold cross-validation."
        )

    class_counts = eligible_specimens[level].value_counts()

    if class_counts.min() < n_splits:
        raise ValueError(
            "Fold generation received a class with fewer specimens "
            f"than n_splits={n_splits}:\n{class_counts}"
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    assignments = eligible_specimens.copy()
    assignments["fold"] = 0

    for fold_number, (_, validation_indices) in enumerate(
        splitter.split(
            assignments,
            assignments[level],
        ),
        start=1,
    ):
        assignments.loc[
            assignments.index[validation_indices],
            "fold",
        ] = fold_number

    assignments["fold"] = assignments["fold"].astype(int)

    if (assignments["fold"] == 0).any():
        raise RuntimeError(
            f"Some {level} specimens were not assigned to a fold."
        )

    return assignments


def merge_folds_with_images(
    dataframe: pd.DataFrame,
    specimen_assignments: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Merge specimen fold assignments back into image-level metadata."""

    labelled_images = dataframe[dataframe[level] != ""].copy()

    fold_images = labelled_images.merge(
        specimen_assignments[["numCol", level, "fold"]],
        on=["numCol", level],
        how="inner",
        validate="many_to_one",
    )

    fold_images = fold_images.sort_values(
        ["fold", level, "numCol"],
    ).reset_index(drop=True)

    return fold_images


def build_fold_summary(
    fold_images: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Build class counts for every fold."""

    summary = (
        fold_images.groupby(["fold", level])
        .agg(
            specimen_count=("numCol", "nunique"),
            image_count=("numCol", "size"),
        )
        .reset_index()
        .sort_values(["fold", level])
        .reset_index(drop=True)
    )

    return summary


def verify_no_specimen_leakage(
    fold_images: pd.DataFrame,
) -> None:
    """Confirm that each specimen appears in exactly one fold."""

    specimen_fold_counts = (
        fold_images.groupby("numCol")["fold"].nunique()
    )

    leaked_specimens = specimen_fold_counts[
        specimen_fold_counts > 1
    ]

    if not leaked_specimens.empty:
        raise RuntimeError(
            "Specimen leakage detected. Some specimens appear in "
            f"multiple folds: {leaked_specimens.index.tolist()[:10]}"
        )


def verify_fold_training_coverage(
    specimen_assignments: pd.DataFrame,
    level: str,
    n_splits: int,
) -> None:
    """Ensure validation classes are represented in training for each fold."""

    for fold_number in range(1, n_splits + 1):
        training_labels = set(
            specimen_assignments.loc[
                specimen_assignments["fold"] != fold_number,
                level,
            ]
        )

        validation_labels = set(
            specimen_assignments.loc[
                specimen_assignments["fold"] == fold_number,
                level,
            ]
        )

        missing_training_labels = validation_labels - training_labels

        if missing_training_labels:
            raise RuntimeError(
                f"{level.capitalize()} fold {fold_number} contains "
                "validation classes with no training specimens: "
                f"{sorted(missing_training_labels)}"
            )


def save_level_outputs(
    dataframe: pd.DataFrame,
    output_dir: Path,
    level: str,
    n_splits: int,
    random_state: int,
) -> dict[str, Any]:
    """Create and save all fold outputs for one taxonomic level."""

    specimen_table = build_specimen_table(
        dataframe=dataframe,
        level=level,
    )

    (
        eligible_specimens,
        rare_specimens,
        class_counts,
    ) = separate_eligible_and_rare_classes(
        specimen_table=specimen_table,
        level=level,
        minimum_specimens=n_splits,
    )

    assignments = assign_stratified_folds(
        eligible_specimens=eligible_specimens,
        level=level,
        n_splits=n_splits,
        random_state=random_state,
    )

    verify_fold_training_coverage(
        specimen_assignments=assignments,
        level=level,
        n_splits=n_splits,
    )

    fold_images = merge_folds_with_images(
        dataframe=dataframe,
        specimen_assignments=assignments,
        level=level,
    )

    verify_no_specimen_leakage(fold_images)

    fold_summary = build_fold_summary(
        fold_images=fold_images,
        level=level,
    )

    folds_path = output_dir / f"{level}_folds.csv"
    assignments_path = (
        output_dir / f"{level}_specimen_fold_assignments.csv"
    )
    summary_path = output_dir / f"{level}_fold_summary.csv"
    class_counts_path = output_dir / f"{level}_class_counts.csv"
    rare_path = (
        output_dir / f"{level}_excluded_rare_specimens.csv"
    )

    fold_images.to_csv(folds_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    fold_summary.to_csv(summary_path, index=False)
    class_counts.to_csv(class_counts_path, index=False)
    rare_specimens.to_csv(rare_path, index=False)

    LOGGER.info("")
    LOGGER.info("%s cross-validation created", level.capitalize())
    LOGGER.info("Eligible classes: %s", assignments[level].nunique())
    LOGGER.info(
        "Eligible specimens: %s",
        assignments["numCol"].nunique(),
    )
    LOGGER.info("Eligible images: %s", len(fold_images))
    LOGGER.info(
        "Excluded rare classes: %s",
        rare_specimens[level].nunique()
        if not rare_specimens.empty
        else 0,
    )
    LOGGER.info(
        "Excluded rare specimens: %s",
        rare_specimens["numCol"].nunique()
        if not rare_specimens.empty
        else 0,
    )
    LOGGER.info("Saved folds: %s", folds_path)
    LOGGER.info("Saved rare specimens: %s", rare_path)

    excluded_classes = []

    if not rare_specimens.empty:
        excluded_classes = (
            rare_specimens[
                [level, "specimen_count", "reason"]
            ]
            .drop_duplicates()
            .sort_values(
                ["specimen_count", level],
                ascending=[False, True],
            )
            .to_dict(orient="records")
        )

    return {
        "level": level,
        "n_splits": n_splits,
        "random_state": random_state,
        "minimum_specimens_per_class": n_splits,
        "input_labelled_specimens": int(
            specimen_table["numCol"].nunique()
        ),
        "eligible_classes": int(assignments[level].nunique()),
        "eligible_specimens": int(
            assignments["numCol"].nunique()
        ),
        "eligible_images": int(len(fold_images)),
        "excluded_classes_count": int(
            rare_specimens[level].nunique()
            if not rare_specimens.empty
            else 0
        ),
        "excluded_specimens_count": int(
            rare_specimens["numCol"].nunique()
            if not rare_specimens.empty
            else 0
        ),
        "excluded_classes": excluded_classes,
        "outputs": {
            "folds": str(folds_path),
            "specimen_assignments": str(assignments_path),
            "fold_summary": str(summary_path),
            "class_counts": str(class_counts_path),
            "excluded_rare_specimens": str(rare_path),
        },
    }


def create_cross_validation_folds(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Create cross-validation files for family and genus."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_supervised_dataset(input_path)

    summary: dict[str, Any] = {
        "input_file": str(input_path),
        "output_directory": str(output_dir),
        "n_splits": n_splits,
        "random_state": random_state,
        "total_input_images": int(len(dataframe)),
        "total_input_specimens": int(
            dataframe["numCol"].nunique()
        ),
        "levels": {},
    }

    for level in VALID_LEVELS:
        summary["levels"][level] = save_level_outputs(
            dataframe=dataframe,
            output_dir=output_dir,
            level=level,
            n_splits=n_splits,
            random_state=random_state,
        )

    summary_path = output_dir / "cross_validation_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    LOGGER.info("")
    LOGGER.info("Cross-validation completed successfully.")
    LOGGER.info("Summary saved to %s", summary_path)

    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create specimen-level stratified cross-validation "
            "folds for family and genus classification."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Input supervised dataset CSV. Default: "
            "metadata/supervised_dataset.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory. Default: "
            "metadata/cross_validation"
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="Number of cross-validation folds. Default: 5",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed. Default: 42",
    )

    return parser.parse_args()


def main() -> None:
    """Run cross-validation fold generation."""

    setup_logging()
    arguments = parse_arguments()

    create_cross_validation_folds(
        input_path=arguments.input,
        output_dir=arguments.output_dir,
        n_splits=arguments.n_splits,
        random_state=arguments.random_state,
    )


if __name__ == "__main__":
    main()
