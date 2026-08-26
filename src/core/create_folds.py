"""Create specimen-level cross-validation folds.

This module creates reproducible stratified folds for family and genus
classification.

Each specimen is assigned to exactly one fold. All anatomical views from
the same specimen therefore remain in the same fold, preventing data
leakage between training and validation datasets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.config import CONFIG
from src.utils.logger import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)


MISSING_LABELS = {
    "",
    "unknown",
    "unlabelled",
    "unidentified",
    "nan",
    "none",
    "not identified",
}


def normalise_column_name(column_name: str) -> str:
    """Return a normalised version of a dataframe column name."""

    return re.sub(r"[^a-z0-9]", "", str(column_name).lower())


def find_column(
    dataframe: pd.DataFrame,
    requested_column: str,
) -> str:
    """Find a dataframe column using case-insensitive matching.

    Parameters
    ----------
    dataframe:
        Input dataframe.
    requested_column:
        Expected column name.

    Returns
    -------
    str
        The actual dataframe column name.

    Raises
    ------
    KeyError
        If the requested column cannot be located.
    """

    normalised_requested = normalise_column_name(requested_column)

    column_map = {
        normalise_column_name(column): column
        for column in dataframe.columns
    }

    if normalised_requested not in column_map:
        raise KeyError(
            f"Column '{requested_column}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return column_map[normalised_requested]


def clean_label(value: Any) -> str | None:
    """Clean a taxonomic label.

    Missing, unknown and unidentified values are returned as ``None``.
    """

    if pd.isna(value):
        return None

    cleaned = str(value).strip()

    if cleaned.lower() in MISSING_LABELS:
        return None

    return cleaned


class CrossValidationFoldBuilder:
    """Create stratified specimen-level cross-validation folds."""

    def __init__(self) -> None:
        """Initialise configuration values and paths."""

        project_config = CONFIG.project()
        path_config = CONFIG.path()

        training_config = project_config["training"]
        cross_validation_config = project_config["cross_validation"]
        dataset_config = project_config["dataset"]

        self.classification_levels = training_config[
            "classification_levels"
        ]

        self.requested_splits = int(
            cross_validation_config["n_splits"]
        )

        self.shuffle = bool(
            cross_validation_config.get("shuffle", True)
        )

        self.random_seed = int(
            cross_validation_config.get(
                "random_seed",
                project_config["project"].get("random_seed", 42),
            )
        )

        self.minimum_specimens_per_class = int(
            cross_validation_config.get(
                "minimum_specimens_per_class",
                self.requested_splits,
            )
        )

        self.specimen_id_column = dataset_config[
            "specimen_id_column"
        ]

        self.taxonomic_columns = dataset_config[
            "taxonomic_columns"
        ]

        self.processed_dataset_path = Path(
            path_config["metadata"]["processed_dataset"]
        )

        self.output_directory = Path(
            path_config["metadata"]["splits_root"]
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_dataset(self) -> pd.DataFrame:
        """Load the processed image dataset."""

        if not self.processed_dataset_path.exists():
            raise FileNotFoundError(
                "Processed dataset not found: "
                f"{self.processed_dataset_path}"
            )

        dataframe = pd.read_csv(self.processed_dataset_path)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {self.processed_dataset_path}"
            )

        logger.info(
            "Loaded processed dataset with %s rows.",
            len(dataframe),
        )

        return dataframe

    def prepare_specimen_table(
        self,
        dataframe: pd.DataFrame,
        classification_level: str,
    ) -> pd.DataFrame:
        """Create one labelled row per specimen.

        The fold assignment is performed at specimen level rather than
        image level.
        """

        if classification_level not in self.taxonomic_columns:
            raise KeyError(
                f"No taxonomic column configured for "
                f"'{classification_level}'."
            )

        specimen_column = find_column(
            dataframe,
            self.specimen_id_column,
        )

        label_column = find_column(
            dataframe,
            self.taxonomic_columns[classification_level],
        )

        specimen_table = dataframe[
            [specimen_column, label_column]
        ].copy()

        specimen_table.columns = [
            "specimen_id",
            "label",
        ]

        specimen_table["specimen_id"] = (
            specimen_table["specimen_id"]
            .astype(str)
            .str.strip()
        )

        specimen_table["label"] = specimen_table[
            "label"
        ].apply(clean_label)

        specimen_table = specimen_table.dropna(
            subset=["specimen_id", "label"]
        )

        specimen_table = specimen_table[
            specimen_table["specimen_id"] != ""
        ]

        duplicate_labels = (
            specimen_table.groupby("specimen_id")["label"]
            .nunique()
        )

        inconsistent_specimens = duplicate_labels[
            duplicate_labels > 1
        ].index.tolist()

        if inconsistent_specimens:
            example_ids = inconsistent_specimens[:10]

            raise ValueError(
                "Some specimens have more than one "
                f"{classification_level} label. Examples: "
                f"{example_ids}"
            )

        specimen_table = specimen_table.drop_duplicates(
            subset=["specimen_id"]
        )

        specimen_table = specimen_table.reset_index(drop=True)

        logger.info(
            "%s: %s labelled specimens before class filtering.",
            classification_level,
            len(specimen_table),
        )

        return specimen_table

    def filter_rare_classes(
        self,
        specimen_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Remove classes with insufficient specimens.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            The retained specimen table and the class-count table.
        """

        class_counts = (
            specimen_table.groupby("label")["specimen_id"]
            .nunique()
            .sort_values(ascending=False)
            .rename("specimen_count")
            .reset_index()
        )

        retained_classes = class_counts.loc[
            class_counts["specimen_count"]
            >= self.minimum_specimens_per_class,
            "label",
        ].tolist()

        excluded_classes = class_counts.loc[
            class_counts["specimen_count"]
            < self.minimum_specimens_per_class
        ]

        if not excluded_classes.empty:
            logger.warning(
                "Excluding %s classes with fewer than %s specimens.",
                len(excluded_classes),
                self.minimum_specimens_per_class,
            )

            for row in excluded_classes.itertuples(index=False):
                logger.warning(
                    "Excluded class '%s': %s specimens.",
                    row.label,
                    row.specimen_count,
                )

        filtered_table = specimen_table[
            specimen_table["label"].isin(retained_classes)
        ].copy()

        filtered_table = filtered_table.reset_index(drop=True)

        return filtered_table, class_counts

    def determine_number_of_splits(
        self,
        specimen_table: pd.DataFrame,
    ) -> int:
        """Determine the valid number of cross-validation folds."""

        if specimen_table.empty:
            raise ValueError(
                "No specimens remain after filtering rare classes."
            )

        minimum_class_size = int(
            specimen_table.groupby("label")["specimen_id"]
            .nunique()
            .min()
        )

        number_of_splits = min(
            self.requested_splits,
            minimum_class_size,
        )

        if number_of_splits < 2:
            raise ValueError(
                "Cross-validation requires at least two specimens "
                "in every retained class."
            )

        if number_of_splits < self.requested_splits:
            logger.warning(
                "Reducing folds from %s to %s because the smallest "
                "class has only %s specimens.",
                self.requested_splits,
                number_of_splits,
                minimum_class_size,
            )

        return number_of_splits

    def assign_folds(
        self,
        specimen_table: pd.DataFrame,
        number_of_splits: int,
    ) -> pd.DataFrame:
        """Assign every specimen to a cross-validation fold."""

        fold_builder = StratifiedGroupKFold(
            n_splits=number_of_splits,
            shuffle=self.shuffle,
            random_state=self.random_seed,
        )

        result = specimen_table.copy()
        result["fold"] = -1

        labels = result["label"]
        groups = result["specimen_id"]

        for fold_number, (_, validation_indices) in enumerate(
            fold_builder.split(
                X=result,
                y=labels,
                groups=groups,
            )
        ):
            result.loc[
                validation_indices,
                "fold",
            ] = fold_number

        if (result["fold"] < 0).any():
            raise RuntimeError(
                "At least one specimen was not assigned to a fold."
            )

        result["fold"] = result["fold"].astype(int)

        return result

    def validate_folds(
        self,
        folds: pd.DataFrame,
        number_of_splits: int,
    ) -> None:
        """Validate the generated fold assignment."""

        duplicated_specimens = folds[
            "specimen_id"
        ].duplicated()

        if duplicated_specimens.any():
            raise ValueError(
                "A specimen occurs more than once in the fold table."
            )

        observed_folds = sorted(
            folds["fold"].unique().tolist()
        )

        expected_folds = list(range(number_of_splits))

        if observed_folds != expected_folds:
            raise ValueError(
                f"Unexpected folds. Expected {expected_folds}, "
                f"found {observed_folds}."
            )

        fold_class_counts = (
            folds.groupby(["fold", "label"])["specimen_id"]
            .nunique()
            .unstack(fill_value=0)
        )

        missing_classes = fold_class_counts.columns[
            (fold_class_counts == 0).any(axis=0)
        ].tolist()

        if missing_classes:
            logger.warning(
                "The following classes are absent from at least "
                "one validation fold: %s",
                missing_classes,
            )

    def save_outputs(
        self,
        classification_level: str,
        folds: pd.DataFrame,
        class_counts: pd.DataFrame,
        number_of_splits: int,
    ) -> None:
        """Save fold assignments and summary information."""

        fold_path = (
            self.output_directory
            / f"{classification_level}_folds.csv"
        )

        class_count_path = (
            self.output_directory
            / f"{classification_level}_class_counts.csv"
        )

        summary_path = (
            self.output_directory
            / f"{classification_level}_fold_summary.json"
        )

        folds_output = folds.copy()
        folds_output.insert(
            1,
            "classification_level",
            classification_level,
        )

        folds_output.to_csv(
            fold_path,
            index=False,
        )

        class_counts.to_csv(
            class_count_path,
            index=False,
        )

        fold_sizes = (
            folds.groupby("fold")["specimen_id"]
            .nunique()
            .to_dict()
        )

        summary = {
            "classification_level": classification_level,
            "number_of_folds": number_of_splits,
            "number_of_specimens": int(
                folds["specimen_id"].nunique()
            ),
            "number_of_classes": int(
                folds["label"].nunique()
            ),
            "random_seed": self.random_seed,
            "minimum_specimens_per_class": (
                self.minimum_specimens_per_class
            ),
            "fold_sizes": {
                str(key): int(value)
                for key, value in fold_sizes.items()
            },
            "classes": sorted(
                folds["label"].unique().tolist()
            ),
        }

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as summary_file:
            json.dump(
                summary,
                summary_file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "%s fold assignments saved to %s.",
            classification_level,
            fold_path,
        )

    def build_level(
        self,
        dataframe: pd.DataFrame,
        classification_level: str,
    ) -> None:
        """Generate folds for one taxonomic classification level."""

        logger.info(
            "Creating folds for classification level: %s",
            classification_level,
        )

        specimen_table = self.prepare_specimen_table(
            dataframe=dataframe,
            classification_level=classification_level,
        )

        filtered_table, class_counts = self.filter_rare_classes(
            specimen_table
        )

        number_of_splits = self.determine_number_of_splits(
            filtered_table
        )

        folds = self.assign_folds(
            specimen_table=filtered_table,
            number_of_splits=number_of_splits,
        )

        self.validate_folds(
            folds=folds,
            number_of_splits=number_of_splits,
        )

        self.save_outputs(
            classification_level=classification_level,
            folds=folds,
            class_counts=class_counts,
            number_of_splits=number_of_splits,
        )

        logger.info(
            "%s completed: %s specimens, %s classes, %s folds.",
            classification_level,
            folds["specimen_id"].nunique(),
            folds["label"].nunique(),
            number_of_splits,
        )

    def run(self) -> None:
        """Create fold assignments for every classification level."""

        dataframe = self.load_dataset()

        for classification_level in self.classification_levels:
            self.build_level(
                dataframe=dataframe,
                classification_level=classification_level,
            )

        logger.info(
            "Cross-validation fold generation completed."
        )


def main() -> None:
    """Run cross-validation fold generation."""

    builder = CrossValidationFoldBuilder()
    builder.run()


if __name__ == "__main__":
    main()
