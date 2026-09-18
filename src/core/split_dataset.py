"""
Dataset Splitter
================

Splits processed images into train/validation/test sets.

Important:
- Split is done by specimen_id, not by image.
- Rare classes with only one specimen are forced into train.
- This prevents rare species from disappearing from training.

Input:
- metadata/processed_dataset.csv

Outputs:
- metadata/train.csv
- metadata/validation.csv
- metadata/test.csv
- metadata/split_summary.json
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class DatasetSplitter:
    """Split processed dataset by specimen."""

    def __init__(self) -> None:
        project_config = CONFIG.project()

        self.input_csv = Path("metadata/processed_dataset.csv")

        self.train_csv = Path("metadata/train.csv")
        self.validation_csv = Path("metadata/validation.csv")
        self.test_csv = Path("metadata/test.csv")
        self.summary_json = Path("metadata/split_summary.json")

        self.random_seed = int(project_config["project"]["random_seed"])

        self.train_ratio = 0.70
        self.validation_ratio = 0.15
        self.test_ratio = 0.15

        self.data: pd.DataFrame | None = None
        self.train_data: pd.DataFrame | None = None
        self.validation_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None

    def load_dataset(self) -> None:
        """Load processed dataset metadata."""
        logger.info("Loading processed dataset: %s", self.input_csv)

        if not self.input_csv.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {self.input_csv}. "
                "Run preprocessing first."
            )

        self.data = pd.read_csv(self.input_csv)

        required_columns = [
            "specimen_id",
            "label",
            "processed_image_path",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise KeyError(
                f"Missing required columns in processed dataset: {missing_columns}"
            )

        logger.info("Loaded %d processed images.", len(self.data))

    def split_by_specimen(self) -> None:
        """Split dataset by specimen while keeping rare labels in train."""
        assert self.data is not None

        specimen_table = (
            self.data[["specimen_id", "label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        label_counts = specimen_table["label"].value_counts()

        rare_labels = set(label_counts[label_counts < 2].index)

        rare_specimens = specimen_table[
            specimen_table["label"].isin(rare_labels)
        ]

        splittable_specimens = specimen_table[
            ~specimen_table["label"].isin(rare_labels)
        ]

        logger.info("Total specimens: %d", len(specimen_table))
        logger.info("Rare labels forced to train: %s", sorted(rare_labels))
        logger.info("Splittable specimens: %d", len(splittable_specimens))

        if len(splittable_specimens) < 3:
            logger.warning(
                "Too few splittable specimens. All specimens will be assigned to train."
            )

            train_specimens = specimen_table
            validation_specimens = specimen_table.iloc[0:0]
            test_specimens = specimen_table.iloc[0:0]

        else:
            train_splittable, temp_specimens = train_test_split(
                splittable_specimens,
                test_size=(1 - self.train_ratio),
                random_state=self.random_seed,
                stratify=(
                    splittable_specimens["label"]
                    if splittable_specimens["label"].value_counts().min() >= 2
                    else None
                ),
            )

            if len(temp_specimens) < 2:
                validation_specimens = temp_specimens
                test_specimens = temp_specimens.iloc[0:0]
            else:
                relative_test_ratio = self.test_ratio / (
                    self.validation_ratio + self.test_ratio
                )

                validation_specimens, test_specimens = train_test_split(
                    temp_specimens,
                    test_size=relative_test_ratio,
                    random_state=self.random_seed,
                    stratify=(
                        temp_specimens["label"]
                        if temp_specimens["label"].value_counts().min() >= 2
                        else None
                    ),
                )

            train_specimens = pd.concat(
                [train_splittable, rare_specimens],
                ignore_index=True,
            )

        train_ids = set(train_specimens["specimen_id"])
        validation_ids = set(validation_specimens["specimen_id"])
        test_ids = set(test_specimens["specimen_id"])

        self.train_data = self.data[
            self.data["specimen_id"].isin(train_ids)
        ].copy()

        self.validation_data = self.data[
            self.data["specimen_id"].isin(validation_ids)
        ].copy()

        self.test_data = self.data[
            self.data["specimen_id"].isin(test_ids)
        ].copy()

        self.train_data["split"] = "train"
        self.validation_data["split"] = "validation"
        self.test_data["split"] = "test"

        logger.info("Train images: %d", len(self.train_data))
        logger.info("Validation images: %d", len(self.validation_data))
        logger.info("Test images: %d", len(self.test_data))

    def save_splits(self) -> None:
        """Save split CSV files."""
        assert self.train_data is not None
        assert self.validation_data is not None
        assert self.test_data is not None

        self.train_csv.parent.mkdir(parents=True, exist_ok=True)

        self.train_data.to_csv(self.train_csv, index=False)
        self.validation_data.to_csv(self.validation_csv, index=False)
        self.test_data.to_csv(self.test_csv, index=False)

        logger.info("Saved train split: %s", self.train_csv)
        logger.info("Saved validation split: %s", self.validation_csv)
        logger.info("Saved test split: %s", self.test_csv)

    def save_summary(self) -> None:
        """Save split summary."""
        assert self.train_data is not None
        assert self.validation_data is not None
        assert self.test_data is not None
        assert self.data is not None

        summary = {
            "total_images": int(len(self.data)),
            "train_images": int(len(self.train_data)),
            "validation_images": int(len(self.validation_data)),
            "test_images": int(len(self.test_data)),
            "train_specimens": int(self.train_data["specimen_id"].nunique()),
            "validation_specimens": int(
                self.validation_data["specimen_id"].nunique()
            ),
            "test_specimens": int(self.test_data["specimen_id"].nunique()),
            "train_labels": int(self.train_data["label"].nunique()),
            "validation_labels": int(self.validation_data["label"].nunique()),
            "test_labels": int(self.test_data["label"].nunique()),
            "split_strategy": "by_specimen_id_with_rare_labels_forced_to_train",
            "random_seed": self.random_seed,
        }

        with open(self.summary_json, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        logger.info("Saved split summary: %s", self.summary_json)

    def run(self) -> None:
        """Run split pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Dataset Split")
        logger.info("=" * 60)

        self.load_dataset()
        self.split_by_specimen()
        self.save_splits()
        self.save_summary()

        logger.info("Dataset split completed successfully.")


if __name__ == "__main__":
    setup_logging()

    splitter = DatasetSplitter()
    splitter.run()
