"""PyTorch dataset utilities for Brachycera image classification.

This module reads the cross-validation CSV files produced by:

    src/core/cross_validation.py

It supports:

- family classification;
- genus classification;
- FDT, FFF, FLP and FLT anatomical views;
- specimen-level cross-validation folds;
- separate training and validation subsets;
- optional image transformations;
- class-to-index mappings;
- strict path and label validation.

Expected input files
--------------------
metadata/cross_validation/family_folds.csv
metadata/cross_validation/genus_folds.csv

Typical use
-----------
from src.ml.dataset import create_datasets

train_dataset, validation_dataset, class_to_idx = create_datasets(
    level="family",
    view_code="FDT",
    validation_fold=1,
    train_transform=train_transform,
    validation_transform=validation_transform,
)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset


# ============================================================
# TYPES
# ============================================================

TaxonomicLevel = Literal["family", "genus"]
DatasetSubset = Literal["train", "validation"]

VALID_LEVELS = {"family", "genus"}
VALID_SUBSETS = {"train", "validation"}
VALID_VIEWS = {"FDT", "FFF", "FLP", "FLT"}


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROSS_VALIDATION_ROOT = (
    PROJECT_ROOT / "metadata" / "cross_validation"
)

FAMILY_FOLDS_PATH = (
    CROSS_VALIDATION_ROOT / "family_folds.csv"
)

GENUS_FOLDS_PATH = (
    CROSS_VALIDATION_ROOT / "genus_folds.csv"
)

CLASS_MAPPING_ROOT = (
    PROJECT_ROOT / "metadata" / "class_mappings"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_level(level: str) -> TaxonomicLevel:
    """Normalize and validate a taxonomic level."""

    normalized = str(level).strip().lower()

    if normalized not in VALID_LEVELS:
        raise ValueError(
            f"Invalid taxonomic level: {level!r}. "
            f"Expected one of {sorted(VALID_LEVELS)}."
        )

    return normalized  # type: ignore[return-value]


def normalize_subset(subset: str) -> DatasetSubset:
    """Normalize and validate a dataset subset."""

    normalized = str(subset).strip().lower()

    if normalized not in VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset: {subset!r}. "
            f"Expected one of {sorted(VALID_SUBSETS)}."
        )

    return normalized  # type: ignore[return-value]


def normalize_view_code(view_code: str) -> str:
    """Normalize and validate an anatomical view code."""

    normalized = str(view_code).strip().upper()

    if normalized not in VALID_VIEWS:
        raise ValueError(
            f"Invalid view code: {view_code!r}. "
            f"Expected one of {sorted(VALID_VIEWS)}."
        )

    return normalized


def get_fold_file(level: TaxonomicLevel) -> Path:
    """Return the correct fold CSV for a taxonomic level."""

    if level == "family":
        return FAMILY_FOLDS_PATH

    return GENUS_FOLDS_PATH


def resolve_image_path(path_value: Any) -> Path:
    """Resolve a CSV image path against the project root.

    The preprocessing pipeline normally stores relative paths such as:

        processed/images/IHMT-E52334/FDT.jpg

    Absolute paths are also supported.
    """

    path = Path(str(path_value).strip())

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def build_class_mapping(
    dataframe: pd.DataFrame,
    label_column: str,
) -> dict[str, int]:
    """Create a deterministic alphabetical class-to-index mapping."""

    labels = sorted(
        dataframe[label_column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not labels:
        raise ValueError(
            f"No labels were found in column {label_column!r}."
        )

    return {
        label: index
        for index, label in enumerate(labels)
    }


def save_class_mapping(
    class_to_idx: dict[str, int],
    level: TaxonomicLevel,
    view_code: str,
) -> Path:
    """Save a class mapping for reproducible model training."""

    CLASS_MAPPING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        CLASS_MAPPING_ROOT
        / f"{level}_{view_code}_class_mapping.json"
    )

    payload = {
        "taxonomic_level": level,
        "view_code": view_code,
        "number_of_classes": len(class_to_idx),
        "class_to_idx": class_to_idx,
        "idx_to_class": {
            str(index): label
            for label, index in class_to_idx.items()
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# DATASET
# ============================================================

class BrachyceraImageDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset for one taxonomic level, view and fold subset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_column: TaxonomicLevel,
        class_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Tensor] | None = None,
        return_metadata: bool = False,
    ) -> None:
        """Initialize the image dataset.

        Parameters
        ----------
        dataframe:
            Filtered image-level dataframe.
        label_column:
            Either ``family`` or ``genus``.
        class_to_idx:
            Mapping from taxonomic label to integer class index.
        transform:
            Optional torchvision-style image transformation.
        return_metadata:
            When True, __getitem__ returns a metadata dictionary in
            addition to the image tensor and target.
        """

        if dataframe.empty:
            raise ValueError(
                "Cannot create a dataset from an empty dataframe."
            )

        required_columns = {
            "numCol",
            label_column,
            "view_code",
            "fold",
            "processed_image_path",
        }

        missing_columns = (
            required_columns - set(dataframe.columns)
        )

        if missing_columns:
            raise KeyError(
                "Dataset dataframe is missing columns: "
                f"{sorted(missing_columns)}"
            )

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.label_column = label_column
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata

        self.idx_to_class = {
            index: label
            for label, index in class_to_idx.items()
        }

        self._validate_labels()
        self._validate_image_paths()

    def _validate_labels(self) -> None:
        """Ensure all labels exist in the mapping."""

        labels = set(
            self.dataframe[self.label_column]
            .astype(str)
            .str.strip()
            .tolist()
        )

        unknown_labels = labels - set(
            self.class_to_idx
        )

        if unknown_labels:
            raise ValueError(
                "The dataset contains labels absent from "
                f"class_to_idx: {sorted(unknown_labels)}"
            )

    def _validate_image_paths(self) -> None:
        """Ensure all processed image files exist."""

        missing_paths: list[str] = []

        for path_value in self.dataframe[
            "processed_image_path"
        ]:
            image_path = resolve_image_path(path_value)

            if not image_path.exists():
                missing_paths.append(str(image_path))

                if len(missing_paths) >= 10:
                    break

        if missing_paths:
            raise FileNotFoundError(
                "Some processed images do not exist. "
                f"Examples: {missing_paths}"
            )

    def __len__(self) -> int:
        """Return the number of image rows."""

        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> (
        tuple[Tensor, int]
        | tuple[Tensor, int, dict[str, Any]]
    ):
        """Load one image and its target class."""

        row = self.dataframe.iloc[index]

        image_path = resolve_image_path(
            row["processed_image_path"]
        )

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image_tensor = self.transform(image)
                else:
                    image_tensor = self._pil_to_tensor(
                        image
                    )

        except UnidentifiedImageError as error:
            raise RuntimeError(
                f"Cannot identify image: {image_path}"
            ) from error

        except OSError as error:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            ) from error

        label = str(
            row[self.label_column]
        ).strip()

        target = self.class_to_idx[label]

        if not self.return_metadata:
            return image_tensor, target

        metadata = {
            "numCol": str(row["numCol"]),
            "label": label,
            "view_code": str(row["view_code"]),
            "fold": int(row["fold"]),
            "processed_image_path": str(
                row["processed_image_path"]
            ),
        }

        return image_tensor, target, metadata

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> Tensor:
        """Convert a PIL image to a float tensor without torchvision."""

        byte_tensor = torch.ByteTensor(
            torch.ByteStorage.from_buffer(
                image.tobytes()
            )
        )

        tensor = byte_tensor.view(
            image.height,
            image.width,
            3,
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        return tensor.float().div(255.0)

    def get_dataframe(self) -> pd.DataFrame:
        """Return a copy of the filtered dataset table."""

        return self.dataframe.copy()

    def get_class_counts(self) -> pd.Series:
        """Return image counts per class."""

        return (
            self.dataframe[self.label_column]
            .value_counts()
            .sort_index()
        )

    def get_specimen_counts(self) -> pd.Series:
        """Return specimen counts per class."""

        specimen_table = self.dataframe.drop_duplicates(
            subset=["numCol"]
        )

        return (
            specimen_table[self.label_column]
            .value_counts()
            .sort_index()
        )


# ============================================================
# DATASET FACTORY
# ============================================================

def load_fold_dataframe(
    level: str,
) -> pd.DataFrame:
    """Load a family or genus fold assignment CSV."""

    normalized_level = normalize_level(level)
    fold_path = get_fold_file(normalized_level)

    if not fold_path.exists():
        raise FileNotFoundError(
            f"Cross-validation fold file not found: {fold_path}"
        )

    dataframe = pd.read_csv(fold_path)

    required_columns = {
        "numCol",
        normalized_level,
        "view_code",
        "fold",
        "processed_image_path",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{fold_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["view_code"] = (
        dataframe["view_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe[normalized_level] = (
        dataframe[normalized_level]
        .astype(str)
        .str.strip()
    )

    dataframe["fold"] = pd.to_numeric(
        dataframe["fold"],
        errors="raise",
    ).astype(int)

    return dataframe


def filter_fold_dataframe(
    dataframe: pd.DataFrame,
    level: str,
    view_code: str,
    validation_fold: int,
    subset: str,
) -> pd.DataFrame:
    """Filter by anatomical view and cross-validation subset."""

    normalized_level = normalize_level(level)
    normalized_view = normalize_view_code(
        view_code
    )
    normalized_subset = normalize_subset(subset)

    available_folds = sorted(
        dataframe["fold"].unique().tolist()
    )

    if validation_fold not in available_folds:
        raise ValueError(
            f"Fold {validation_fold} is unavailable. "
            f"Available folds: {available_folds}"
        )

    view_data = dataframe.loc[
        dataframe["view_code"] == normalized_view
    ].copy()

    if view_data.empty:
        available_views = sorted(
            dataframe["view_code"]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"No {normalized_level} images were found for "
            f"view {normalized_view}. "
            f"Available views: {available_views}"
        )

    if normalized_subset == "train":
        subset_data = view_data.loc[
            view_data["fold"] != validation_fold
        ].copy()
    else:
        subset_data = view_data.loc[
            view_data["fold"] == validation_fold
        ].copy()

    if subset_data.empty:
        raise ValueError(
            f"The {normalized_subset} subset is empty for "
            f"level={normalized_level}, "
            f"view={normalized_view}, "
            f"validation_fold={validation_fold}."
        )

    subset_data = subset_data.sort_values(
        by=[
            "fold",
            normalized_level,
            "numCol",
            "processed_image_path",
        ]
    ).reset_index(drop=True)

    return subset_data


def create_datasets(
    level: str,
    view_code: str,
    validation_fold: int,
    train_transform: Callable[
        [Image.Image],
        Tensor,
    ] | None = None,
    validation_transform: Callable[
        [Image.Image],
        Tensor,
    ] | None = None,
    return_metadata: bool = False,
    save_mapping: bool = True,
) -> tuple[
    BrachyceraImageDataset,
    BrachyceraImageDataset,
    dict[str, int],
]:
    """Create training and validation datasets for one experiment.

    The class mapping is built from the complete selected-view dataset,
    ensuring that training and validation use identical class indices.

    All images from the same specimen remain in the same fold because
    the fold assignments were previously generated with
    StratifiedGroupKFold.
    """

    normalized_level = normalize_level(level)
    normalized_view = normalize_view_code(
        view_code
    )

    dataframe = load_fold_dataframe(
        level=normalized_level
    )

    selected_view_data = dataframe.loc[
        dataframe["view_code"] == normalized_view
    ].copy()

    if selected_view_data.empty:
        raise ValueError(
            f"No records exist for view {normalized_view} "
            f"at level {normalized_level}."
        )

    class_to_idx = build_class_mapping(
        dataframe=selected_view_data,
        label_column=normalized_level,
    )

    train_data = filter_fold_dataframe(
        dataframe=dataframe,
        level=normalized_level,
        view_code=normalized_view,
        validation_fold=validation_fold,
        subset="train",
    )

    validation_data = filter_fold_dataframe(
        dataframe=dataframe,
        level=normalized_level,
        view_code=normalized_view,
        validation_fold=validation_fold,
        subset="validation",
    )

    train_specimens = set(
        train_data["numCol"].astype(str)
    )

    validation_specimens = set(
        validation_data["numCol"].astype(str)
    )

    overlap = (
        train_specimens
        & validation_specimens
    )

    if overlap:
        raise RuntimeError(
            "Specimen leakage detected between training and "
            f"validation datasets: {sorted(overlap)[:10]}"
        )

    validation_labels = set(
        validation_data[normalized_level]
        .astype(str)
        .str.strip()
    )

    training_labels = set(
        train_data[normalized_level]
        .astype(str)
        .str.strip()
    )

    validation_only_labels = (
        validation_labels - training_labels
    )

    if validation_only_labels:
        raise ValueError(
            "The validation fold contains classes with no "
            "training examples. This model cannot learn these "
            f"classes: {sorted(validation_only_labels)}"
        )

    train_dataset = BrachyceraImageDataset(
        dataframe=train_data,
        label_column=normalized_level,
        class_to_idx=class_to_idx,
        transform=train_transform,
        return_metadata=return_metadata,
    )

    validation_dataset = (
        BrachyceraImageDataset(
            dataframe=validation_data,
            label_column=normalized_level,
            class_to_idx=class_to_idx,
            transform=validation_transform,
            return_metadata=return_metadata,
        )
    )

    if save_mapping:
        save_class_mapping(
            class_to_idx=class_to_idx,
            level=normalized_level,
            view_code=normalized_view,
        )

    return (
        train_dataset,
        validation_dataset,
        class_to_idx,
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main() -> None:
    """Run a simple dataset integrity test."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Test the Brachycera PyTorch dataset loader."
        )
    )

    parser.add_argument(
        "--level",
        choices=sorted(VALID_LEVELS),
        required=True,
        help="Taxonomic classification level.",
    )

    parser.add_argument(
        "--view",
        choices=sorted(VALID_VIEWS),
        required=True,
        help="Anatomical image view.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Validation fold number.",
    )

    parser.add_argument(
        "--return-metadata",
        action="store_true",
        help="Return sample metadata.",
    )

    arguments = parser.parse_args()

    train_dataset, validation_dataset, class_to_idx = (
        create_datasets(
            level=arguments.level,
            view_code=arguments.view,
            validation_fold=arguments.fold,
            return_metadata=arguments.return_metadata,
        )
    )

    print()
    print("Dataset configuration")
    print("---------------------")
    print(f"Taxonomic level: {arguments.level}")
    print(f"View: {arguments.view}")
    print(f"Validation fold: {arguments.fold}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Training images: {len(train_dataset)}")
    print(
        "Training specimens: "
        f"{train_dataset.get_dataframe()['numCol'].nunique()}"
    )
    print(
        f"Validation images: {len(validation_dataset)}"
    )
    print(
        "Validation specimens: "
        f"{validation_dataset.get_dataframe()['numCol'].nunique()}"
    )

    sample = train_dataset[0]

    if arguments.return_metadata:
        image_tensor, target, metadata = sample
        print(f"Sample metadata: {metadata}")
    else:
        image_tensor, target = sample

    print(f"Sample tensor shape: {tuple(image_tensor.shape)}")
    print(f"Sample target index: {target}")
    print(
        "Sample target label: "
        f"{train_dataset.idx_to_class[target]}"
    )
    print()
    print("Dataset integrity test completed successfully.")


if __name__ == "__main__":
    main()
