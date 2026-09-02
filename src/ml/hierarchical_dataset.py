"""Dataset utilities for hierarchical family-genus classification.


"""
# import modules
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


# ============================================================
# PROJECT PATHS
# ============================================================
# creates the project path for hierarchical classification
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IDENTIFICATION_DATASET = (
    PROJECT_ROOT
    / "metadata"
    / "identification"
    / "identification_dataset.csv"
)

DEFAULT_MAPPING_DIRECTORY = (
    PROJECT_ROOT
    / "metadata"
    / "class_mappings"
    / "hierarchical"
)


# ============================================================
# MISSING VALUES
# ============================================================

MISSING_VALUES = {
    "",
    "-",
    "--",
    "nan",
    "none",
    "null",
    "unknown",
    "unidentified",
    "unlabelled",
    "unlabeled",
}


def clean_taxonomic_value(value: Any) -> str | None:
    """Return a normalized taxonomy value or None."""

    if pd.isna(value):
        return None

    cleaned = str(value).strip()

    if cleaned.lower() in MISSING_VALUES:
        return None

    return cleaned


def normalize_boolean(value: Any) -> bool:
    """Convert common representations to bool."""

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "sim",
    }


def resolve_image_path(path_value: Any) -> Path:
    """Resolve a processed image path against the project root."""

    path = Path(str(path_value).strip())

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


# ============================================================
# TAXONOMY MAPPINGS
# ============================================================
# creates a taxanomy mappings hierarchical classification
def build_class_mapping(
    values: pd.Series,
) -> dict[str, int]:
    """Build a deterministic alphabetical class mapping."""

    labels = sorted(
        {
            cleaned
            for value in values
            if (cleaned := clean_taxonomic_value(value))
            is not None
        }
    )

    if not labels:
        raise ValueError(
            "No valid taxonomic labels were available."
        )

    return {
        label: index
        for index, label in enumerate(labels)
    }


def build_genus_to_family_mapping(
    dataframe: pd.DataFrame,
    family_to_idx: dict[str, int],
    genus_to_idx: dict[str, int],
) -> dict[int, int]:
    """Map every genus index to exactly one family index.

    Raises
    ------
    ValueError
        If one genus is associated with multiple families in the
        metadata.
    """

    labelled = dataframe[
        dataframe["family"].notna()
        & dataframe["genus"].notna()
    ].copy()

    genus_to_family: dict[int, int] = {}

    for genus, group in labelled.groupby("genus"):
        families = sorted(
            group["family"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        if len(families) != 1:
            raise ValueError(
                f"Genus {genus!r} is associated with "
                f"multiple families: {families}"
            )

        family = families[0]

        genus_index = genus_to_idx[genus]
        family_index = family_to_idx[family]

        genus_to_family[genus_index] = family_index

    if len(genus_to_family) != len(genus_to_idx):
        missing_genus_indices = (
            set(genus_to_idx.values())
            - set(genus_to_family)
        )

        raise ValueError(
            "Some genera could not be associated with a family. "
            f"Missing genus indices: "
            f"{sorted(missing_genus_indices)}"
        )

    return genus_to_family


def build_family_to_genera_mapping(
    genus_to_family: dict[int, int],
) -> dict[int, list[int]]:
    """Return the allowed genus indices for every family."""

    family_to_genera: dict[int, list[int]] = {}

    for genus_index, family_index in genus_to_family.items():
        family_to_genera.setdefault(
            family_index,
            [],
        ).append(genus_index)

    for family_index in family_to_genera:
        family_to_genera[family_index] = sorted(
            family_to_genera[family_index]
        )

    return family_to_genera


def save_hierarchical_mappings(
    family_to_idx: dict[str, int],
    genus_to_idx: dict[str, int],
    genus_to_family: dict[int, int],
    output_directory: Path = DEFAULT_MAPPING_DIRECTORY,
) -> Path:
    """Save all mappings needed for training and prediction."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_to_genera = build_family_to_genera_mapping(
        genus_to_family
    )

    output_path = (
        output_directory / "taxonomy_mappings.json"
    )

    payload = {
        "number_of_families": len(family_to_idx),
        "number_of_genera": len(genus_to_idx),
        "family_to_idx": family_to_idx,
        "idx_to_family": {
            str(index): family
            for family, index in family_to_idx.items()
        },
        "genus_to_idx": genus_to_idx,
        "idx_to_genus": {
            str(index): genus
            for genus, index in genus_to_idx.items()
        },
        "genus_to_family_idx": {
            str(genus_index): family_index
            for genus_index, family_index
            in genus_to_family.items()
        },
        "family_to_genus_indices": {
            str(family_index): genus_indices
            for family_index, genus_indices
            in family_to_genera.items()
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
# DATA LOADING
# ============================================================
# loads datasets for hierarchical classification
def load_hierarchical_training_dataframe(
    input_path: Path = DEFAULT_IDENTIFICATION_DATASET,
) -> pd.DataFrame:
    """Load specimens eligible for hierarchical training.

    A specimen enters the dataset when it has:

    - an existing processed FLP image;
    - a valid family label;
    - ``eligible_family_training`` set to True.

    A valid genus is optional. Specimens without genus labels still
    contribute to family training.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Identification dataset not found: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    required_columns = {
        "numCol",
        "family",
        "genus",
        "view_code",
        "processed_image_path",
        "eligible_family_training",
        "eligible_genus_training",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Identification dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["family"] = dataframe[
        "family"
    ].apply(clean_taxonomic_value)

    dataframe["genus"] = dataframe[
        "genus"
    ].apply(clean_taxonomic_value)

    dataframe["eligible_family_training"] = dataframe[
        "eligible_family_training"
    ].apply(normalize_boolean)

    dataframe["eligible_genus_training"] = dataframe[
        "eligible_genus_training"
    ].apply(normalize_boolean)

    dataframe["view_code"] = (
        dataframe["view_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    training = dataframe[
        dataframe["eligible_family_training"]
        & dataframe["family"].notna()
        & (dataframe["view_code"] == "FLP")
    ].copy()

    if training.empty:
        raise ValueError(
            "No specimens were eligible for hierarchical training."
        )

    training["has_genus_label"] = (
        training["eligible_genus_training"]
        & training["genus"].notna()
    )

    missing_paths: list[str] = []

    for path_value in training["processed_image_path"]:
        image_path = resolve_image_path(path_value)

        if not image_path.exists() or not image_path.is_file():
            missing_paths.append(str(image_path))

    if missing_paths:
        preview = missing_paths[:10]

        raise FileNotFoundError(
            "Some hierarchical training images do not exist. "
            f"Examples: {preview}"
        )

    training = training.drop_duplicates(
        subset=["numCol"],
        keep="first",
    ).reset_index(drop=True)

    return training


# ============================================================
# PYTORCH DATASET
# ============================================================
# uses pytorch for hierarchical classification
class HierarchicalBrachyceraDataset(Dataset):
    """Dataset returning family and genus targets together."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        family_to_idx: dict[str, int],
        genus_to_idx: dict[str, int],
        transform: Callable[[Image.Image], Tensor] | None = None,
        return_metadata: bool = True,
    ) -> None:
        if dataframe.empty:
            raise ValueError(
                "Cannot create a hierarchical dataset "
                "from an empty dataframe."
            )

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.family_to_idx = family_to_idx
        self.genus_to_idx = genus_to_idx
        self.transform = transform
        self.return_metadata = return_metadata

        self._validate_taxonomy()

    def _validate_taxonomy(self) -> None:
        """Check that all labels exist in their mappings."""

        family_labels = set(
            self.dataframe["family"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unknown_families = (
            family_labels - set(self.family_to_idx)
        )

        if unknown_families:
            raise ValueError(
                "Families absent from family mapping: "
                f"{sorted(unknown_families)}"
            )

        genus_labels = set(
            self.dataframe.loc[
                self.dataframe["has_genus_label"],
                "genus",
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unknown_genera = (
            genus_labels - set(self.genus_to_idx)
        )

        if unknown_genera:
            raise ValueError(
                "Genera absent from genus mapping: "
                f"{sorted(unknown_genera)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = resolve_image_path(
            row["processed_image_path"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                raise ValueError(
                    "A transform is required to convert "
                    "PIL images into tensors."
                )

        family_label = str(row["family"]).strip()
        family_target = self.family_to_idx[
            family_label
        ]

        has_genus_label = bool(
            row["has_genus_label"]
        )

        if has_genus_label:
            genus_label = str(row["genus"]).strip()
            genus_target = self.genus_to_idx[
                genus_label
            ]
        else:
            genus_label = None
            genus_target = -1

        result = {
            "image": image_tensor,
            "family_target": family_target,
            "genus_target": genus_target,
            "has_genus_label": has_genus_label,
        }

        if self.return_metadata:
            result["metadata"] = {
                "numCol": str(row["numCol"]),
                "family": family_label,
                "genus": genus_label,
                "processed_image_path": str(
                    row["processed_image_path"]
                ),
            }

        return result


def create_hierarchical_dataset(
    transform: Callable[[Image.Image], Tensor],
    input_path: Path = DEFAULT_IDENTIFICATION_DATASET,
    save_mapping: bool = True,
    return_metadata: bool = True,
) -> tuple[
    HierarchicalBrachyceraDataset,
    dict[str, int],
    dict[str, int],
    dict[int, int],
]:
    """Create the complete hierarchical training dataset."""

    dataframe = load_hierarchical_training_dataframe(
        input_path
    )

    family_to_idx = build_class_mapping(
        dataframe["family"]
    )

    genus_dataframe = dataframe[
        dataframe["has_genus_label"]
    ].copy()

    genus_to_idx = build_class_mapping(
        genus_dataframe["genus"]
    )

    genus_to_family = build_genus_to_family_mapping(
        dataframe=genus_dataframe,
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
    )

    if save_mapping:
        save_hierarchical_mappings(
            family_to_idx=family_to_idx,
            genus_to_idx=genus_to_idx,
            genus_to_family=genus_to_family,
        )

    dataset = HierarchicalBrachyceraDataset(
        dataframe=dataframe,
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
        transform=transform,
        return_metadata=return_metadata,
    )

    return (
        dataset,
        family_to_idx,
        genus_to_idx,
        genus_to_family,
    )
