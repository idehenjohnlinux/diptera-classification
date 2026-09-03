"""Hierarchical identification of Brachycera specimens.


"""
# import modules 
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch import Tensor

from src.ml.hierarchical_model import create_hierarchical_model
from src.ml.transforms import get_validation_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "identification"
    / "identification_dataset.csv"
)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
    / "best_model.pt"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
    / "identification"
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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


# ============================================================
# GENERAL UTILITIES
# ============================================================

# creates the utilities to be used hierarchical identification
def select_device() -> torch.device:
    """Select CUDA, Apple MPS or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def clean_taxonomic_label(value: Any) -> str | None:
    """Clean a family or genus label."""

    if value is None:
        return None

    label = str(value).strip()

    if label.casefold() in UNKNOWN_VALUES:
        return None

    return label


def normalize_taxonomic_value(value: Any) -> str | None:
    """Return a clean taxonomic label or None.

    This function safely handles None, pandas NaN values, empty strings,
    and ordinary string values before any case-insensitive comparison.
    """

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


def labels_match(
    first: Any,
    second: Any,
) -> bool | None:
    """Compare two taxonomic labels safely.

    Returns None when the first value is missing because, in that case,
    there is no metadata label available for comparison.
    """

    first_label = normalize_taxonomic_value(first)
    second_label = normalize_taxonomic_value(second)

    if first_label is None:
        return None

    if second_label is None:
        return False

    return first_label.casefold() == second_label.casefold()


def parse_boolean(value: Any) -> bool:
    """Interpret common Boolean representations."""

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
        "valid",
        "eligible",
    }


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    """Find a dataframe column using several possible names."""

    normalized_columns = {
        str(column).strip().casefold(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        column = normalized_columns.get(
            candidate.strip().casefold()
        )

        if column is not None:
            return column

    if required:
        raise KeyError(
            "Required column not found.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def normalize_label_to_index(
    mapping: dict[Any, Any],
) -> dict[str, int]:
    """Convert a label-to-index mapping to standard types."""

    return {
        str(label): int(index)
        for label, index in mapping.items()
    }


def normalize_index_mapping(
    mapping: dict[Any, Any],
) -> dict[int, int]:
    """Convert an index mapping loaded from a checkpoint."""

    return {
        int(key): int(value)
        for key, value in mapping.items()
    }


def percentages(value: float) -> str:
    """Format a probability as a percentage."""

    return f"{value * 100:.2f}%"


# ============================================================
# METADATA STATUS
# ============================================================

# status for each identification 
def determine_metadata_status(
    family: str | None,
    genus: str | None,
) -> str:
    """Describe the specimen's existing identification status."""

    if family is not None and genus is not None:
        return "existing_identification"

    if family is not None and genus is None:
        return "partial_identification"

    if family is None and genus is not None:
        return "genus_only_identification"

    return "unidentified_specimen"


def determine_identification_result(
    metadata_family: str | None,
    metadata_genus: str | None,
    predicted_family: str,
    predicted_genus: str,
) -> str:
    """Compare the model prediction with existing metadata safely."""

    metadata_family = normalize_taxonomic_value(
        metadata_family
    )
    metadata_genus = normalize_taxonomic_value(
        metadata_genus
    )
    predicted_family = normalize_taxonomic_value(
        predicted_family
    )
    predicted_genus = normalize_taxonomic_value(
        predicted_genus
    )

    if predicted_family is None or predicted_genus is None:
        raise ValueError(
            "Predicted family and genus must contain valid labels."
        )

    family_matches = labels_match(
        metadata_family,
        predicted_family,
    )

    genus_matches = labels_match(
        metadata_genus,
        predicted_genus,
    )

    if metadata_family is None and metadata_genus is None:
        return "family_and_genus_suggested"

    if metadata_family is not None and metadata_genus is None:
        if family_matches is True:
            return "family_confirmed_genus_suggested"

        return "family_disagreement_genus_suggested"

    if metadata_family is None and metadata_genus is not None:
        if genus_matches is True:
            return "genus_confirmed_family_suggested"

        return "genus_disagreement_family_suggested"

    if family_matches is True and genus_matches is True:
        return "existing_identification_confirmed"

    if family_matches is True and genus_matches is False:
        return "family_confirmed_genus_disagreement"

    if family_matches is False and genus_matches is True:
        return "genus_confirmed_family_disagreement"

    return "family_and_genus_disagreement"


# ============================================================
# REPORTING
# ============================================================


@dataclass(frozen=True)
class ReliabilityAssessment:
    """Human-readable reliability assessment."""

    category: str
    stars: str


class IdentificationReport:
    """Create curator-friendly interpretations and reports."""

    @staticmethod
    def assess_reliability(
        confidence: float,
    ) -> ReliabilityAssessment:
        """Convert a probability into a reliability category."""

        if confidence >= 0.95:
            return ReliabilityAssessment(
                "Exceptional",
                "★★★★★",
            )

        if confidence >= 0.85:
            return ReliabilityAssessment(
                "Very high",
                "★★★★★",
            )

        if confidence >= 0.75:
            return ReliabilityAssessment(
                "High",
                "★★★★☆",
            )

        if confidence >= 0.60:
            return ReliabilityAssessment(
                "Moderate",
                "★★★☆☆",
            )

        return ReliabilityAssessment(
            "Low",
            "★★☆☆☆",
        )

    @staticmethod
    def build_decision(record: dict[str, Any]) -> str:
        """Create a curator-friendly decision message."""

        result = record["identification_result"]

        messages = {
            "existing_identification_confirmed": (
                "The hierarchical model supports the existing family "
                "and genus identification."
            ),
            "family_confirmed_genus_suggested": (
                "The existing family is supported and the model proposes "
                "a compatible genus for the missing genus label."
            ),
            "family_disagreement_genus_suggested": (
                "The model proposes a different family and a compatible "
                "genus for this partially identified specimen."
            ),
            "genus_confirmed_family_suggested": (
                "The existing genus is supported and the model proposes "
                "its associated family."
            ),
            "genus_disagreement_family_suggested": (
                "The predicted genus differs from metadata. The model also "
                "proposes the family associated with its genus prediction."
            ),
            "family_confirmed_genus_disagreement": (
                "The existing family is supported, but the predicted genus "
                "differs from the current metadata."
            ),
            "genus_confirmed_family_disagreement": (
                "The existing genus is supported, but the family prediction "
                "differs from the current metadata."
            ),
            "family_and_genus_disagreement": (
                "Both family and genus predictions differ from the current "
                "collection metadata."
            ),
            "family_and_genus_suggested": (
                "The model proposes family and genus labels for a specimen "
                "without taxonomic metadata."
            ),
        }

        return messages.get(
            result,
            "The model generated a hierarchical family-genus suggestion.",
        )

    @staticmethod
    def build_recommendation(
        record: dict[str, Any],
    ) -> str:
        """Generate an expert recommendation."""

        family_confidence = float(
            record["family_confidence"]
        )

        genus_confidence = float(
            record["genus_confidence"]
        )

        family_match = record[
            "family_matches_metadata"
        ]

        genus_match = record[
            "genus_matches_metadata"
        ]

        if family_match is False:
            return (
                "Prioritise manual review of diagnostic family characters. "
                "Do not replace the collection identification using this "
                "prediction alone."
            )

        if (
            family_match is True
            and genus_match is False
        ):
            return (
                "Retain the family assignment and compare morphological "
                "characters that distinguish the metadata genus from the "
                "top compatible genus candidates."
            )

        if family_confidence < 0.60:
            return (
                "Treat the family result as preliminary. Obtain another "
                "standardised image or perform manual family-level review."
            )

        if genus_confidence < 0.60:
            return (
                "Use the family prediction for triage and manually review "
                "the top compatible genera before assigning a genus."
            )

        if (
            family_match is True
            and genus_match is True
        ):
            return (
                "Retain the current identification. The hierarchical model "
                "provides supporting evidence, but morphology remains the "
                "authoritative basis for curation."
            )

        return (
            "Use the prediction to prioritise expert verification. Confirm "
            "diagnostic morphological characters before updating metadata."
        )

    @classmethod
    def enrich(
        cls,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Add reliability and interpretation fields."""

        family_assessment = cls.assess_reliability(
            float(record["family_confidence"])
        )

        genus_assessment = cls.assess_reliability(
            float(record["genus_confidence"])
        )

        record["family_reliability"] = (
            family_assessment.category
        )

        record["family_stars"] = (
            family_assessment.stars
        )

        record["genus_reliability"] = (
            genus_assessment.category
        )

        record["genus_stars"] = (
            genus_assessment.stars
        )

        record["decision_message"] = (
            cls.build_decision(record)
        )

        record["expert_recommendation"] = (
            cls.build_recommendation(record)
        )

        return record

    @staticmethod
    def format_text_report(
        record: dict[str, Any],
    ) -> str:
        """Create a detailed human-readable report."""

        metadata_family = (
            record["metadata_family"]
            or "Unidentified"
        )

        metadata_genus = (
            record["metadata_genus"]
            or "Unidentified"
        )

        allowed_genera = "\n".join(
            f"  • {genus}"
            for genus in record["allowed_genera"]
        )

        top_candidates = "\n".join(
            (
                f"  {rank}. {candidate['genus']} "
                f"({percentages(candidate['confidence'])})"
            )
            for rank, candidate in enumerate(
                record["top_genus_candidates"],
                start=1,
            )
        )

        return "\n".join(
            [
                "=" * 78,
                "HIERARCHICAL BRACHYCERA IDENTIFICATION REPORT",
                "=" * 78,
                "",
                "Specimen",
                "-" * 78,
                f"Specimen ID: {record['specimen_id']}",
                f"Image: {record['image_path']}",
                f"View: {record['view_code']}",
                "",
                "Current collection metadata",
                "-" * 78,
                f"Status: {record['metadata_status']}",
                f"Family: {metadata_family}",
                f"Genus: {metadata_genus}",
                "",
                "Hierarchical EfficientNet-B0 prediction",
                "-" * 78,
                (
                    "Predicted family: "
                    f"{record['predicted_family']}"
                ),
                (
                    "Family confidence: "
                    f"{percentages(record['family_confidence'])}"
                ),
                (
                    "Family reliability: "
                    f"{record['family_reliability']} "
                    f"{record['family_stars']}"
                ),
                "",
                (
                    "Suggested genus: "
                    f"{record['predicted_genus']}"
                ),
                (
                    "Genus confidence: "
                    f"{percentages(record['genus_confidence'])}"
                ),
                (
                    "Genus reliability: "
                    f"{record['genus_reliability']} "
                    f"{record['genus_stars']}"
                ),
                "",
                "Genera compatible with the predicted family",
                "-" * 78,
                allowed_genera,
                "",
                "Top compatible genus candidates",
                "-" * 78,
                top_candidates,
                "",
                "Curator-friendly decision",
                "-" * 78,
                record["decision_message"],
                "",
                "Expert recommendation",
                "-" * 78,
                record["expert_recommendation"],
                "",
                "Taxonomic interpretation",
                "-" * 78,
                (
                    "The genus result was constrained to genera mapped to "
                    "the predicted family. This report supports expert "
                    "curation and does not replace morphological or "
                    "molecular confirmation."
                ),
            ]
        )


# ============================================================
# HIERARCHICAL IDENTIFIER
# ============================================================

# creates class for hierarchical identifier 
class HierarchicalIdentifier:
    """Hierarchical EfficientNet-B0 identification engine."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
        top_k: int = 5,
        device: torch.device | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_directory = Path(output_directory)
        self.top_k = top_k
        self.device = device or select_device()

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.checkpoint = self._load_checkpoint()

        self.family_to_idx = self._load_family_mapping()
        self.genus_to_idx = self._load_genus_mapping()

        self.idx_to_family = {
            index: family
            for family, index in self.family_to_idx.items()
        }

        self.idx_to_genus = {
            index: genus
            for genus, index in self.genus_to_idx.items()
        }

        self.genus_to_family_idx = (
            self._load_genus_to_family_mapping()
        )

        self.taxonomy_mask = self._build_taxonomy_mask()

        self.configuration = self.checkpoint.get(
            "configuration",
            {},
        )

        self.image_size = int(
            self.configuration.get("image_size", 224)
        )

        self.view_code = str(
            self.checkpoint.get(
                "view_code",
                self.configuration.get(
                    "view_code",
                    "FLP",
                ),
            )
        ).upper()

        self.model = self._load_model()

        self.transform = get_validation_transforms(
            image_size=(
                self.image_size,
                self.image_size,
            )
        )

    # --------------------------------------------------------
    # CHECKPOINT AND MODEL
    # --------------------------------------------------------

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load the trained hierarchical checkpoint."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "The checkpoint must contain a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict'.\n"
                f"Available keys: {list(checkpoint.keys())}"
            )

        return checkpoint

    def _load_family_mapping(self) -> dict[str, int]:
        """Read family class mappings from the checkpoint."""

        for key in (
            "family_to_idx",
            "family_class_to_idx",
            "family_mapping",
        ):
            mapping = self.checkpoint.get(key)

            if mapping:
                return normalize_label_to_index(mapping)

        raise KeyError(
            "Family mapping was not found in the checkpoint.\n"
            f"Available keys: {list(self.checkpoint.keys())}"
        )

    def _load_genus_mapping(self) -> dict[str, int]:
        """Read genus class mappings from the checkpoint."""

        for key in (
            "genus_to_idx",
            "genus_class_to_idx",
            "genus_mapping",
        ):
            mapping = self.checkpoint.get(key)

            if mapping:
                return normalize_label_to_index(mapping)

        raise KeyError(
            "Genus mapping was not found in the checkpoint.\n"
            f"Available keys: {list(self.checkpoint.keys())}"
        )

    def _load_genus_to_family_mapping(
        self,
    ) -> dict[int, int]:
        """Read genus-index to family-index relationships."""

        for key in (
            "genus_to_family_idx",
            "genus_to_family",
            "genus_family_mapping",
        ):
            mapping = self.checkpoint.get(key)

            if mapping:
                return normalize_index_mapping(mapping)

        taxonomy_mapping = self.checkpoint.get(
            "taxonomy_mappings"
        )

        if isinstance(taxonomy_mapping, dict):
            mapping = taxonomy_mapping.get(
                "genus_to_family_idx"
            )

            if mapping:
                return normalize_index_mapping(mapping)

        raise KeyError(
            "Genus-to-family mapping was not found in "
            "the checkpoint.\n"
            f"Available keys: {list(self.checkpoint.keys())}"
        )

    def _build_taxonomy_mask(self) -> Tensor:
        """Build a family × genus Boolean taxonomy matrix."""

        number_of_families = len(self.family_to_idx)
        number_of_genera = len(self.genus_to_idx)

        mask = torch.zeros(
            (
                number_of_families,
                number_of_genera,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        for (
            genus_index,
            family_index,
        ) in self.genus_to_family_idx.items():
            if genus_index not in self.idx_to_genus:
                raise ValueError(
                    "Unknown genus index in taxonomy mapping: "
                    f"{genus_index}"
                )

            if family_index not in self.idx_to_family:
                raise ValueError(
                    "Unknown family index in taxonomy mapping: "
                    f"{family_index}"
                )

            mask[family_index, genus_index] = True

        genera_without_family = torch.where(
            mask.sum(dim=0) == 0
        )[0].cpu().tolist()

        if genera_without_family:
            names = [
                self.idx_to_genus[index]
                for index in genera_without_family
            ]

            raise ValueError(
                "The following genera are not assigned to "
                f"a family: {names}"
            )

        return mask

    def _load_model(self) -> torch.nn.Module:
        """Rebuild the hierarchical EfficientNet-B0 model."""

        model = create_hierarchical_model(
            number_of_families=len(
                self.family_to_idx
            ),
            number_of_genera=len(
                self.genus_to_idx
            ),
            pretrained=False,
            family_dropout=float(
                self.configuration.get(
                    "family_dropout",
                    0.2,
                )
            ),
            genus_dropout=float(
                self.configuration.get(
                    "genus_dropout",
                    0.3,
                )
            ),
            genus_hidden_dimension=int(
                self.configuration.get(
                    "genus_hidden_dimension",
                    512,
                )
            ),
        )

        model.load_state_dict(
            self.checkpoint["model_state_dict"],
            strict=True,
        )

        model.to(self.device)
        model.eval()

        return model

    # --------------------------------------------------------
    # TAXONOMY
    # --------------------------------------------------------

    def allowed_genus_indices(
        self,
        family_index: int,
    ) -> list[int]:
        """Return genus indices belonging to one family."""

        if family_index not in self.idx_to_family:
            raise KeyError(
                f"Unknown family index: {family_index}"
            )

        indices = torch.where(
            self.taxonomy_mask[family_index]
        )[0]

        return [
            int(index)
            for index in indices.cpu().tolist()
        ]

    def allowed_genera(
        self,
        family_index: int,
    ) -> list[str]:
        """Return genus names belonging to one family."""

        return [
            self.idx_to_genus[index]
            for index in self.allowed_genus_indices(
                family_index
            )
        ]

    def family_for_genus(
        self,
        genus_index: int,
    ) -> str:
        """Return the family associated with a genus."""

        family_index = self.genus_to_family_idx[
            genus_index
        ]

        return self.idx_to_family[family_index]

    # --------------------------------------------------------
    # DATASET PREPARATION
    # --------------------------------------------------------

    def load_identification_dataset(
        self,
    ) -> pd.DataFrame:
        """Load all eligible specimens for the checkpoint view."""

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                "Identification dataset not found: "
                f"{self.dataset_path}"
            )

        dataframe = pd.read_csv(
            self.dataset_path,
            dtype=str,
            keep_default_na=False,
        )

        specimen_column = find_column(
            dataframe,
            [
                "specimen_id",
                "numCol",
                "numcol",
                "specimen",
            ],
        )

        family_column = find_column(
            dataframe,
            [
                "family",
                "familia",
            ],
            required=False,
        )

        genus_column = find_column(
            dataframe,
            [
                "genus",
                "genero",
                "género",
            ],
            required=False,
        )

        view_column = find_column(
            dataframe,
            [
                "view_code",
                "view",
                "image_view",
            ],
            required=False,
        )

        image_path_column = find_column(
            dataframe,
            [
                "processed_image_path",
                "processed_path",
                "image_path",
                "path",
                "filepath",
                "file_path",
                "filename",
            ],
            required=False,
        )

        eligible_column = find_column(
            dataframe,
            [
                "eligible",
                "is_eligible",
                "eligible_for_identification",
                "identification_eligible",
            ],
            required=False,
        )

        prepared = pd.DataFrame()

        prepared["specimen_id"] = (
            dataframe[specimen_column]
            .astype(str)
            .str.strip()
        )

        if family_column is None:
            prepared["metadata_family"] = None
        else:
            prepared["metadata_family"] = dataframe[
                family_column
            ].apply(normalize_taxonomic_value)

        if genus_column is None:
            prepared["metadata_genus"] = None
        else:
            prepared["metadata_genus"] = dataframe[
                genus_column
            ].apply(normalize_taxonomic_value)

        if view_column is None:
            prepared["view_code"] = self.view_code
        else:
            prepared["view_code"] = (
                dataframe[view_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            prepared.loc[
                prepared["view_code"].eq(""),
                "view_code",
            ] = self.view_code

        if image_path_column is None:
            prepared["raw_image_path"] = None
        else:
            prepared["raw_image_path"] = dataframe[
                image_path_column
            ]

        if eligible_column is None:
            prepared["eligible"] = True
        else:
            prepared["eligible"] = dataframe[
                eligible_column
            ].apply(parse_boolean)

        prepared = prepared[
            prepared["eligible"]
        ].copy()

        prepared = prepared[
            prepared["view_code"].eq(
                self.view_code
            )
        ].copy()

        prepared = prepared.drop_duplicates(
            subset=["specimen_id"],
            keep="first",
        )

        prepared = prepared.reset_index(drop=True)

        if prepared.empty:
            raise ValueError(
                "No eligible specimens were found for "
                f"view {self.view_code}."
            )

        return prepared

    def resolve_image_path(
        self,
        raw_path: Any,
        specimen_id: str,
        view_code: str,
    ) -> Path:
        """Locate the specimen image using several project layouts."""

        candidates: list[Path] = []

        if raw_path is not None:
            raw_string = str(raw_path).strip()

            if raw_string.casefold() not in UNKNOWN_VALUES:
                path = Path(raw_string).expanduser()

                if path.is_absolute():
                    candidates.append(path)
                else:
                    candidates.extend(
                        [
                            PROJECT_ROOT / path,
                            self.dataset_path.parent / path,
                        ]
                    )

        root_candidates = [
            PROJECT_ROOT / "data" / "processed",
            PROJECT_ROOT / "processed",
            PROJECT_ROOT / "images_processed",
            PROJECT_ROOT / "images",
            PROJECT_ROOT / "data" / "images",
        ]

        for root in root_candidates:
            for extension in sorted(
                IMAGE_EXTENSIONS
            ):
                # Layout: root/specimen/FLP.jpg
                candidates.append(
                    root
                    / specimen_id
                    / f"{view_code}{extension}"
                )

                # Layout: root/FLP/specimen.jpg
                candidates.append(
                    root
                    / view_code
                    / f"{specimen_id}{extension}"
                )

                # Layout: root/specimen_FLP.jpg
                candidates.append(
                    root
                    / f"{specimen_id}_{view_code}{extension}"
                )

                # Layout: root/FLP_specimen.jpg
                candidates.append(
                    root
                    / f"{view_code}_{specimen_id}{extension}"
                )

        seen: set[str] = set()

        for candidate in candidates:
            candidate_key = str(candidate)

            if candidate_key in seen:
                continue

            seen.add(candidate_key)

            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        # Final recursive search, used only after direct layouts fail.
        search_patterns = [
            f"**/{specimen_id}/{view_code}.*",
            f"**/{specimen_id}_{view_code}.*",
            f"**/{view_code}_{specimen_id}.*",
            f"**/{view_code}/{specimen_id}.*",
        ]

        for root in root_candidates:
            if not root.exists():
                continue

            for pattern in search_patterns:
                for candidate in root.glob(pattern):
                    if (
                        candidate.is_file()
                        and candidate.suffix.casefold()
                        in IMAGE_EXTENSIONS
                    ):
                        return candidate.resolve()

        raise FileNotFoundError(
            "Processed image not found for "
            f"{specimen_id}, view {view_code}. "
            "Checked explicit CSV paths and common processed-image layouts."
        )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    def predict_image(
        self,
        image_path: str | Path,
    ) -> dict[str, Any]:
        """Predict family and a compatible genus."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        with Image.open(image_path) as image:
            image_tensor = self.transform(
                image.convert("RGB")
            )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            outputs = self.model(image_tensor)

        family_logits = outputs["family_logits"]
        genus_logits = outputs["genus_logits"]

        family_probabilities = torch.softmax(
            family_logits,
            dim=1,
        )

        raw_genus_probabilities = torch.softmax(
            genus_logits,
            dim=1,
        )

        predicted_family_index = int(
            family_probabilities.argmax(
                dim=1
            ).item()
        )

        predicted_family = self.idx_to_family[
            predicted_family_index
        ]

        family_confidence = float(
            family_probabilities[
                0,
                predicted_family_index,
            ].item()
        )

        allowed_mask = self.taxonomy_mask[
            predicted_family_index
        ]

        allowed_indices = self.allowed_genus_indices(
            predicted_family_index
        )

        if not allowed_indices:
            raise RuntimeError(
                "The predicted family has no compatible genera "
                "in the checkpoint taxonomy mapping."
            )

        constrained_logits = genus_logits[
            0,
            allowed_indices,
        ]

        constrained_subset = torch.softmax(
            constrained_logits,
            dim=0,
        )

        constrained_probabilities = torch.zeros_like(
            raw_genus_probabilities[0]
        )

        constrained_probabilities[
            allowed_indices
        ] = constrained_subset

        predicted_genus_index = int(
            constrained_probabilities.argmax().item()
        )

        predicted_genus = self.idx_to_genus[
            predicted_genus_index
        ]

        genus_confidence = float(
            constrained_probabilities[
                predicted_genus_index
            ].item()
        )

        raw_genus_index = int(
            raw_genus_probabilities.argmax(
                dim=1
            ).item()
        )

        raw_genus = self.idx_to_genus[
            raw_genus_index
        ]

        raw_genus_confidence = float(
            raw_genus_probabilities[
                0,
                raw_genus_index,
            ].item()
        )

        raw_genus_family = self.family_for_genus(
            raw_genus_index
        )

        raw_prediction_is_consistent = (
            raw_genus_family == predicted_family
        )

        effective_top_k = min(
            self.top_k,
            len(allowed_indices),
        )

        top_probabilities, top_indices = torch.topk(
            constrained_probabilities,
            k=effective_top_k,
        )

        top_candidates = []

        for probability, genus_index in zip(
            top_probabilities.cpu().tolist(),
            top_indices.cpu().tolist(),
        ):
            top_candidates.append(
                {
                    "genus": self.idx_to_genus[
                        int(genus_index)
                    ],
                    "confidence": float(
                        probability
                    ),
                }
            )

        return {
            "predicted_family": predicted_family,
            "predicted_family_index": (
                predicted_family_index
            ),
            "family_confidence": family_confidence,
            "predicted_genus": predicted_genus,
            "predicted_genus_index": (
                predicted_genus_index
            ),
            "genus_confidence": genus_confidence,
            "allowed_genera": self.allowed_genera(
                predicted_family_index
            ),
            "top_genus_candidates": top_candidates,
            "raw_genus": raw_genus,
            "raw_genus_confidence": (
                raw_genus_confidence
            ),
            "raw_genus_family": raw_genus_family,
            "raw_prediction_is_consistent": (
                raw_prediction_is_consistent
            ),
        }

    # --------------------------------------------------------
    # IDENTIFICATION RECORD
    # --------------------------------------------------------

    def build_record(
        self,
        specimen_id: str,
        image_path: Path,
        metadata_family: str | None,
        metadata_genus: str | None,
        prediction: dict[str, Any],
    ) -> dict[str, Any]:
        """Build and enrich the complete specimen result."""

        predicted_family = prediction[
            "predicted_family"
        ]

        predicted_genus = prediction[
            "predicted_genus"
        ]

        metadata_status = determine_metadata_status(
            family=metadata_family,
            genus=metadata_genus,
        )

        identification_result = (
            determine_identification_result(
                metadata_family=metadata_family,
                metadata_genus=metadata_genus,
                predicted_family=predicted_family,
                predicted_genus=predicted_genus,
            )
        )

        metadata_family = normalize_taxonomic_value(
            metadata_family
        )
        metadata_genus = normalize_taxonomic_value(
            metadata_genus
        )

        family_matches_metadata = labels_match(
            metadata_family,
            predicted_family,
        )

        genus_matches_metadata = labels_match(
            metadata_genus,
            predicted_genus,
        )

        requires_expert_verification = (
            metadata_family is None
            or metadata_genus is None
            or family_matches_metadata is False
            or genus_matches_metadata is False
            or prediction["family_confidence"] < 0.60
            or prediction["genus_confidence"] < 0.60
        )

        record = {
            "specimen_id": specimen_id,
            "image_path": str(image_path),
            "view_code": self.view_code,
            "metadata_status": metadata_status,
            "metadata_family": metadata_family,
            "metadata_genus": metadata_genus,
            "predicted_family": predicted_family,
            "family_confidence": prediction[
                "family_confidence"
            ],
            "allowed_genera": prediction[
                "allowed_genera"
            ],
            "predicted_genus": predicted_genus,
            "genus_confidence": prediction[
                "genus_confidence"
            ],
            "raw_genus": prediction[
                "raw_genus"
            ],
            "raw_genus_confidence": prediction[
                "raw_genus_confidence"
            ],
            "raw_genus_family": prediction[
                "raw_genus_family"
            ],
            "raw_prediction_is_consistent": (
                prediction[
                    "raw_prediction_is_consistent"
                ]
            ),
            "top_genus_candidates": prediction[
                "top_genus_candidates"
            ],
            "family_matches_metadata": (
                family_matches_metadata
            ),
            "genus_matches_metadata": (
                genus_matches_metadata
            ),
            "identification_result": (
                identification_result
            ),
            "requires_expert_verification": (
                requires_expert_verification
            ),
            "identified_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        return IdentificationReport.enrich(record)

    # --------------------------------------------------------
    # COMPLETE DATASET IDENTIFICATION
    # --------------------------------------------------------

    def identify_all(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, str]],
    ]:
        """Identify every eligible specimen."""

        dataframe = self.load_identification_dataset()

        records: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        print("\nHierarchical Brachycera identification")
        print("=" * 72)
        print(f"Device: {self.device}")
        print(f"View: {self.view_code}")
        print("Backbone: EfficientNet-B0")
        print(f"Dataset: {self.dataset_path}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Eligible specimens: {len(dataframe)}")
        print("=" * 72)

        for index, row in dataframe.iterrows():
            specimen_id = row["specimen_id"]

            try:
                image_path = self.resolve_image_path(
                    raw_path=row["raw_image_path"],
                    specimen_id=specimen_id,
                    view_code=row["view_code"],
                )

                prediction = self.predict_image(
                    image_path=image_path
                )

                record = self.build_record(
                    specimen_id=specimen_id,
                    image_path=image_path,
                    metadata_family=row[
                        "metadata_family"
                    ],
                    metadata_genus=row[
                        "metadata_genus"
                    ],
                    prediction=prediction,
                )

                records.append(record)

            except Exception as error:
                errors.append(
                    {
                        "specimen_id": specimen_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

            processed = index + 1

            if (
                processed % 25 == 0
                or processed == len(dataframe)
            ):
                print(
                    f"Processed {processed}/{len(dataframe)} "
                    f"| identified: {len(records)} "
                    f"| errors: {len(errors)}"
                )

        self.save_reports(
            records=records,
            errors=errors,
        )

        self.print_summary(
            records=records,
            errors=errors,
        )

        return records, errors

    # --------------------------------------------------------
    # SAVE REPORTING
    # --------------------------------------------------------

    def save_reports(
        self,
        records: list[dict[str, Any]],
        errors: list[dict[str, str]],
    ) -> None:
        """Save CSV, JSON, TXT and operational summary outputs."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            self.output_directory
            / "specimen_identifications.json"
        )

        csv_path = (
            self.output_directory
            / "specimen_identifications.csv"
        )

        text_path = (
            self.output_directory
            / "specimen_identifications.txt"
        )

        errors_path = (
            self.output_directory
            / "identification_errors.csv"
        )

        summary_path = (
            self.output_directory
            / "identification_summary.json"
        )

        with json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                records,
                file,
                indent=4,
                ensure_ascii=False,
            )

        csv_records = []

        for record in records:
            csv_record = record.copy()

            csv_record["allowed_genera"] = "; ".join(
                record["allowed_genera"]
            )

            csv_record["top_genus_candidates"] = (
                "; ".join(
                    (
                        f"{item['genus']}:"
                        f"{item['confidence']:.6f}"
                    )
                    for item in record[
                        "top_genus_candidates"
                    ]
                )
            )

            csv_records.append(csv_record)

        pd.DataFrame(csv_records).to_csv(
            csv_path,
            index=False,
        )

        pd.DataFrame(errors).to_csv(
            errors_path,
            index=False,
        )

        with text_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for index, record in enumerate(records):
                if index > 0:
                    file.write("\n\n")

                file.write(
                    IdentificationReport.format_text_report(
                        record
                    )
                )

        summary = self.build_operational_summary(
            records=records,
            errors=errors,
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print("\nReports saved")
        print("-" * 72)
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
        print(f"Text: {text_path}")
        print(f"Errors: {errors_path}")
        print(f"Summary: {summary_path}")

    @staticmethod
    def build_operational_summary(
        records: list[dict[str, Any]],
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Build a non-misleading operational summary."""

        metadata_status_counts = Counter(
            record["metadata_status"]
            for record in records
        )

        identification_result_counts = Counter(
            record["identification_result"]
            for record in records
        )

        predicted_family_counts = Counter(
            record["predicted_family"]
            for record in records
        )

        predicted_genus_counts = Counter(
            record["predicted_genus"]
            for record in records
        )

        expert_review_count = sum(
            record["requires_expert_verification"]
            is True
            for record in records
        )

        confirmed_count = sum(
            record["identification_result"]
            == "existing_identification_confirmed"
            for record in records
        )

        partial_completed = sum(
            record["identification_result"]
            in {
                "family_confirmed_genus_suggested",
                "genus_confirmed_family_suggested",
            }
            for record in records
        )

        new_family_suggestions = sum(
            record["metadata_family"] is None
            for record in records
        )

        new_genus_suggestions = sum(
            record["metadata_genus"] is None
            for record in records
        )

        return {
            "generated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "successful_identifications": len(records),
            "errors": len(errors),
            "existing_identifications_confirmed": (
                confirmed_count
            ),
            "partial_identifications_completed": (
                partial_completed
            ),
            "previously_missing_family_labels_receiving_suggestion": (
                new_family_suggestions
            ),
            "previously_missing_genus_labels_receiving_suggestion": (
                new_genus_suggestions
            ),
            "specimens_requiring_expert_review": (
                expert_review_count
            ),
            "families_represented_in_predictions": len(
                predicted_family_counts
            ),
            "genera_represented_in_predictions": len(
                predicted_genus_counts
            ),
            "metadata_status_counts": dict(
                metadata_status_counts
            ),
            "identification_result_counts": dict(
                identification_result_counts
            ),
            "predicted_family_counts": dict(
                predicted_family_counts
            ),
            "predicted_genus_counts": dict(
                predicted_genus_counts
            ),
            "interpretation": (
                "This is an operational identification summary. "
                "It does not report classifier accuracy. Scientific "
                "performance must be obtained from the dedicated "
                "hierarchical evaluation workflow."
            ),
        }

    @staticmethod
    def print_summary(
        records: list[dict[str, Any]],
        errors: list[dict[str, str]],
    ) -> None:
        """Print final operational identification statistics."""

        summary = (
            HierarchicalIdentifier
            .build_operational_summary(
                records=records,
                errors=errors,
            )
        )

        print("\nIdentification summary")
        print("=" * 72)
        print(
            "Successful identifications: "
            f"{summary['successful_identifications']}"
        )
        print(f"Errors: {summary['errors']}")
        print(
            "Existing identifications confirmed: "
            f"{summary['existing_identifications_confirmed']}"
        )
        print(
            "Partial identifications completed: "
            f"{summary['partial_identifications_completed']}"
        )
        print(
            "Missing family labels receiving a suggestion: "
            f"{summary['previously_missing_family_labels_receiving_suggestion']}"
        )
        print(
            "Missing genus labels receiving a suggestion: "
            f"{summary['previously_missing_genus_labels_receiving_suggestion']}"
        )
        print(
            "Specimens requiring expert review: "
            f"{summary['specimens_requiring_expert_review']}"
        )
        print(
            "Families represented in predictions: "
            f"{summary['families_represented_in_predictions']}"
        )
        print(
            "Genera represented in predictions: "
            f"{summary['genera_represented_in_predictions']}"
        )
        print(
            "\nScientific classifier accuracy is not calculated "
            "by this identification script."
        )


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

# creates command line entry point 
def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict Brachycera family and a compatible "
            "genus for eligible processed specimens."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to identification_dataset.csv.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the trained best_model.pt.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory where reports will be saved.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of compatible genus candidates "
            "to retain."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete identification workflow."""

    arguments = parse_arguments()

    identifier = HierarchicalIdentifier(
        dataset_path=arguments.dataset,
        checkpoint_path=arguments.checkpoint,
        output_directory=(
            arguments.output_directory
        ),
        top_k=arguments.top_k,
    )

    identifier.identify_all()


if __name__ == "__main__":
    main()
