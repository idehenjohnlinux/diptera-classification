"""Evaluate trained Brachycera CNN checkpoints.

The evaluator reconstructs the validation fold associated with a trained
checkpoint and calculates classification performance at two levels:

1. Image level:
   Each photograph is treated as one prediction.

2. Specimen level:
   Probabilities from all photographs belonging to the same specimen
   are averaged before assigning the final predicted class.

Generated files
---------------
- evaluation_summary.json
- per_image_predictions.csv
- per_specimen_predictions.csv
- classification_report_image.csv
- classification_report_specimen.csv
- confusion_matrix_image.csv
- confusion_matrix_image_normalized.csv
- confusion_matrix_image.png
- confusion_matrix_image_normalized.png
- confusion_matrix_specimen.csv
- confusion_matrix_specimen_normalized.csv
- confusion_matrix_specimen.png
- confusion_matrix_specimen_normalized.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.dataset import create_datasets
from src.ml.models import create_model
from src.ml.transforms import get_validation_transforms


# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_WORKERS: Final[int] = 2

OPTIONAL_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "family",
    "genus",
    "specific_epithet",
    "specificEpithet",
    "scientific_name",
    "scientificName",
)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def select_device() -> torch.device:
    """Select CUDA when available; otherwise use the CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a dictionary as formatted JSON."""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def safe_column_name(value: str) -> str:
    """Convert a class name into a safe CSV column suffix."""

    cleaned = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value).strip(),
    )

    return cleaned.strip("_")


def find_scientific_name_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find the scientific-name column when one is available."""

    possible_columns = (
        "scientific_name",
        "scientificName",
    )

    for column in possible_columns:
        if column in dataframe.columns:
            return column

    return None


def normalize_metadata_series(
    series: pd.Series,
) -> pd.Series:
    """Convert a metadata column to clean text."""

    return (
        series
        .fillna("")
        .astype(str)
        .replace(
            {
                "nan": "",
                "None": "",
                "<NA>": "",
            }
        )
        .str.strip()
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load and validate a trained checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must contain a dictionary."
        )

    required_keys = {
        "architecture",
        "taxonomic_level",
        "view_code",
        "validation_fold",
        "image_size",
        "class_to_idx",
        "model_state_dict",
    }

    missing_keys = required_keys - set(checkpoint)

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return checkpoint


def normalize_class_mapping(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    """Return normalized class-to-index and index-to-class mappings."""

    class_to_idx = {
        str(class_name): int(index)
        for class_name, index
        in checkpoint["class_to_idx"].items()
    }

    if "idx_to_class" in checkpoint:
        idx_to_class = {
            int(index): str(class_name)
            for index, class_name
            in checkpoint["idx_to_class"].items()
        }
    else:
        idx_to_class = {
            index: class_name
            for class_name, index in class_to_idx.items()
        }

    return class_to_idx, idx_to_class


# ============================================================
# DATASET AND MODEL RECONSTRUCTION
# ============================================================

def create_validation_loader(
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, Any, dict[str, int]]:
    """Reconstruct the validation dataset used by the experiment."""

    image_size = int(checkpoint["image_size"])

    validation_transform = get_validation_transforms(
        image_size=(image_size, image_size)
    )

    (
        _,
        validation_dataset,
        generated_class_to_idx,
    ) = create_datasets(
        level=str(checkpoint["taxonomic_level"]),
        view_code=str(checkpoint["view_code"]),
        validation_fold=int(
            checkpoint["validation_fold"]
        ),
        train_transform=None,
        validation_transform=validation_transform,
        return_metadata=False,
        save_mapping=False,
    )

    checkpoint_class_to_idx = {
        str(class_name): int(index)
        for class_name, index
        in checkpoint["class_to_idx"].items()
    }

    if generated_class_to_idx != checkpoint_class_to_idx:
        raise RuntimeError(
            "The class mapping reconstructed from the dataset does "
            "not match the class mapping stored in the checkpoint.\n"
            f"Checkpoint mapping: {checkpoint_class_to_idx}\n"
            f"Generated mapping: {generated_class_to_idx}"
        )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    return (
        validation_loader,
        validation_dataset,
        checkpoint_class_to_idx,
    )


def reconstruct_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Recreate the CNN architecture and load its trained weights."""

    number_of_classes = len(
        checkpoint["class_to_idx"]
    )

    model = create_model(
        architecture=str(checkpoint["architecture"]),
        num_classes=number_of_classes,
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model: nn.Module,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return true labels, predicted labels and class probabilities."""

    all_true_labels: list[int] = []
    all_predicted_labels: list[int] = []
    all_probabilities: list[np.ndarray] = []

    model.eval()

    with torch.inference_mode():
        for images, true_labels in tqdm(
            validation_loader,
            desc="Evaluating",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            logits: Tensor = model(images)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_labels = probabilities.argmax(
                dim=1
            )

            all_true_labels.extend(
                true_labels.cpu().numpy().tolist()
            )

            all_predicted_labels.extend(
                predicted_labels.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.asarray(all_true_labels, dtype=int),
        np.asarray(all_predicted_labels, dtype=int),
        np.asarray(all_probabilities, dtype=float),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Calculate overall classification metrics."""

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels,
    )

    (
        precision_macro,
        recall_macro,
        f1_macro,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    (
        precision_weighted,
        recall_weighted,
        f1_weighted,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision_macro": float(
            precision_macro
        ),
        "recall_macro": float(
            recall_macro
        ),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(
            precision_weighted
        ),
        "recall_weighted": float(
            recall_weighted
        ),
        "f1_weighted": float(
            f1_weighted
        ),
    }


def create_classification_report_dataframe(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: list[str],
    label_indices: list[int],
) -> pd.DataFrame:
    """Create a class-by-class precision, recall and F1 report."""

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )


# ============================================================
# IMAGE-LEVEL PREDICTIONS
# ============================================================

def create_image_prediction_dataframe(
    validation_dataset: Any,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    idx_to_class: dict[int, str],
) -> pd.DataFrame:
    """Create a table containing one prediction per photograph."""

    dataframe = (
        validation_dataset.dataframe
        .copy()
        .reset_index(drop=True)
    )

    if len(dataframe) != len(true_labels):
        raise RuntimeError(
            "The validation dataframe and prediction count differ: "
            f"{len(dataframe)} rows versus "
            f"{len(true_labels)} predictions."
        )

    true_class_names = [
        idx_to_class[int(index)]
        for index in true_labels
    ]

    predicted_class_names = [
        idx_to_class[int(index)]
        for index in predicted_labels
    ]

    confidence_scores = probabilities.max(axis=1)

    prediction_data: dict[str, Any] = {
        "specimen_id": (
            dataframe["numCol"].astype(str)
        ),
        "processed_image_path": (
            dataframe[
                "processed_image_path"
            ].astype(str)
        ),
        "view_code": (
            dataframe["view_code"].astype(str)
        ),
        "fold": dataframe["fold"].astype(int),
        "true_class_index": true_labels,
        "true_class": true_class_names,
        "predicted_class_index": predicted_labels,
        "predicted_class": predicted_class_names,
        "confidence": confidence_scores,
        "correct": (
            true_labels == predicted_labels
        ),
    }

    for column in OPTIONAL_METADATA_COLUMNS:
        if column in dataframe.columns:
            prediction_data[column] = (
                normalize_metadata_series(
                    dataframe[column]
                )
            )

    scientific_name_column = (
        find_scientific_name_column(dataframe)
    )

    specimen_ids = dataframe[
        "numCol"
    ].astype(str)

    if scientific_name_column is not None:
        scientific_names = normalize_metadata_series(
            dataframe[scientific_name_column]
        )

        prediction_data["specimen_name"] = [
            (
                f"{specimen_id} - {scientific_name}"
                if scientific_name
                else specimen_id
            )
            for specimen_id, scientific_name
            in zip(
                specimen_ids,
                scientific_names,
            )
        ]
    else:
        prediction_data["specimen_name"] = (
            specimen_ids
        )

    predictions_dataframe = pd.DataFrame(
        prediction_data
    )

    for class_index, class_name in sorted(
        idx_to_class.items()
    ):
        safe_name = safe_column_name(class_name)

        probability_column = (
            f"probability_{class_index}_{safe_name}"
        )

        predictions_dataframe[
            probability_column
        ] = probabilities[:, class_index]

    return predictions_dataframe


# ============================================================
# SPECIMEN-LEVEL PREDICTIONS
# ============================================================

def validate_specimen_labels(
    image_predictions: pd.DataFrame,
) -> None:
    """Ensure every specimen has only one true label."""

    labels_per_specimen = (
        image_predictions
        .groupby("specimen_id")["true_class"]
        .nunique()
    )

    invalid_specimens = labels_per_specimen[
        labels_per_specimen > 1
    ]

    if not invalid_specimens.empty:
        raise RuntimeError(
            "Some specimens contain more than one true taxonomic "
            "label: "
            f"{invalid_specimens.index.tolist()[:10]}"
        )


def create_specimen_prediction_dataframe(
    image_predictions: pd.DataFrame,
    idx_to_class: dict[int, str],
) -> pd.DataFrame:
    """Average image probabilities to obtain specimen predictions."""

    validate_specimen_labels(
        image_predictions=image_predictions
    )

    probability_columns = [
        column
        for column in image_predictions.columns
        if column.startswith("probability_")
    ]

    if not probability_columns:
        raise RuntimeError(
            "No probability columns were found in the "
            "image-level prediction table."
        )

    metadata_columns = [
        column
        for column in (
            "specimen_name",
            "scientific_name",
            "scientificName",
            "family",
            "genus",
            "specific_epithet",
            "specificEpithet",
            "view_code",
            "fold",
            "true_class",
            "true_class_index",
        )
        if column in image_predictions.columns
    ]

    aggregation_rules: dict[str, str] = {
        column: "first"
        for column in metadata_columns
    }

    aggregation_rules.update(
        {
            column: "mean"
            for column in probability_columns
        }
    )

    specimen_predictions = (
        image_predictions
        .groupby(
            "specimen_id",
            as_index=False,
        )
        .agg(aggregation_rules)
    )

    image_counts = (
        image_predictions
        .groupby("specimen_id")
        .size()
        .rename("number_of_images")
        .reset_index()
    )

    specimen_predictions = (
        specimen_predictions.merge(
            image_counts,
            on="specimen_id",
            how="left",
        )
    )

    probability_matrix = specimen_predictions[
        probability_columns
    ].to_numpy(dtype=float)

    predicted_class_indices = (
        probability_matrix.argmax(axis=1)
    )

    predicted_class_names = [
        idx_to_class[int(index)]
        for index in predicted_class_indices
    ]

    specimen_predictions[
        "predicted_class_index"
    ] = predicted_class_indices

    specimen_predictions[
        "predicted_class"
    ] = predicted_class_names

    specimen_predictions[
        "confidence"
    ] = probability_matrix.max(axis=1)

    specimen_predictions[
        "correct"
    ] = (
        specimen_predictions["true_class"]
        == specimen_predictions[
            "predicted_class"
        ]
    )

    preferred_order = [
        "specimen_id",
        "specimen_name",
        "scientific_name",
        "scientificName",
        "family",
        "genus",
        "specific_epithet",
        "specificEpithet",
        "view_code",
        "fold",
        "number_of_images",
        "true_class_index",
        "true_class",
        "predicted_class_index",
        "predicted_class",
        "confidence",
        "correct",
    ]

    ordered_columns = [
        column
        for column in preferred_order
        if column in specimen_predictions.columns
    ]

    remaining_columns = [
        column
        for column in specimen_predictions.columns
        if column not in ordered_columns
    ]

    return specimen_predictions[
        ordered_columns + remaining_columns
    ]


# ============================================================
# CONFUSION MATRICES
# ============================================================

def create_confusion_matrices(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create raw and normalized confusion matrices."""

    raw_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
    )

    normalized_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
        normalize="true",
    )

    normalized_matrix = np.nan_to_num(
        normalized_matrix,
        nan=0.0,
    )

    return raw_matrix, normalized_matrix


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    """Save a confusion matrix as a PNG figure."""

    figure_size = max(
        8,
        int(len(class_names) * 1.2),
    )

    figure, axis = plt.subplots(
        figsize=(figure_size, figure_size)
    )

    matrix_image = axis.imshow(matrix)

    figure.colorbar(
        matrix_image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    maximum_value = (
        float(matrix.max())
        if matrix.size
        else 0.0
    )

    threshold = maximum_value / 2.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            displayed_value = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            axis.text(
                column_index,
                row_index,
                displayed_value,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_outputs(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    label_indices: list[int],
    class_names: list[str],
    experiment_directory: Path,
    evaluation_level: str,
    experiment_name: str,
) -> None:
    """Save raw and normalized confusion-matrix outputs."""

    (
        raw_matrix,
        normalized_matrix,
    ) = create_confusion_matrices(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        label_indices=label_indices,
    )

    raw_dataframe = pd.DataFrame(
        raw_matrix,
        index=class_names,
        columns=class_names,
    )

    normalized_dataframe = pd.DataFrame(
        normalized_matrix,
        index=class_names,
        columns=class_names,
    )

    raw_dataframe.to_csv(
        experiment_directory
        / f"confusion_matrix_{evaluation_level}.csv",
        index_label="true_class",
    )

    normalized_dataframe.to_csv(
        experiment_directory
        / (
            f"confusion_matrix_"
            f"{evaluation_level}_normalized.csv"
        ),
        index_label="true_class",
    )

    save_confusion_matrix_plot(
        matrix=raw_matrix,
        class_names=class_names,
        output_path=(
            experiment_directory
            / f"confusion_matrix_{evaluation_level}.png"
        ),
        title=(
            f"{evaluation_level.capitalize()}-level "
            f"Confusion Matrix\n{experiment_name}"
        ),
        normalized=False,
    )

    save_confusion_matrix_plot(
        matrix=normalized_matrix,
        class_names=class_names,
        output_path=(
            experiment_directory
            / (
                f"confusion_matrix_"
                f"{evaluation_level}_normalized.png"
            )
        ),
        title=(
            f"Normalized {evaluation_level.capitalize()}-level "
            f"Confusion Matrix\n{experiment_name}"
        ),
        normalized=True,
    )


# ============================================================
# COMPLETE CHECKPOINT EVALUATION
# ============================================================

def evaluate_checkpoint(
    checkpoint_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint at image and specimen levels."""

    checkpoint_path = checkpoint_path.resolve()
    experiment_directory = checkpoint_path.parent

    summary_path = (
        experiment_directory
        / "evaluation_summary.json"
    )

    if summary_path.exists() and not overwrite:
        print(
            "Evaluation already exists. Skipping: "
            f"{summary_path}"
        )

        with summary_path.open(
            encoding="utf-8"
        ) as file:
            return json.load(file)

    device = select_device()

    print("=" * 72)
    print("BRACHYCERA CNN EVALUATION")
    print("=" * 72)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    (
        class_to_idx,
        idx_to_class,
    ) = normalize_class_mapping(checkpoint)

    (
        validation_loader,
        validation_dataset,
        generated_mapping,
    ) = create_validation_loader(
        checkpoint=checkpoint,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if generated_mapping != class_to_idx:
        raise RuntimeError(
            "The generated class mapping does not match "
            "the normalized checkpoint mapping."
        )

    model = reconstruct_model(
        checkpoint=checkpoint,
        device=device,
    )

    (
        image_true_labels,
        image_predicted_labels,
        image_probabilities,
    ) = run_inference(
        model=model,
        validation_loader=validation_loader,
        device=device,
    )

    label_indices = sorted(idx_to_class)

    class_names = [
        idx_to_class[index]
        for index in label_indices
    ]

    # --------------------------------------------------------
    # Image-level evaluation
    # --------------------------------------------------------

    image_metrics = calculate_metrics(
        true_labels=image_true_labels,
        predicted_labels=image_predicted_labels,
    )

    image_predictions = (
        create_image_prediction_dataframe(
            validation_dataset=validation_dataset,
            true_labels=image_true_labels,
            predicted_labels=image_predicted_labels,
            probabilities=image_probabilities,
            idx_to_class=idx_to_class,
        )
    )

    image_predictions.to_csv(
        experiment_directory
        / "per_image_predictions.csv",
        index=False,
    )

    image_report = (
        create_classification_report_dataframe(
            true_labels=image_true_labels,
            predicted_labels=image_predicted_labels,
            class_names=class_names,
            label_indices=label_indices,
        )
    )

    image_report.to_csv(
        experiment_directory
        / "classification_report_image.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Specimen-level evaluation
    # --------------------------------------------------------

    specimen_predictions = (
        create_specimen_prediction_dataframe(
            image_predictions=image_predictions,
            idx_to_class=idx_to_class,
        )
    )

    specimen_predictions.to_csv(
        experiment_directory
        / "per_specimen_predictions.csv",
        index=False,
    )

    specimen_true_labels = specimen_predictions[
        "true_class_index"
    ].to_numpy(dtype=int)

    specimen_predicted_labels = specimen_predictions[
        "predicted_class_index"
    ].to_numpy(dtype=int)

    specimen_metrics = calculate_metrics(
        true_labels=specimen_true_labels,
        predicted_labels=specimen_predicted_labels,
    )

    specimen_report = (
        create_classification_report_dataframe(
            true_labels=specimen_true_labels,
            predicted_labels=(
                specimen_predicted_labels
            ),
            class_names=class_names,
            label_indices=label_indices,
        )
    )

    specimen_report.to_csv(
        experiment_directory
        / "classification_report_specimen.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    experiment_name = (
        f"{checkpoint['architecture']} | "
        f"{checkpoint['taxonomic_level']} | "
        f"{checkpoint['view_code']} | "
        f"fold {checkpoint['validation_fold']}"
    )

    save_confusion_outputs(
        true_labels=image_true_labels,
        predicted_labels=image_predicted_labels,
        label_indices=label_indices,
        class_names=class_names,
        experiment_directory=experiment_directory,
        evaluation_level="image",
        experiment_name=experiment_name,
    )

    save_confusion_outputs(
        true_labels=specimen_true_labels,
        predicted_labels=specimen_predicted_labels,
        label_indices=label_indices,
        class_names=class_names,
        experiment_directory=experiment_directory,
        evaluation_level="specimen",
        experiment_name=experiment_name,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    evaluation_summary: dict[str, Any] = {
        "architecture": str(
            checkpoint["architecture"]
        ),
        "taxonomic_level": str(
            checkpoint["taxonomic_level"]
        ),
        "view_code": str(
            checkpoint["view_code"]
        ),
        "validation_fold": int(
            checkpoint["validation_fold"]
        ),
        "number_of_classes": len(class_to_idx),
        "checkpoint_epoch": int(
            checkpoint.get("epoch", -1)
        ),
        "checkpoint_path": str(checkpoint_path),
        "image_level": {
            "number_of_images": int(
                len(image_predictions)
            ),
            **image_metrics,
        },
        "specimen_level": {
            "number_of_specimens": int(
                len(specimen_predictions)
            ),
            **specimen_metrics,
        },
    }

    if "validation_metrics" in checkpoint:
        evaluation_summary[
            "checkpoint_validation_metrics"
        ] = checkpoint["validation_metrics"]

    save_json(
        data=evaluation_summary,
        output_path=summary_path,
    )

    print("\nIMAGE-LEVEL RESULTS")
    print("-" * 72)
    print(
        f"Images:            "
        f"{len(image_predictions)}"
    )
    print(
        f"Accuracy:          "
        f"{image_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{image_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro precision:   "
        f"{image_metrics['precision_macro']:.4f}"
    )
    print(
        f"Macro recall:      "
        f"{image_metrics['recall_macro']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{image_metrics['f1_macro']:.4f}"
    )

    print("\nSPECIMEN-LEVEL RESULTS")
    print("-" * 72)
    print(
        f"Specimens:         "
        f"{len(specimen_predictions)}"
    )
    print(
        f"Accuracy:          "
        f"{specimen_metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{specimen_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro precision:   "
        f"{specimen_metrics['precision_macro']:.4f}"
    )
    print(
        f"Macro recall:      "
        f"{specimen_metrics['recall_macro']:.4f}"
    )
    print(
        f"Macro F1:          "
        f"{specimen_metrics['f1_macro']:.4f}"
    )

    print("-" * 72)
    print(
        f"Outputs saved in: {experiment_directory}"
    )

    return evaluation_summary


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Brachycera CNN checkpoint "
            "at image and specimen levels."
        )
    )

    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the best_model.pt checkpoint.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Evaluation batch size. "
            f"Default: {DEFAULT_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Number of DataLoader workers. "
            f"Default: {DEFAULT_NUM_WORKERS}."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing evaluation.",
    )

    return parser


def main() -> None:
    """Run evaluation from the command line."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=arguments.checkpoint,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
