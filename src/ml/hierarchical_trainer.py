

# import modules
from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.ml.hierarchical_dataset import (
    DEFAULT_IDENTIFICATION_DATASET,
    HierarchicalBrachyceraDataset,
    build_class_mapping,
    build_family_to_genera_mapping,
    build_genus_to_family_mapping,
    load_hierarchical_training_dataframe,
    save_hierarchical_mappings,
)
from src.ml.hierarchical_loss import HierarchicalLoss
from src.ml.hierarchical_model import create_hierarchical_model
from src.ml.transforms import (
    get_training_transforms,
    get_validation_transforms,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
)


# ============================================================
# CONFIGURATION
# ============================================================
# creates class configuration 
@dataclass
class HierarchicalTrainingConfiguration:
    """Configuration for hierarchical model training."""

    input_csv: str = str(DEFAULT_IDENTIFICATION_DATASET)
    output_directory: str = str(DEFAULT_OUTPUT_DIRECTORY)

    architecture: str = "efficientnet_b0"
    view_code: str = "FLP"

    epochs: int = 20
    batch_size: int = 16
    image_size: int = 224
    num_workers: int = 0

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    family_weight: float = 1.0
    genus_weight: float = 0.5
    consistency_weight: float = 0.2

    label_smoothing: float = 0.0

    family_dropout: float = 0.2
    genus_dropout: float = 0.3
    genus_hidden_dimension: int = 512

    validation_fraction: float = 0.20
    seed: int = 42

    pretrained: bool = True
    use_mixed_precision: bool = True

    early_stopping_patience: int = 6
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5

    save_every_epoch: bool = False


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

# ensure that the trainer is reproducible 
def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# DATA SPLITTING
# ============================================================

# splits datasets 
def stratified_family_split(
    dataframe: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split specimens while preserving family representation.

    Every family keeps at least one specimen in the training set.
    Families represented by only one specimen remain in training.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    random_generator = np.random.default_rng(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []

    for _, family_group in dataframe.groupby(
        "family",
        sort=True,
    ):
        indices = family_group.index.to_numpy().copy()
        random_generator.shuffle(indices)

        number_of_specimens = len(indices)

        if number_of_specimens == 1:
            train_indices.extend(indices.tolist())
            continue

        number_for_validation = max(
            1,
            int(round(
                number_of_specimens
                * validation_fraction
            )),
        )

        number_for_validation = min(
            number_for_validation,
            number_of_specimens - 1,
        )

        validation_indices.extend(
            indices[:number_for_validation].tolist()
        )

        train_indices.extend(
            indices[number_for_validation:].tolist()
        )

    train_dataframe = dataframe.loc[
        sorted(train_indices)
    ].reset_index(drop=True)

    validation_dataframe = dataframe.loc[
        sorted(validation_indices)
    ].reset_index(drop=True)

    if train_dataframe.empty:
        raise ValueError("Training split is empty.")

    if validation_dataframe.empty:
        raise ValueError("Validation split is empty.")

    return train_dataframe, validation_dataframe


# ============================================================
# DATASETS AND LOADERS
# ============================================================

# loads the datasets for training 
def create_training_components(
    configuration: HierarchicalTrainingConfiguration,
) -> tuple[
    DataLoader,
    DataLoader,
    dict[str, int],
    dict[str, int],
    dict[int, int],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create mappings, datasets and data loaders."""

    dataframe = load_hierarchical_training_dataframe(
        Path(configuration.input_csv)
    )

    family_to_idx = build_class_mapping(
        dataframe["family"]
    )

    genus_labelled_dataframe = dataframe[
        dataframe["has_genus_label"]
    ].copy()

    genus_to_idx = build_class_mapping(
        genus_labelled_dataframe["genus"]
    )

    genus_to_family = build_genus_to_family_mapping(
        dataframe=genus_labelled_dataframe,
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
    )

    save_hierarchical_mappings(
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
        genus_to_family=genus_to_family,
    )

    (
        train_dataframe,
        validation_dataframe,
    ) = stratified_family_split(
        dataframe=dataframe,
        validation_fraction=(
            configuration.validation_fraction
        ),
        seed=configuration.seed,
    )

    training_transform = get_training_transforms(
        image_size=(
            configuration.image_size,
            configuration.image_size,
        )
    )

    validation_transform = get_validation_transforms(
        image_size=(
            configuration.image_size,
            configuration.image_size,
        )
    )

    training_dataset = HierarchicalBrachyceraDataset(
        dataframe=train_dataframe,
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
        transform=training_transform,
        return_metadata=False,
    )

    validation_dataset = HierarchicalBrachyceraDataset(
        dataframe=validation_dataframe,
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
        transform=validation_transform,
        return_metadata=False,
    )

    pin_memory = torch.cuda.is_available()

    # drop_last prevents BatchNorm1d from receiving a training
    # batch containing only one specimen.
    training_loader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=configuration.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        shuffle=False,
        num_workers=configuration.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return (
        training_loader,
        validation_loader,
        family_to_idx,
        genus_to_idx,
        genus_to_family,
        train_dataframe,
        validation_dataframe,
    )


# ============================================================
# METRICS
# ============================================================

# calculates the metrics 
def calculate_batch_statistics(
    family_logits: Tensor,
    genus_logits: Tensor,
    family_targets: Tensor,
    genus_targets: Tensor,
    genus_to_family_tensor: Tensor,
) -> dict[str, int]:
    """Calculate classification statistics for one batch."""

    family_predictions = family_logits.argmax(dim=1)
    genus_predictions = genus_logits.argmax(dim=1)

    family_correct = (
        family_predictions == family_targets
    ).sum().item()

    family_total = family_targets.numel()

    genus_label_mask = genus_targets != -1

    if genus_label_mask.any():
        genus_correct = (
            genus_predictions[genus_label_mask]
            == genus_targets[genus_label_mask]
        ).sum().item()

        genus_total = genus_label_mask.sum().item()
    else:
        genus_correct = 0
        genus_total = 0

    predicted_genus_families = genus_to_family_tensor[
        genus_predictions
    ]

    hierarchy_correct = (
        predicted_genus_families
        == family_predictions
    ).sum().item()

    hierarchy_total = family_total

    true_family_consistency_correct = (
        predicted_genus_families
        == family_targets
    ).sum().item()

    return {
        "family_correct": int(family_correct),
        "family_total": int(family_total),
        "genus_correct": int(genus_correct),
        "genus_total": int(genus_total),
        "hierarchy_correct": int(hierarchy_correct),
        "hierarchy_total": int(hierarchy_total),
        "true_family_consistency_correct": int(
            true_family_consistency_correct
        ),
    }


def safe_percentage(
    numerator: float,
    denominator: float,
) -> float:
    """Return a percentage without division by zero."""

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


# ============================================================
# EPOCH EXECUTION
# ============================================================

# executes each epoch for training 
def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: HierarchicalLoss,
    device: torch.device,
    genus_to_family_tensor: Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_mixed_precision: bool = False,
) -> dict[str, float]:
    """Run one training or validation epoch."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    accumulated = {
        "total_loss": 0.0,
        "family_loss": 0.0,
        "genus_loss": 0.0,
        "consistency_loss": 0.0,
    }

    statistics = {
        "family_correct": 0,
        "family_total": 0,
        "genus_correct": 0,
        "genus_total": 0,
        "hierarchy_correct": 0,
        "hierarchy_total": 0,
        "true_family_consistency_correct": 0,
    }

    number_of_batches = 0

    for batch in data_loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        family_targets = batch["family_target"].to(
            device,
            non_blocking=True,
        )

        genus_targets = batch["genus_target"].to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        gradient_context = (
            torch.enable_grad()
            if is_training
            else torch.no_grad()
        )

        with gradient_context:
            autocast_enabled = (
                use_mixed_precision
                and device.type == "cuda"
            )

            with torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
            ):
                outputs = model(images)

                losses = criterion(
                    family_logits=outputs[
                        "family_logits"
                    ],
                    genus_logits=outputs[
                        "genus_logits"
                    ],
                    family_targets=family_targets,
                    genus_targets=genus_targets,
                )

                total_loss = losses["total_loss"]

            if is_training:
                if scaler is not None and autocast_enabled:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    optimizer.step()

        batch_size = images.shape[0]

        for loss_name in accumulated:
            accumulated[loss_name] += (
                losses[loss_name].detach().item()
                * batch_size
            )

        batch_statistics = calculate_batch_statistics(
            family_logits=outputs[
                "family_logits"
            ].detach(),
            genus_logits=outputs[
                "genus_logits"
            ].detach(),
            family_targets=family_targets,
            genus_targets=genus_targets,
            genus_to_family_tensor=(
                genus_to_family_tensor
            ),
        )

        for statistic_name, statistic_value in (
            batch_statistics.items()
        ):
            statistics[statistic_name] += statistic_value

        number_of_batches += 1

    number_of_specimens = statistics["family_total"]

    if number_of_batches == 0 or number_of_specimens == 0:
        raise RuntimeError(
            "The data loader produced no usable batches."
        )

    return {
        "total_loss": (
            accumulated["total_loss"]
            / number_of_specimens
        ),
        "family_loss": (
            accumulated["family_loss"]
            / number_of_specimens
        ),
        "genus_loss": (
            accumulated["genus_loss"]
            / number_of_specimens
        ),
        "consistency_loss": (
            accumulated["consistency_loss"]
            / number_of_specimens
        ),
        "family_accuracy": safe_percentage(
            statistics["family_correct"],
            statistics["family_total"],
        ),
        "genus_accuracy": safe_percentage(
            statistics["genus_correct"],
            statistics["genus_total"],
        ),
        "hierarchy_consistency": safe_percentage(
            statistics["hierarchy_correct"],
            statistics["hierarchy_total"],
        ),
        "true_family_genus_consistency": (
            safe_percentage(
                statistics[
                    "true_family_consistency_correct"
                ],
                statistics["family_total"],
            )
        ),
        "family_specimens": float(
            statistics["family_total"]
        ),
        "genus_labelled_specimens": float(
            statistics["genus_total"]
        ),
    }


# ============================================================
# SAVING
# ============================================================

#saves the files 
def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    configuration: HierarchicalTrainingConfiguration,
    epoch: int,
    validation_metrics: dict[str, float],
    family_to_idx: dict[str, int],
    genus_to_idx: dict[str, int],
    genus_to_family: dict[int, int],
) -> None:
    """Save a complete hierarchical model checkpoint."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_type": "hierarchical_family_genus",
        "architecture": configuration.architecture,
        "view_code": configuration.view_code,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "configuration": asdict(configuration),
        "validation_metrics": validation_metrics,
        "family_to_idx": family_to_idx,
        "idx_to_family": {
            index: family
            for family, index in family_to_idx.items()
        },
        "genus_to_idx": genus_to_idx,
        "idx_to_genus": {
            index: genus
            for genus, index in genus_to_idx.items()
        },
        "genus_to_family_idx": genus_to_family,
        "saved_at": datetime.now().isoformat(),
    }

    torch.save(checkpoint, output_path)


def save_history(
    history: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """Save training history as JSON and CSV."""

    json_path = output_directory / "history.json"
    csv_path = output_directory / "history.csv"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )

    if history:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(history[0].keys()),
            )

            writer.writeheader()
            writer.writerows(history)


# ============================================================
# TRAINING
# ============================================================

# trains the model 
def train_hierarchical_model(
    configuration: HierarchicalTrainingConfiguration,
) -> dict[str, Any]:
    """Train and save the hierarchical production candidate."""

    set_random_seed(configuration.seed)

    device = select_device()

    output_directory = Path(
        configuration.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        training_loader,
        validation_loader,
        family_to_idx,
        genus_to_idx,
        genus_to_family,
        training_dataframe,
        validation_dataframe,
    ) = create_training_components(configuration)

    family_to_genera = build_family_to_genera_mapping(
        genus_to_family
    )

    model = create_hierarchical_model(
        number_of_families=len(family_to_idx),
        number_of_genera=len(genus_to_idx),
        pretrained=configuration.pretrained,
        family_dropout=configuration.family_dropout,
        genus_dropout=configuration.genus_dropout,
        genus_hidden_dimension=(
            configuration.genus_hidden_dimension
        ),
    ).to(device)

    criterion = HierarchicalLoss(
        number_of_families=len(family_to_idx),
        number_of_genera=len(genus_to_idx),
        family_to_genus_indices=family_to_genera,
        family_weight=configuration.family_weight,
        genus_weight=configuration.genus_weight,
        consistency_weight=(
            configuration.consistency_weight
        ),
        label_smoothing=configuration.label_smoothing,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=configuration.scheduler_factor,
        patience=configuration.scheduler_patience,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            configuration.use_mixed_precision
            and device.type == "cuda"
        ),
    )

    genus_to_family_tensor = torch.empty(
        len(genus_to_idx),
        dtype=torch.long,
        device=device,
    )

    for genus_index, family_index in (
        genus_to_family.items()
    ):
        genus_to_family_tensor[genus_index] = (
            family_index
        )

    history: list[dict[str, Any]] = []

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\nHierarchical training")
    print("=" * 72)
    print(f"Device: {device}")
    print(
        f"Training specimens: "
        f"{len(training_dataframe)}"
    )
    print(
        f"Validation specimens: "
        f"{len(validation_dataframe)}"
    )
    print(f"Families: {len(family_to_idx)}")
    print(f"Genera: {len(genus_to_idx)}")
    print(
        "Training genus-labelled specimens: "
        f"{int(training_dataframe['has_genus_label'].sum())}"
    )
    print(
        "Validation genus-labelled specimens: "
        f"{int(validation_dataframe['has_genus_label'].sum())}"
    )
    print("=" * 72)

    for epoch in range(1, configuration.epochs + 1):
        training_metrics = run_epoch(
            model=model,
            data_loader=training_loader,
            criterion=criterion,
            device=device,
            genus_to_family_tensor=(
                genus_to_family_tensor
            ),
            optimizer=optimizer,
            scaler=scaler,
            use_mixed_precision=(
                configuration.use_mixed_precision
            ),
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            genus_to_family_tensor=(
                genus_to_family_tensor
            ),
            optimizer=None,
            scaler=None,
            use_mixed_precision=(
                configuration.use_mixed_precision
            ),
        )

        scheduler.step(
            validation_metrics["total_loss"]
        )

        learning_rate = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
        }

        for name, value in training_metrics.items():
            epoch_record[f"train_{name}"] = value

        for name, value in validation_metrics.items():
            epoch_record[f"validation_{name}"] = value

        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{configuration.epochs:02d} | "
            f"Train loss: "
            f"{training_metrics['total_loss']:.4f} | "
            f"Val loss: "
            f"{validation_metrics['total_loss']:.4f} | "
            f"Family acc: "
            f"{validation_metrics['family_accuracy']:.2f}% | "
            f"Genus acc: "
            f"{validation_metrics['genus_accuracy']:.2f}% | "
            f"Hierarchy: "
            f"{validation_metrics['hierarchy_consistency']:.2f}%"
        )

        if configuration.save_every_epoch:
            save_checkpoint(
                output_path=(
                    output_directory
                    / f"checkpoint_epoch_{epoch:02d}.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                configuration=configuration,
                epoch=epoch,
                validation_metrics=validation_metrics,
                family_to_idx=family_to_idx,
                genus_to_idx=genus_to_idx,
                genus_to_family=genus_to_family,
            )

        current_validation_loss = validation_metrics[
            "total_loss"
        ]

        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=(
                    output_directory / "best_model.pt"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                configuration=configuration,
                epoch=epoch,
                validation_metrics=validation_metrics,
                family_to_idx=family_to_idx,
                genus_to_idx=genus_to_idx,
                genus_to_family=genus_to_family,
            )

            print("  ↳ Saved new best model.")

        else:
            epochs_without_improvement += 1

        save_history(
            history=history,
            output_directory=output_directory,
        )

        if (
            epochs_without_improvement
            >= configuration.early_stopping_patience
        ):
            print(
                "Early stopping activated after "
                f"{epochs_without_improvement} epochs "
                "without validation improvement."
            )
            break

    save_checkpoint(
        output_path=output_directory / "last_model.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        configuration=configuration,
        epoch=history[-1]["epoch"],
        validation_metrics={
            key.removeprefix("validation_"): value
            for key, value in history[-1].items()
            if key.startswith("validation_")
        },
        family_to_idx=family_to_idx,
        genus_to_idx=genus_to_idx,
        genus_to_family=genus_to_family,
    )

    summary = {
        "model_type": "hierarchical_family_genus",
        "architecture": configuration.architecture,
        "view_code": configuration.view_code,
        "device": str(device),
        "number_of_families": len(family_to_idx),
        "number_of_genera": len(genus_to_idx),
        "total_eligible_specimens": (
            len(training_dataframe)
            + len(validation_dataframe)
        ),
        "training_specimens": len(training_dataframe),
        "validation_specimens": len(
            validation_dataframe
        ),
        "training_genus_labelled_specimens": int(
            training_dataframe["has_genus_label"].sum()
        ),
        "validation_genus_labelled_specimens": int(
            validation_dataframe["has_genus_label"].sum()
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "completed_epochs": len(history),
        "best_checkpoint": str(
            output_directory / "best_model.pt"
        ),
        "last_checkpoint": str(
            output_directory / "last_model.pt"
        ),
        "configuration": asdict(configuration),
    }

    with (
        output_directory / "training_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nTraining completed")
    print("=" * 72)
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(
        f"Best checkpoint: "
        f"{output_directory / 'best_model.pt'}"
    )

    return summary


def main() -> None:
    """Command-line entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    configuration = (
        HierarchicalTrainingConfiguration()
    )

    train_hierarchical_model(configuration)


if __name__ == "__main__":
    main()
