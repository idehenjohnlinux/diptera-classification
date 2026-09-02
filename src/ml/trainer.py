"""
Training engine for Brachycera image classification.



"""
# imports modules
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ml.dataset import (
    BrachyceraImageDataset,
    create_datasets,
)
from src.ml.models import (
    SUPPORTED_ARCHITECTURES,
    count_parameters,
    create_model,
)
from src.ml.transforms import (
    get_training_transforms,
    get_validation_transforms,
)

from src.ml.reports import create_dataset_report
# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_ROOT: Final[Path] = (
    PROJECT_ROOT / "results" / "training"
)

VALID_LEVELS: Final[tuple[str, ...]] = (
    "family",
    "genus",
)

VALID_VIEWS: Final[tuple[str, ...]] = (
    "FDT",
    "FFF",
    "FLP",
    "FLT",
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================
# create a training configuration
@dataclass
class TrainingConfiguration:
    """Configuration for one CNN training experiment."""

    level: str
    view_code: str
    validation_fold: int
    architecture: str

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    patience: int = 7
    num_workers: int = 0
    seed: int = 42

    pretrained: bool = True
    freeze_backbone: bool = False
    use_class_weights: bool = True
    use_mixed_precision: bool = True

    image_size: int = 224
    minimum_delta: float = 0.0001


# ============================================================
# REPRODUCIBILITY
# ============================================================
# adding reproducibility 
def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Set a reproducible seed for each DataLoader worker."""

    worker_seed = (
        torch.initial_seed() + worker_id
    ) % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# DEVICE
# ============================================================
# adds cuda and return cuda when available 
# it serves as an additional help to cpu 
def select_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATA LOADERS
# ============================================================
# create a dataset loader 
def create_data_loaders(
    configuration: TrainingConfiguration,
    device: torch.device,
) -> tuple[
    DataLoader,
    DataLoader,
    BrachyceraImageDataset,
    BrachyceraImageDataset,
    dict[str, int],
]:
    """Create datasets and DataLoaders for one experiment."""

    image_size = (
        configuration.image_size,
        configuration.image_size,
    )

    training_transform = get_training_transforms(
        image_size=image_size,
    )

    validation_transform = get_validation_transforms(
        image_size=image_size,
    )

    (
        training_dataset,
        validation_dataset,
        class_to_idx,
    ) = create_datasets(
        level=configuration.level,
        view_code=configuration.view_code,
        validation_fold=configuration.validation_fold,
        train_transform=training_transform,
        validation_transform=validation_transform,
        return_metadata=False,
        save_mapping=True,
    )

    generator = torch.Generator()
    generator.manual_seed(configuration.seed)

    common_loader_arguments = {
        "batch_size": configuration.batch_size,
        "num_workers": configuration.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (
            configuration.num_workers > 0
        ),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    training_loader = DataLoader(
        dataset=training_dataset,
        shuffle=True,
        drop_last=False,
        **common_loader_arguments,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    return (
        training_loader,
        validation_loader,
        training_dataset,
        validation_dataset,
        class_to_idx,
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================
# calculate class weight 
def calculate_class_weights(
    dataset: BrachyceraImageDataset,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> Tensor:
    """Calculate inverse-frequency class weights.

    Rare classes receive larger weights in the loss function, reducing
    bias towards classes with many more training images.
    """

    class_counts = dataset.get_class_counts()

    total_images = float(class_counts.sum())
    number_of_classes = len(class_to_idx)

    weights = torch.ones(
        number_of_classes,
        dtype=torch.float32,
    )

    for class_name, class_index in class_to_idx.items():
        class_count = int(
            class_counts.get(class_name, 0)
        )

        if class_count < 1:
            raise ValueError(
                "A training class contains no images: "
                f"{class_name}"
            )

        weights[class_index] = (
            total_images
            / (number_of_classes * class_count)
        )

    return weights.to(device)


# ============================================================
# METRIC ACCUMULATOR
# ============================================================
# accumulate all metrics 
class ClassificationAccumulator:
    """Accumulate classification results across mini-batches."""

    def __init__(self, number_of_classes: int) -> None:
        self.number_of_classes = number_of_classes
        self.total_examples = 0
        self.total_correct = 0

        self.confusion_matrix = torch.zeros(
            (
                number_of_classes,
                number_of_classes,
            ),
            dtype=torch.long,
        )

    def update(
        self,
        outputs: Tensor,
        targets: Tensor,
    ) -> None:
        """Add one batch to the accumulated metrics."""

        predictions = outputs.argmax(dim=1)

        predictions_cpu = (
            predictions.detach().cpu().long()
        )

        targets_cpu = (
            targets.detach().cpu().long()
        )

        self.total_examples += targets_cpu.numel()

        self.total_correct += int(
            (
                predictions_cpu == targets_cpu
            ).sum().item()
        )

        combined_indices = (
            targets_cpu * self.number_of_classes
            + predictions_cpu
        )

        batch_confusion = torch.bincount(
            combined_indices,
            minlength=(
                self.number_of_classes
                * self.number_of_classes
            ),
        )

        batch_confusion = batch_confusion.reshape(
            self.number_of_classes,
            self.number_of_classes,
        )

        self.confusion_matrix += batch_confusion

    def calculate(self) -> dict[str, float]:
        """Calculate accuracy and macro-averaged metrics."""

        if self.total_examples == 0:
            raise ValueError(
                "No examples were accumulated."
            )

        matrix = self.confusion_matrix.float()

        true_positives = torch.diag(matrix)

        false_positives = (
            matrix.sum(dim=0) - true_positives
        )

        false_negatives = (
            matrix.sum(dim=1) - true_positives
        )

        precision = true_positives / (
            true_positives
            + false_positives
        ).clamp(min=1.0)

        recall = true_positives / (
            true_positives
            + false_negatives
        ).clamp(min=1.0)

        f1_score = (
            2.0 * precision * recall
            / (precision + recall).clamp(min=1e-12)
        )

        accuracy = (
            self.total_correct / self.total_examples
        )

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(
                precision.mean().item()
            ),
            "recall_macro": float(
                recall.mean().item()
            ),
            "f1_macro": float(
                f1_score.mean().item()
            ),
        }


# ============================================================
# ONE TRAINING EPOCH
# ============================================================

# trains the epoch
def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_mixed_precision: bool,
    number_of_classes: int,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """Train a model for one complete epoch."""

    model.train()

    accumulated_loss = 0.0
    processed_examples = 0

    metrics = ClassificationAccumulator(
        number_of_classes=number_of_classes,
    )

    progress_bar = tqdm(
        data_loader,
        desc=f"Training {epoch}/{total_epochs}",
        leave=False,
    )

    for images, targets in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_mixed_precision,
        ):
            outputs = model(images)
            loss = loss_function(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)

        accumulated_loss += (
            float(loss.item()) * batch_size
        )

        processed_examples += batch_size

        metrics.update(
            outputs=outputs,
            targets=targets,
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    epoch_metrics = metrics.calculate()

    epoch_metrics["loss"] = (
        accumulated_loss / processed_examples
    )

    return epoch_metrics


# ============================================================
# ONE VALIDATION EPOCH
# ============================================================

def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    use_mixed_precision: bool,
    number_of_classes: int,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    """Evaluate a model on the validation dataset."""

    model.eval()

    accumulated_loss = 0.0
    processed_examples = 0

    metrics = ClassificationAccumulator(
        number_of_classes=number_of_classes,
    )

    progress_bar = tqdm(
        data_loader,
        desc=f"Validation {epoch}/{total_epochs}",
        leave=False,
    )

    with torch.no_grad():
        for images, targets in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=device.type,
                enabled=use_mixed_precision,
            ):
                outputs = model(images)
                loss = loss_function(
                    outputs,
                    targets,
                )

            batch_size = targets.size(0)

            accumulated_loss += (
                float(loss.item()) * batch_size
            )

            processed_examples += batch_size

            metrics.update(
                outputs=outputs,
                targets=targets,
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
            )

    epoch_metrics = metrics.calculate()

    epoch_metrics["loss"] = (
        accumulated_loss / processed_examples
    )

    return epoch_metrics


# ============================================================
# OUTPUT FILES
# ============================================================
# create the output files
def build_experiment_directory(
    configuration: TrainingConfiguration,
    output_root: Path,
) -> Path:
    """Create the directory for one experiment."""

    experiment_directory = (
        output_root
        / configuration.architecture
        / configuration.level
        / configuration.view_code
        / f"fold_{configuration.validation_fold}"
    )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiment_directory


def save_configuration(
    configuration: TrainingConfiguration,
    output_path: Path,
) -> None:
    """Save the training configuration as JSON."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(configuration),
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_history_csv(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save epoch-level training history as CSV."""

    if not history:
        raise ValueError(
            "Training history is empty."
        )

    fieldnames = list(history[0].keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def save_history_json(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """Save epoch-level training history as JSON."""

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    configuration: TrainingConfiguration,
    class_to_idx: dict[str, int],
    epoch: int,
    validation_metrics: dict[str, float],
) -> None:
    """Save a complete reusable model checkpoint."""

    checkpoint = {
        "epoch": epoch,
        "architecture": configuration.architecture,
        "taxonomic_level": configuration.level,
        "view_code": configuration.view_code,
        "validation_fold": (
            configuration.validation_fold
        ),
        "image_size": configuration.image_size,
        "class_to_idx": class_to_idx,
        "idx_to_class": {
            index: label
            for label, index in class_to_idx.items()
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "validation_metrics": validation_metrics,
        "configuration": asdict(configuration),
    }

    torch.save(
        checkpoint,
        output_path,
    )


# ============================================================
# COMPLETE TRAINING EXPERIMENT
# ============================================================

def run_training_experiment(
    configuration: TrainingConfiguration,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Run one complete CNN training experiment."""

    set_random_seed(configuration.seed)

    device = select_device()

    use_mixed_precision = (
        configuration.use_mixed_precision
        and device.type == "cuda"
    )

    experiment_directory = (
        build_experiment_directory(
            configuration=configuration,
            output_root=output_root,
        )
    )

    configuration_path = (
        experiment_directory
        / "training_configuration.json"
    )

    history_csv_path = (
        experiment_directory
        / "history.csv"
    )

    history_json_path = (
        experiment_directory
        / "history.json"
    )

    best_checkpoint_path = (
        experiment_directory
        / "best_model.pt"
    )

    last_checkpoint_path = (
        experiment_directory
        / "last_model.pt"
    )

    save_configuration(
        configuration=configuration,
        output_path=configuration_path,
    )

    (
        training_loader,
        validation_loader,
        training_dataset,
        validation_dataset,
        class_to_idx,
    ) = create_data_loaders(
        configuration=configuration,
        device=device,
    )
    training_dataset_summary = create_dataset_report(
        dataset=training_dataset,
        level=configuration.level,
        subset="training",
        output_directory=experiment_directory,
        print_report=True,
    )

    validation_dataset_summary = create_dataset_report(
        dataset=validation_dataset,
        level=configuration.level,
        subset="validation",
        output_directory=experiment_directory,
        print_report=True,
    )
    # ============================================================
    # DATASET TOTALS
    # ============================================================

    training_specimen_count = int(
        training_dataset_summary["Specimens"].sum()
    )

    validation_specimen_count = int(
        validation_dataset_summary["Specimens"].sum()
    )

    training_image_count = int(
        training_dataset_summary["Images"].sum()
    )

    validation_image_count = int(
        validation_dataset_summary["Images"].sum()
    )

    number_of_classes = len(class_to_idx)

    model = create_model(
        architecture=configuration.architecture,
        num_classes=number_of_classes,
        pretrained=configuration.pretrained,
        freeze_backbone=(
            configuration.freeze_backbone
        ),
    )

    model = model.to(device)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    if configuration.use_class_weights:
        class_weights = calculate_class_weights(
            dataset=training_dataset,
            class_to_idx=class_to_idx,
            device=device,
        )
    else:
        class_weights = None

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=use_mixed_precision,
    )

    print()
    print("=" * 70)
    print("BRACHYCERA CNN TRAINING")
    print("=" * 70)
    print(
        f"Architecture        : "
        f"{configuration.architecture}"
    )
    print(
        f"Taxonomic level     : "
        f"{configuration.level}"
    )
    print(
        f"View                : "
        f"{configuration.view_code}"
    )
    print(
        f"Validation fold     : "
        f"{configuration.validation_fold}"
    )
    print(f"Device              : {device}")
    print(
        f"Mixed precision     : "
        f"{use_mixed_precision}"
    )
    print(
        f"Pretrained          : "
        f"{configuration.pretrained}"
    )
    print(
        f"Backbone frozen     : "
        f"{configuration.freeze_backbone}"
    )
    print(
        f"Weighted loss       : "
        f"{configuration.use_class_weights}"
    )
    print(f"Classes             : {number_of_classes}")
    print(
        f"Training images     : "
        f"{len(training_dataset)}"
    )
    print(f"Training images      : {training_image_count}")
    print(f"Training specimens   : {training_specimen_count}")

    print(f"Validation images    : {validation_image_count}")
    print(f"Validation specimens : {validation_specimen_count}")
    print(
        f"Training specimens  : "
        f"{training_dataset.get_dataframe()['numCol'].nunique()}"
    )
    print(
        f"Validation images   : "
        f"{len(validation_dataset)}"
    )
    print(
        f"Validation specimens: "
        f"{validation_dataset.get_dataframe()['numCol'].nunique()}"
    )
    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(
        f"Output directory    : "
        f"{experiment_directory}"
    )
    print("=" * 70)
    

    history: list[
        dict[str, float | int]
    ] = []

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_start_time = time.time()

    for epoch in range(
        1,
        configuration.epochs + 1,
    ):
        epoch_start_time = time.time()

        training_metrics = train_one_epoch(
            model=model,
            data_loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
            number_of_classes=number_of_classes,
            epoch=epoch,
            total_epochs=configuration.epochs,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
            use_mixed_precision=use_mixed_precision,
            number_of_classes=number_of_classes,
            epoch=epoch,
            total_epochs=configuration.epochs,
        )

        scheduler.step(
            validation_metrics["loss"]
        )

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        epoch_duration = (
            time.time() - epoch_start_time
        )

        epoch_record: dict[
            str,
            float | int
        ] = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": training_metrics["loss"],
            "train_accuracy": (
                training_metrics["accuracy"]
            ),
            "train_precision_macro": (
                training_metrics["precision_macro"]
            ),
            "train_recall_macro": (
                training_metrics["recall_macro"]
            ),
            "train_f1_macro": (
                training_metrics["f1_macro"]
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_accuracy": (
                validation_metrics["accuracy"]
            ),
            "validation_precision_macro": (
                validation_metrics[
                    "precision_macro"
                ]
            ),
            "validation_recall_macro": (
                validation_metrics[
                    "recall_macro"
                ]
            ),
            "validation_f1_macro": (
                validation_metrics["f1_macro"]
            ),
            "epoch_seconds": epoch_duration,
        }

        history.append(epoch_record)

        save_history_csv(
            history=history,
            output_path=history_csv_path,
        )

        save_history_json(
            history=history,
            output_path=history_json_path,
        )

        improvement = (
            best_validation_loss
            - validation_metrics["loss"]
        )

        if improvement > configuration.minimum_delta:
            best_validation_loss = (
                validation_metrics["loss"]
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                configuration=configuration,
                class_to_idx=class_to_idx,
                epoch=epoch,
                validation_metrics=validation_metrics,
            )

            checkpoint_status = "best model saved"

        else:
            epochs_without_improvement += 1
            checkpoint_status = "no improvement"

        save_checkpoint(
            output_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            configuration=configuration,
            class_to_idx=class_to_idx,
            epoch=epoch,
            validation_metrics=validation_metrics,
        )

        print()
        print(
            f"Epoch {epoch:02d}/"
            f"{configuration.epochs:02d}"
        )
        print(
            f"Train loss: "
            f"{training_metrics['loss']:.4f} | "
            f"accuracy: "
            f"{training_metrics['accuracy']:.4f} | "
            f"F1: "
            f"{training_metrics['f1_macro']:.4f}"
        )
        print(
            f"Valid loss: "
            f"{validation_metrics['loss']:.4f} | "
            f"accuracy: "
            f"{validation_metrics['accuracy']:.4f} | "
            f"F1: "
            f"{validation_metrics['f1_macro']:.4f}"
        )
        print(
            f"Learning rate: "
            f"{current_learning_rate:.8f} | "
            f"{checkpoint_status}"
        )
        print(
            f"Epoch duration: "
            f"{epoch_duration:.1f} seconds"
        )

        if (
            epochs_without_improvement
            >= configuration.patience
        ):
            print()
            print(
                "Early stopping activated after "
                f"{configuration.patience} epochs "
                "without validation-loss improvement."
            )
            break

    total_training_seconds = (
        time.time() - training_start_time
    )

    summary = {
        "architecture": configuration.architecture,
        "taxonomic_level": configuration.level,
        "view_code": configuration.view_code,
        "validation_fold": (
            configuration.validation_fold
        ),
        "number_of_classes": number_of_classes,
        "training_images": len(training_dataset),
        "training_images": training_image_count,
        "training_specimens": training_specimen_count,
        "validation_images": validation_image_count,
        "validation_specimens": validation_specimen_count,
        "validation_images": len(
            validation_dataset
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "epochs_completed": len(history),
        "training_seconds": (
            total_training_seconds
        ),
        "best_checkpoint": str(
            best_checkpoint_path
        ),
    }

    summary_path = (
        experiment_directory
        / "training_summary.json"
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

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best epoch          : {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(
        f"Completed epochs    : "
        f"{len(history)}"
    )
    print(
        f"Total duration      : "
        f"{total_training_seconds / 60:.2f} minutes"
    )
    print(
        f"Best checkpoint     : "
        f"{best_checkpoint_path}"
    )
    print("=" * 70)

    return best_checkpoint_path


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Train one Brachycera CNN classification experiment."
        )
    )

    parser.add_argument(
        "--level",
        choices=VALID_LEVELS,
        required=True,
        help="Taxonomic classification level.",
    )

    parser.add_argument(
        "--view",
        choices=VALID_VIEWS,
        required=True,
        help="Morphological image view.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Validation fold number.",
    )

    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        required=True,
        help="CNN architecture.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Images per mini-batch.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Initial AdamW learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="AdamW weight-decay value.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="CNN image size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet pretrained weights.",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the final output layer.",
    )

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable weighted cross-entropy loss.",
    )

    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable CUDA mixed-precision training.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for training outputs.",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    """Validate command-line values."""

    if arguments.fold < 1:
        parser.error(
            "--fold must be at least 1."
        )

    if arguments.epochs < 1:
        parser.error(
            "--epochs must be at least 1."
        )

    if arguments.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1."
        )

    if arguments.learning_rate <= 0:
        parser.error(
            "--learning-rate must be positive."
        )

    if arguments.weight_decay < 0:
        parser.error(
            "--weight-decay cannot be negative."
        )

    if arguments.patience < 1:
        parser.error(
            "--patience must be at least 1."
        )

    if arguments.num_workers < 0:
        parser.error(
            "--num-workers cannot be negative."
        )

    if arguments.image_size < 32:
        parser.error(
            "--image-size must be at least 32."
        )


def main() -> None:
    """Command-line entry point."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    validate_arguments(
        parser=parser,
        arguments=arguments,
    )

    configuration = TrainingConfiguration(
        level=arguments.level,
        view_code=arguments.view,
        validation_fold=arguments.fold,
        architecture=arguments.architecture,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        patience=arguments.patience,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        pretrained=not arguments.no_pretrained,
        freeze_backbone=(
            arguments.freeze_backbone
        ),
        use_class_weights=(
            not arguments.no_class_weights
        ),
        use_mixed_precision=(
            not arguments.no_mixed_precision
        ),
        image_size=arguments.image_size,
    )

    run_training_experiment(
        configuration=configuration,
        output_root=arguments.output_root,
    )


if __name__ == "__main__":
    main()
