"""
Run all Brachycera CNN training experiments.


"""
# import modules 
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from src.ml.models import SUPPORTED_ARCHITECTURES
from src.ml.trainer import (
    DEFAULT_OUTPUT_ROOT,
    TrainingConfiguration,
    run_training_experiment,
)

# Taxonomic levels
LEVELS = (
    "family",
    "genus",
)
# anatomical views used to evaluate 
# specimen orientation
VIEWS = (
    "FDT",
    "FFF",
    "FLP",
    "FLT",
)

# Five specimen-level cross-validation folds are evaluated.
# Each fold is used once as validation while the remaining
# folds provide the corresponding training data.
FOLDS = (
    1,
    2,
    3,
    4,
    5,
)

# Build the complete experimental matrix by combining all taxonomic
def build_experiment_matrix(
    levels: tuple[str, ...],
    views: tuple[str, ...],
    folds: tuple[int, ...],
    architectures: tuple[str, ...],
) -> list[dict[str, str | int]]:
    """Build every requested experiment combination."""
    
    # Store the configuration of every experiment that will be executed.
    experiments: list[dict[str, str | int]] = []
    
    # Generate the complete factorial combination of architectures,
    # taxonomic levels, anatomical views and validation folds.
    for architecture in architectures:
        for level in levels:
            for view_code in views:
                for validation_fold in folds:
                    experiments.append(
                        {
                            "architecture": architecture,
                            "level": level,
                            "view_code": view_code,
                            "validation_fold": validation_fold,
                        }
                    )

    return experiments

# Create a structured output path 
def experiment_output_directory(
    output_root: Path,
    architecture: str,
    level: str,
    view_code: str,
    validation_fold: int,
) -> Path:
    """Return the expected output directory for one experiment."""

    return (
        output_root
        / architecture
        / level
        / view_code
        / f"fold_{validation_fold}"
    )


def experiment_is_complete(
    output_root: Path,
    architecture: str,
    level: str,
    view_code: str,
    validation_fold: int,
) -> bool:
    """Check whether an experiment already completed successfully."""
    # Reconstruct the expected directory of the current experiment.
    output_directory = experiment_output_directory(
        output_root=output_root,
        architecture=architecture,
        level=level,
        view_code=view_code,
        validation_fold=validation_fold,
    )
    # These files represent the essential outputs of a successfully
    # completed training experiment.
    required_files = (
        output_directory / "best_model.pt",
        output_directory / "history.csv",
        output_directory / "training_summary.json",
    )

    return all(path.exists() for path in required_files)

# Save the execution status of all experiments in CSV and JSON formats,
# allowing training progress, failures, and completed runs to be tracked.
def save_run_status(
    records: list[dict[str, object]],
    output_root: Path,
) -> None:
    """Save progress in CSV and JSON formats."""

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_root / "all_experiments_status.csv"
    json_path = output_root / "all_experiments_status.json"

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

    if not records:
        return

    fieldnames = list(records[0].keys())

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def run_all_experiments(
    levels: tuple[str, ...],
    views: tuple[str, ...],
    folds: tuple[int, ...],
    architectures: tuple[str, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    num_workers: int,
    seed: int,
    pretrained: bool,
    freeze_backbone: bool,
    use_class_weights: bool,
    use_mixed_precision: bool,
    image_size: int,
    output_root: Path,
    skip_completed: bool,
    stop_on_error: bool,
) -> None:
    """Run the full experiment matrix."""

    experiments = build_experiment_matrix(
        levels=levels,
        views=views,
        folds=folds,
        architectures=architectures,
    )

    total_experiments = len(experiments)

    print("=" * 72)
    print("BRACHYCERA COMPLETE TRAINING MATRIX")
    print("=" * 72)
    print(f"Architectures : {architectures}")
    print(f"Levels        : {levels}")
    print(f"Views         : {views}")
    print(f"Folds         : {folds}")
    print(f"Experiments   : {total_experiments}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Output root   : {output_root}")
    print("=" * 72)

    status_records: list[dict[str, object]] = []

    successful = 0
    failed = 0
    skipped = 0

    complete_start_time = time.time()

    for experiment_number, experiment in enumerate(
        experiments,
        start=1,
    ):
        architecture = str(
            experiment["architecture"]
        )

        level = str(
            experiment["level"]
        )

        view_code = str(
            experiment["view_code"]
        )

        validation_fold = int(
            experiment["validation_fold"]
        )

        print()
        print("#" * 72)
        print(
            f"EXPERIMENT {experiment_number}/"
            f"{total_experiments}"
        )
        print("#" * 72)
        print(f"Architecture    : {architecture}")
        print(f"Level           : {level}")
        print(f"View            : {view_code}")
        print(f"Validation fold : {validation_fold}")
   

# Large experiment matrices may require several hours or days.
# When --skip-completed is enabled, experiments with all required
# output files are not trained again, allowing interrupted runs
# to resume safely.
        if (
            skip_completed
            and experiment_is_complete(
                output_root=output_root,
                architecture=architecture,
                level=level,
                view_code=view_code,
                validation_fold=validation_fold,
            )
        ):
            print("Status          : skipped; already complete")

            skipped += 1

            status_records.append(
                {
                    "experiment_number": experiment_number,
                    "architecture": architecture,
                    "level": level,
                    "view_code": view_code,
                    "validation_fold": validation_fold,
                    "status": "skipped",
                    "duration_seconds": 0.0,
                    "checkpoint": "",
                    "error": "",
                }
            )

            save_run_status(
                records=status_records,
                output_root=output_root,
            )

            continue

        configuration = TrainingConfiguration(
            level=level,
            view_code=view_code,
            validation_fold=validation_fold,
            architecture=architecture,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            use_class_weights=use_class_weights,
            use_mixed_precision=use_mixed_precision,
            image_size=image_size,
        )

        experiment_start_time = time.time()

        try:
            checkpoint_path = run_training_experiment(
                configuration=configuration,
                output_root=output_root,
            )

            duration_seconds = (
                time.time() - experiment_start_time
            )

            successful += 1

            status_records.append(
                {
                    "experiment_number": experiment_number,
                    "architecture": architecture,
                    "level": level,
                    "view_code": view_code,
                    "validation_fold": validation_fold,
                    "status": "successful",
                    "duration_seconds": duration_seconds,
                    "checkpoint": str(checkpoint_path),
                    "error": "",
                }
            )

        except Exception as error:
            duration_seconds = (
                time.time() - experiment_start_time
            )

            failed += 1

            error_message = (
                f"{type(error).__name__}: {error}"
            )

            print()
            print("EXPERIMENT FAILED")
            print(error_message)
            traceback.print_exc()

            status_records.append(
                {
                    "experiment_number": experiment_number,
                    "architecture": architecture,
                    "level": level,
                    "view_code": view_code,
                    "validation_fold": validation_fold,
                    "status": "failed",
                    "duration_seconds": duration_seconds,
                    "checkpoint": "",
                    "error": error_message,
                }
            )

            if stop_on_error:
                save_run_status(
                    records=status_records,
                    output_root=output_root,
                )

                raise

        save_run_status(
            records=status_records,
            output_root=output_root,
        )

    total_seconds = (
        time.time() - complete_start_time
    )

    print()
    print("=" * 72)
    print("COMPLETE TRAINING MATRIX FINISHED")
    print("=" * 72)
    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Skipped    : {skipped}")
    print(f"Total      : {total_experiments}")
    print(
        f"Duration   : "
        f"{total_seconds / 3600:.2f} hours"
    )
    print(
        f"Status CSV : "
        f"{output_root / 'all_experiments_status.csv'}"
    )
    print("=" * 72)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Brachycera CNN experiment matrix."
        )
    )

    parser.add_argument(
        "--levels",
        nargs="+",
        choices=LEVELS,
        default=list(LEVELS),
        help="Taxonomic levels to train.",
    )

    parser.add_argument(
        "--views",
        nargs="+",
        choices=VIEWS,
        default=list(VIEWS),
        help="Anatomical views to train.",
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(FOLDS),
        help="Cross-validation folds to train.",
    )

    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=SUPPORTED_ARCHITECTURES,
        default=list(SUPPORTED_ARCHITECTURES),
        help="CNN architectures to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
    )

    parser.add_argument(
        "--no-class-weights",
        action="store_true",
    )

    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip experiments whose outputs already exist.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one experiment fails.",
    )

    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    """Validate CLI parameters."""

    invalid_folds = [
        fold
        for fold in arguments.folds
        if fold not in FOLDS
    ]

    if invalid_folds:
        parser.error(
            f"Invalid folds: {invalid_folds}. "
            f"Expected folds 1 to 5."
        )

    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    if arguments.weight_decay < 0:
        parser.error("--weight-decay cannot be negative.")

    if arguments.patience < 1:
        parser.error("--patience must be at least 1.")

    if arguments.num_workers < 0:
        parser.error("--num-workers cannot be negative.")

    if arguments.image_size < 32:
        parser.error("--image-size must be at least 32.")


def main() -> None:
    """Command-line entry point."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    validate_arguments(
        parser=parser,
        arguments=arguments,
    )

    run_all_experiments(
        levels=tuple(arguments.levels),
        views=tuple(arguments.views),
        folds=tuple(arguments.folds),
        architectures=tuple(arguments.architectures),
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        patience=arguments.patience,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        pretrained=not arguments.no_pretrained,
        freeze_backbone=arguments.freeze_backbone,
        use_class_weights=(
            not arguments.no_class_weights
        ),
        use_mixed_precision=(
            not arguments.no_mixed_precision
        ),
        image_size=arguments.image_size,
        output_root=arguments.output_root,
        skip_completed=arguments.skip_completed,
        stop_on_error=arguments.stop_on_error,
    )


if __name__ == "__main__":
    main()
