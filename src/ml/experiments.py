"""
Experiment Runner
=================

Runs CNN model comparison experiments.

Supports:
- cross_validation mode
- split mode
- single-view experiments such as FDT, FFF, FLT, FLP
- classification levels such as species, genus, family
"""

from pathlib import Path
import json

import pandas as pd

from src.ml.trainer import Trainer
from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class ExperimentRunner:
    """Run model comparison experiments."""

    def __init__(self) -> None:
        self.cfg = CONFIG.project()

        self.models = self.cfg["training"]["models"]
        self.mode = self.cfg["experiment"]["mode"]

        self.view_code = self.cfg["experiment"].get("view_code", "ALL")
        self.classification_level = self.cfg["training"].get(
            "classification_level",
            "species",
        )

        self.models_dir = Path("models")
        self.reports_dir = Path("reports")

        self.comparison_csv = (
            self.reports_dir
            / self.mode
            / self.classification_level
            / self.view_code
            / "model_comparison.csv"
        )

        self.comparison_json = (
            self.reports_dir
            / self.mode
            / self.classification_level
            / self.view_code
            / "model_comparison.json"
        )

        self.results: list[dict] = []

    def discover_cross_validation_folds(self) -> list[tuple[str, Path, Path]]:
        """Find CV fold train/test files."""
        cv_dir = Path("metadata/cv_folds") / self.view_code

        if not cv_dir.exists():
            raise FileNotFoundError(
                "metadata/cv_folds not found. Run cross_validation first."
            )

        folds = []

        for train_file in sorted(cv_dir.glob("fold_*_train.csv")):
            fold_number = train_file.stem.split("_")[1]
            test_file = cv_dir / f"fold_{fold_number}_test.csv"

            if not test_file.exists():
                raise FileNotFoundError(f"Missing test file: {test_file}")

            folds.append((f"fold_{fold_number}", train_file, test_file))

        if not folds:
            raise FileNotFoundError("No cross-validation folds found.")

        return folds

    def discover_fixed_split(self) -> list[tuple[str, Path, Path]]:
        """Use fixed train/test split."""
        train_csv = Path("metadata/train.csv")
        test_csv = Path("metadata/test.csv")

        if not train_csv.exists():
            raise FileNotFoundError("metadata/train.csv not found.")

        if not test_csv.exists():
            raise FileNotFoundError("metadata/test.csv not found.")

        return [("split", train_csv, test_csv)]

    def get_experiments(self) -> list[tuple[str, Path, Path]]:
        """Choose experiment mode."""
        if self.mode == "cross_validation":
            logger.info("Experiment mode: cross_validation")
            return self.discover_cross_validation_folds()

        if self.mode == "split":
            logger.info("Experiment mode: split")
            return self.discover_fixed_split()

        raise ValueError(
            f"Invalid experiment mode: {self.mode}. "
            "Use 'cross_validation' or 'split'."
        )

    def summarize_history(
        self,
        model_name: str,
        experiment_name: str,
        history: dict,
    ) -> dict:
        """Summarize trainer history."""
        best_epoch = max(
            history["epochs"],
            key=lambda epoch: epoch["f1_weighted"],
        )

        return {
            "model": model_name,
            "classification_level": self.classification_level,
            "view_code": self.view_code,
            "experiment": experiment_name,
            "best_epoch": best_epoch["epoch"],
            "best_accuracy": history["best_accuracy"],
            "best_f1": history["best_f1"],
            "best_precision_weighted": best_epoch["precision_weighted"],
            "best_recall_weighted": best_epoch["recall_weighted"],
            "best_loss": best_epoch["loss"],
            "training_time_seconds": history["training_time_seconds"],
            "train_csv": history["train_csv"],
            "test_csv": history["test_csv"],
            "classes": ",".join(history["classes"]),
            "mode": self.mode,
        }

    def run_experiments(self) -> None:
        """Run all configured models."""
        experiments = self.get_experiments()

        for model_name in self.models:
            logger.info("=" * 60)
            logger.info("Starting model: %s", model_name)
            logger.info("=" * 60)

            for experiment_name, train_csv, test_csv in experiments:
                logger.info(
                    "Training %s | %s | %s | %s",
                    model_name,
                    self.classification_level,
                    self.view_code,
                    experiment_name,
                )

                output_dir = (
                    self.models_dir
                    / self.mode
                    / self.classification_level
                    / self.view_code
                    / model_name
                    / experiment_name
                )

                trainer = Trainer(
                    model_name=model_name,
                    train_csv=str(train_csv),
                    test_csv=str(test_csv),
                    output_dir=str(output_dir),
                )

                history = trainer.run()

                result = self.summarize_history(
                    model_name=model_name,
                    experiment_name=experiment_name,
                    history=history,
                )

                self.results.append(result)
                self.save_results()

    def save_results(self) -> None:
        """Save comparison results."""
        self.comparison_csv.parent.mkdir(parents=True, exist_ok=True)

        comparison = pd.DataFrame(self.results)
        comparison.to_csv(self.comparison_csv, index=False)

        with open(self.comparison_json, "w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=4, ensure_ascii=False)

        logger.info("Saved: %s", self.comparison_csv)
        logger.info("Saved: %s", self.comparison_json)

    def run(self) -> None:
        """Run experiment pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Experiments")
        logger.info("Mode: %s", self.mode)
        logger.info("Classification level: %s", self.classification_level)
        logger.info("View code: %s", self.view_code)
        logger.info("=" * 60)

        self.run_experiments()

        logger.info("Experiments completed.")


if __name__ == "__main__":
    setup_logging()

    runner = ExperimentRunner()
    runner.run()
