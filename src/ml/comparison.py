"""
Model and View Comparison
=========================

Combines experiment/evaluation results across views and CNN models.

Outputs:
- reports/final_comparison/all_experiment_results.csv
- reports/final_comparison/model_summary.csv
- reports/final_comparison/view_summary.csv
- reports/final_comparison/best_model_per_view.csv
- reports/final_comparison/best_view_per_model.csv
- reports/final_comparison/overall_best_combination.csv
"""

from pathlib import Path

import pandas as pd

from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class ResultComparison:
    """Compare CNN models and anatomical views."""

    def __init__(self) -> None:
        self.cfg = CONFIG.project()

        self.mode = self.cfg["experiment"]["mode"]
        self.views = self.cfg["experiment"].get("views", [])
        self.classification_level = self.cfg["training"].get(
            "classification_level",
            "species",
        )

        self.output_dir = Path("reports/final_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_model_comparison_files(self) -> list[Path]:
        """Find all model_comparison.csv files for configured views."""
        files = []

        for view in self.views:
            path = (
                Path("reports")
                / self.mode
                / self.classification_level
                / view
                / "model_comparison.csv"
            )

            if path.exists():
                files.append(path)
            else:
                logger.warning("Missing comparison file: %s", path)

        return files

    def load_all_results(self) -> pd.DataFrame:
        """Load and combine all model comparison CSVs."""
        files = self.find_model_comparison_files()

        if not files:
            raise FileNotFoundError(
                "No model_comparison.csv files found. Run experiments first."
            )

        frames = []

        for file in files:
            df = pd.read_csv(file)

            if "view_code" not in df.columns:
                view = file.parent.name
                df["view_code"] = view

            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        output_path = self.output_dir / "all_experiment_results.csv"
        combined.to_csv(output_path, index=False)

        logger.info("Saved combined results: %s", output_path)

        return combined

    def create_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize performance by CNN model."""
        summary = (
            df.groupby("model")
            .agg(
                mean_accuracy=("best_accuracy", "mean"),
                std_accuracy=("best_accuracy", "std"),
                mean_f1=("best_f1", "mean"),
                std_f1=("best_f1", "std"),
                mean_precision=("best_precision_weighted", "mean"),
                mean_recall=("best_recall_weighted", "mean"),
                mean_training_time_seconds=("training_time_seconds", "mean"),
                experiments=("model", "count"),
            )
            .reset_index()
            .sort_values("mean_f1", ascending=False)
        )

        output_path = self.output_dir / "model_summary.csv"
        summary.to_csv(output_path, index=False)

        logger.info("Saved model summary: %s", output_path)

        return summary

    def create_view_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize performance by anatomical view."""
        summary = (
            df.groupby("view_code")
            .agg(
                mean_accuracy=("best_accuracy", "mean"),
                std_accuracy=("best_accuracy", "std"),
                mean_f1=("best_f1", "mean"),
                std_f1=("best_f1", "std"),
                mean_precision=("best_precision_weighted", "mean"),
                mean_recall=("best_recall_weighted", "mean"),
                mean_training_time_seconds=("training_time_seconds", "mean"),
                experiments=("view_code", "count"),
            )
            .reset_index()
            .sort_values("mean_f1", ascending=False)
        )

        output_path = self.output_dir / "view_summary.csv"
        summary.to_csv(output_path, index=False)

        logger.info("Saved view summary: %s", output_path)

        return summary

    def create_model_view_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize performance by CNN model and view."""
        summary = (
            df.groupby(["view_code", "model"])
            .agg(
                mean_accuracy=("best_accuracy", "mean"),
                std_accuracy=("best_accuracy", "std"),
                mean_f1=("best_f1", "mean"),
                std_f1=("best_f1", "std"),
                mean_precision=("best_precision_weighted", "mean"),
                mean_recall=("best_recall_weighted", "mean"),
                mean_training_time_seconds=("training_time_seconds", "mean"),
                experiments=("model", "count"),
            )
            .reset_index()
            .sort_values("mean_f1", ascending=False)
        )

        output_path = self.output_dir / "model_view_summary.csv"
        summary.to_csv(output_path, index=False)

        logger.info("Saved model-view summary: %s", output_path)

        return summary

    def create_best_model_per_view(self, model_view: pd.DataFrame) -> pd.DataFrame:
        """Find best CNN model for each view."""
        best = (
            model_view.sort_values("mean_f1", ascending=False)
            .groupby("view_code")
            .head(1)
            .reset_index(drop=True)
        )

        output_path = self.output_dir / "best_model_per_view.csv"
        best.to_csv(output_path, index=False)

        logger.info("Saved best model per view: %s", output_path)

        return best

    def create_best_view_per_model(self, model_view: pd.DataFrame) -> pd.DataFrame:
        """Find best view for each CNN model."""
        best = (
            model_view.sort_values("mean_f1", ascending=False)
            .groupby("model")
            .head(1)
            .reset_index(drop=True)
        )

        output_path = self.output_dir / "best_view_per_model.csv"
        best.to_csv(output_path, index=False)

        logger.info("Saved best view per model: %s", output_path)

        return best

    def create_overall_best(self, model_view: pd.DataFrame) -> pd.DataFrame:
        """Find overall best CNN + view combination."""
        best = model_view.sort_values("mean_f1", ascending=False).head(1)

        output_path = self.output_dir / "overall_best_combination.csv"
        best.to_csv(output_path, index=False)

        logger.info("Saved overall best combination: %s", output_path)

        return best

    def run(self) -> None:
        """Run complete comparison."""
        logger.info("=" * 60)
        logger.info("Starting Final Comparison")
        logger.info("=" * 60)

        df = self.load_all_results()

        self.create_model_summary(df)
        self.create_view_summary(df)

        model_view = self.create_model_view_summary(df)

        self.create_best_model_per_view(model_view)
        self.create_best_view_per_model(model_view)
        self.create_overall_best(model_view)

        logger.info("Final comparison completed successfully.")


if __name__ == "__main__":
    setup_logging()

    comparison = ResultComparison()
    comparison.run()
