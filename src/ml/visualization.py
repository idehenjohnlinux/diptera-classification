"""
Visualization Module
====================

Generates plots for model comparison and evaluation results.

Inputs:
- reports/model_comparison.csv
- reports/evaluation/evaluation_summary.csv

Outputs:
- reports/figures/
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logger import setup_logging, get_logger


logger = get_logger(__name__)


class ResultVisualizer:
    """Create plots from experiment and evaluation outputs."""

    def __init__(self) -> None:
        self.model_comparison_csv = Path("reports/model_comparison.csv")
        self.evaluation_summary_csv = Path("reports/evaluation/evaluation_summary.csv")

        self.output_dir = Path("reports/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load CSV safely."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return pd.read_csv(path)

    def plot_metric_by_model(
        self,
        data: pd.DataFrame,
        metric: str,
        title: str,
        output_name: str,
    ) -> None:
        """Plot mean metric by model."""
        summary = (
            data.groupby("model")[metric]
            .mean()
            .reset_index()
            .sort_values(metric, ascending=False)
        )

        plt.figure(figsize=(8, 5))
        plt.bar(summary["model"], summary[metric])
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info("Saved figure: %s", output_path)

    def plot_model_comparison(self) -> None:
        """Plot comparison metrics from model_comparison.csv."""
        data = self.load_csv(self.model_comparison_csv)

        metric_map = {
            "best_accuracy": "model_accuracy_comparison.png",
            "best_f1": "model_f1_comparison.png",
            "best_precision_weighted": "model_precision_comparison.png",
            "best_recall_weighted": "model_recall_comparison.png",
            "training_time_seconds": "model_training_time_comparison.png",
        }

        for metric, output_name in metric_map.items():
            if metric in data.columns:
                self.plot_metric_by_model(
                    data=data,
                    metric=metric,
                    title=f"Mean {metric} by Model",
                    output_name=output_name,
                )

    def plot_evaluation_summary(self) -> None:
        """Plot metrics from evaluation_summary.csv."""
        if not self.evaluation_summary_csv.exists():
            logger.warning(
                "Evaluation summary not found, skipping: %s",
                self.evaluation_summary_csv,
            )
            return

        data = self.load_csv(self.evaluation_summary_csv)

        metric_map = {
            "accuracy": "evaluation_accuracy_comparison.png",
            "f1_weighted": "evaluation_f1_comparison.png",
            "precision_weighted": "evaluation_precision_comparison.png",
            "recall_weighted": "evaluation_recall_comparison.png",
        }

        for metric, output_name in metric_map.items():
            if metric in data.columns:
                self.plot_metric_by_model(
                    data=data,
                    metric=metric,
                    title=f"Evaluation {metric} by Model",
                    output_name=output_name,
                )

    def save_summary_tables(self) -> None:
        """Save aggregated summary tables."""
        data = self.load_csv(self.model_comparison_csv)

        metrics = [
            column
            for column in [
                "best_accuracy",
                "best_f1",
                "best_precision_weighted",
                "best_recall_weighted",
                "training_time_seconds",
            ]
            if column in data.columns
        ]

        summary = (
            data.groupby("model")[metrics]
            .agg(["mean", "std"])
        )

        output_path = self.output_dir / "model_summary_table.csv"
        summary.to_csv(output_path)

        logger.info("Saved summary table: %s", output_path)

    def run(self) -> None:
        """Run all visualizations."""
        logger.info("=" * 60)
        logger.info("Starting Visualization")
        logger.info("=" * 60)

        self.plot_model_comparison()
        self.plot_evaluation_summary()
        self.save_summary_tables()

        logger.info("Visualization completed successfully.")


if __name__ == "__main__":
    setup_logging()

    visualizer = ResultVisualizer()
