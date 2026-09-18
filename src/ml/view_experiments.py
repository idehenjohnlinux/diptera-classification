"""
View Experiment Runner
======================

Runs the full CNN experiment pipeline for multiple image views.

For each view:
1. Create cross-validation folds
2. Train all configured CNN models
3. Evaluate all trained models

Outputs are separated by view:
- metadata/cv_folds/<VIEW>/
- models/cross_validation/species/<VIEW>/
- reports/cross_validation/species/<VIEW>/
- reports/evaluation/cross_validation/species/<VIEW>/
"""

import copy

from src.utils.config import CONFIG
from src.utils.logger import setup_logging, get_logger

from src.core.cross_validation import CrossValidationSplitter
from src.ml.experiments import ExperimentRunner
from src.ml.evaluator import ModelEvaluator


logger = get_logger(__name__)


class ViewExperimentRunner:
    """Run experiments for all configured anatomical views."""

    def __init__(self) -> None:
        self.cfg = CONFIG.project()
        self.views = self.cfg["experiment"].get("views", [])

        if not self.views:
            raise ValueError(
                "No views found in config.yaml. Add experiment.views."
            )

    def set_active_view(self, view_code: str) -> None:
        """Update active view in global config."""
        CONFIG.config["experiment"]["view_code"] = view_code

    def run_one_view(self, view_code: str) -> None:
        """Run full pipeline for one view."""
        logger.info("=" * 80)
        logger.info("STARTING VIEW EXPERIMENT: %s", view_code)
        logger.info("=" * 80)

        self.set_active_view(view_code)

        logger.info("Creating cross-validation folds for %s", view_code)
        splitter = CrossValidationSplitter()
        splitter.run()

        logger.info("Training CNN models for %s", view_code)
        experiments = ExperimentRunner()
        experiments.run()

        logger.info("Evaluating CNN models for %s", view_code)
        evaluator = ModelEvaluator()
        evaluator.run()

        logger.info("COMPLETED VIEW EXPERIMENT: %s", view_code)

    def run(self) -> None:
        """Run all configured view experiments."""
        logger.info("=" * 80)
        logger.info("RUNNING ALL VIEW EXPERIMENTS")
        logger.info("Views: %s", self.views)
        logger.info("=" * 80)

        for view_code in self.views:
            self.run_one_view(view_code)

        logger.info("ALL VIEW EXPERIMENTS COMPLETED")


if __name__ == "__main__":
    setup_logging()

    runner = ViewExperimentRunner()
    runner.run()
