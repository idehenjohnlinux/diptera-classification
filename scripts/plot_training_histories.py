from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DIR = PROJECT_ROOT / "results" / "training"

OUTPUT_DIR = PROJECT_ROOT / "results" / "training_history_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "mobilenet_v3_large": "MobileNetV3 Large",
    "mobilenetv3_large": "MobileNetV3 Large",
}

REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "validation_loss",
    "train_accuracy",
    "validation_accuracy",
}


# ============================================================
# UTILITIES
# ============================================================

def display_architecture_name(folder_name: str) -> str:
    """Return a dissertation-friendly architecture name."""

    return ARCHITECTURE_NAMES.get(
        folder_name,
        folder_name.replace("_", " ").title(),
    )


def find_history_files() -> list[Path]:
    """Find every history.csv file under results/training."""

    return sorted(
        TRAINING_DIR.glob(
            "*/*/*/fold_*/history.csv"
        )
    )


def parse_history_path(
    history_file: Path,
) -> dict[str, object]:
    """
    Parse metadata from:

    results/training/
        architecture/
            taxonomic_level/
                view/
                    fold_X/
                        history.csv
    """

    relative_path = history_file.relative_to(TRAINING_DIR)

    architecture_folder = relative_path.parts[0]
    taxonomic_level = relative_path.parts[1]
    view_code = relative_path.parts[2]
    fold_folder = relative_path.parts[3]

    fold_number = int(
        fold_folder.replace("fold_", "")
    )

    return {
        "architecture_folder": architecture_folder,
        "architecture": display_architecture_name(
            architecture_folder
        ),
        "taxonomic_level": taxonomic_level,
        "view_code": view_code,
        "fold": fold_number,
    }


def load_all_histories(
    history_files: list[Path],
) -> pd.DataFrame:
    """Load and combine all fold training histories."""

    loaded_histories: list[pd.DataFrame] = []

    for history_file in history_files:
        metadata = parse_history_path(history_file)

        dataframe = pd.read_csv(history_file)

        missing_columns = (
            REQUIRED_COLUMNS - set(dataframe.columns)
        )

        if missing_columns:
            print(
                f"Skipping {history_file}: missing columns "
                f"{sorted(missing_columns)}"
            )
            continue

        dataframe = dataframe.copy()

        dataframe["epoch"] = pd.to_numeric(
            dataframe["epoch"],
            errors="coerce",
        )

        dataframe = dataframe.dropna(
            subset=["epoch"]
        )

        dataframe["epoch"] = dataframe[
            "epoch"
        ].astype(int)

        numeric_columns = [
            column
            for column in dataframe.columns
            if column != "epoch"
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        dataframe["architecture_folder"] = metadata[
            "architecture_folder"
        ]
        dataframe["architecture"] = metadata[
            "architecture"
        ]
        dataframe["taxonomic_level"] = metadata[
            "taxonomic_level"
        ]
        dataframe["view_code"] = metadata[
            "view_code"
        ]
        dataframe["fold"] = metadata["fold"]
        dataframe["history_file"] = str(
            history_file
        )

        loaded_histories.append(dataframe)

        print(
            f"Loaded: {metadata['architecture']} | "
            f"{metadata['taxonomic_level']} | "
            f"{metadata['view_code']} | "
            f"fold {metadata['fold']}"
        )

    if not loaded_histories:
        raise RuntimeError(
            "No valid history.csv files were loaded."
        )

    return pd.concat(
        loaded_histories,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_epoch_statistics(
    experiment: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate mean and standard deviation per epoch."""

    available_columns = [
        column
        for column in columns
        if column in experiment.columns
        and column != "epoch"
    ]

    if not available_columns:
        raise ValueError(
            "No requested metric columns were found."
        )

    mean_values = (
        experiment
        .groupby("epoch", as_index=False)[
            available_columns
        ]
        .mean()
        .sort_values("epoch")
    )

    standard_deviations = (
        experiment
        .groupby("epoch", as_index=False)[
            available_columns
        ]
        .std()
        .fillna(0)
        .sort_values("epoch")
    )

    return mean_values, standard_deviations


# ============================================================
# PLOTTING
# ============================================================

def plot_mean_training_validation(
    experiment: pd.DataFrame,
    train_column: str,
    validation_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    """Plot mean training and validation curves across five folds."""

    if train_column not in experiment.columns:
        print(
            f"Missing {train_column}; skipping {output_path.name}"
        )
        return

    if validation_column not in experiment.columns:
        print(
            f"Missing {validation_column}; "
            f"skipping {output_path.name}"
        )
        return

    mean_values, standard_deviations = (
        calculate_epoch_statistics(
            experiment=experiment,
            columns=[
                train_column,
                validation_column,
            ],
        )
    )

    epochs = mean_values[
        "epoch"
    ].to_numpy(dtype=int)

    train_mean = mean_values[
        train_column
    ].to_numpy(dtype=float)

    train_std = standard_deviations[
        train_column
    ].to_numpy(dtype=float)

    validation_mean = mean_values[
        validation_column
    ].to_numpy(dtype=float)

    validation_std = standard_deviations[
        validation_column
    ].to_numpy(dtype=float)

    if percentage:
        train_mean = train_mean * 100
        train_std = train_std * 100
        validation_mean = validation_mean * 100
        validation_std = validation_std * 100

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        epochs,
        train_mean,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Training mean",
    )

    axis.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    axis.plot(
        epochs,
        validation_mean,
        marker="o",
        markersize=3,
        linewidth=2,
        label="Validation mean",
    )

    axis.fill_between(
        epochs,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.2,
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    axis.grid(
        linestyle="--",
        alpha=0.3,
    )

    axis.legend()

    if percentage:
        axis.set_ylim(0, 100)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_validation_by_fold(
    experiment: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    """Plot one validation curve for each fold."""

    if metric_column not in experiment.columns:
        return

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    for fold, fold_dataframe in experiment.groupby(
        "fold"
    ):
        fold_dataframe = fold_dataframe.sort_values(
            "epoch"
        )

        values = fold_dataframe[
            metric_column
        ].to_numpy(dtype=float)

        if percentage:
            values = values * 100

        axis.plot(
            fold_dataframe["epoch"],
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"Fold {int(fold)}",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)

    axis.grid(
        linestyle="--",
        alpha=0.3,
    )

    axis.legend(
        title="Validation fold",
        ncol=2,
    )

    if percentage:
        axis.set_ylim(0, 100)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# SUMMARY TABLES
# ============================================================

def create_best_epoch_table(
    histories: pd.DataFrame,
) -> pd.DataFrame:
    """Select the best epoch in each fold."""

    if "validation_f1_macro" in histories.columns:
        selection_metric = "validation_f1_macro"
    else:
        selection_metric = "validation_accuracy"

    valid_histories = histories.dropna(
        subset=[selection_metric]
    ).copy()

    group_columns = [
        "architecture_folder",
        "architecture",
        "taxonomic_level",
        "view_code",
        "fold",
    ]

    best_indices = (
        valid_histories
        .groupby(group_columns)[selection_metric]
        .idxmax()
    )

    best_epochs = valid_histories.loc[
        best_indices
    ].copy()

    preferred_columns = [
        "architecture",
        "taxonomic_level",
        "view_code",
        "fold",
        "epoch",
        "train_loss",
        "validation_loss",
        "train_accuracy",
        "validation_accuracy",
        "train_precision_macro",
        "validation_precision_macro",
        "train_recall_macro",
        "validation_recall_macro",
        "train_f1_macro",
        "validation_f1_macro",
        "learning_rate",
        "epoch_duration_seconds",
        "history_file",
    ]

    existing_columns = [
        column
        for column in preferred_columns
        if column in best_epochs.columns
    ]

    return (
        best_epochs[existing_columns]
        .sort_values(
            [
                "architecture",
                "taxonomic_level",
                "view_code",
                "fold",
            ]
        )
        .reset_index(drop=True)
    )


def create_experiment_summary(
    best_epochs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the best epochs across five folds."""

    group_columns = [
        "architecture",
        "taxonomic_level",
        "view_code",
    ]

    metric_columns = [
        column
        for column in (
            "validation_loss",
            "validation_accuracy",
            "validation_precision_macro",
            "validation_recall_macro",
            "validation_f1_macro",
        )
        if column in best_epochs.columns
    ]

    summary = (
        best_epochs
        .groupby(group_columns)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        (
            "_".join(
                str(value)
                for value in column
                if str(value)
            )
            if isinstance(column, tuple)
            else column
        )
        for column in summary.columns
    ]

    fold_counts = (
        best_epochs
        .groupby(group_columns)["fold"]
        .nunique()
        .rename("number_of_folds")
        .reset_index()
    )

    return summary.merge(
        fold_counts,
        on=group_columns,
        how="left",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Generate all training-history figures."""

    if not TRAINING_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: "
            f"{TRAINING_DIR}"
        )

    history_files = find_history_files()

    if not history_files:
        raise FileNotFoundError(
            "No history.csv files were found using:\n"
            "results/training/"
            "architecture/level/view/fold_X/history.csv"
        )

    print(
        f"\nFound {len(history_files)} history files.\n"
    )

    histories = load_all_histories(
        history_files
    )

    grouping_columns = [
        "architecture_folder",
        "architecture",
        "taxonomic_level",
        "view_code",
    ]

    number_of_experiments = 0

    for keys, experiment in histories.groupby(
        grouping_columns
    ):
        (
            architecture_folder,
            architecture,
            taxonomic_level,
            view_code,
        ) = keys

        folds = sorted(
            experiment["fold"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        print("\n" + "=" * 70)
        print(
            f"{architecture} | "
            f"{taxonomic_level.upper()} | "
            f"{view_code}"
        )
        print(f"Folds found: {folds}")
        print("=" * 70)

        experiment_output = (
            OUTPUT_DIR
            / architecture_folder
            / taxonomic_level
            / view_code
        )

        experiment_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        title_suffix = (
            f"{architecture} – "
            f"{taxonomic_level.capitalize()} – "
            f"{view_code}"
        )

        plot_mean_training_validation(
            experiment=experiment,
            train_column="train_loss",
            validation_column="validation_loss",
            ylabel="Loss",
            title=(
                "Mean training and validation loss\n"
                f"{title_suffix}"
            ),
            output_path=(
                experiment_output
                / "mean_training_validation_loss.png"
            ),
        )

        plot_mean_training_validation(
            experiment=experiment,
            train_column="train_accuracy",
            validation_column="validation_accuracy",
            ylabel="Accuracy (%)",
            title=(
                "Mean training and validation accuracy\n"
                f"{title_suffix}"
            ),
            output_path=(
                experiment_output
                / "mean_training_validation_accuracy.png"
            ),
            percentage=True,
        )

        if (
            "train_f1_macro" in experiment.columns
            and "validation_f1_macro"
            in experiment.columns
        ):
            plot_mean_training_validation(
                experiment=experiment,
                train_column="train_f1_macro",
                validation_column=(
                    "validation_f1_macro"
                ),
                ylabel="Macro F1-score (%)",
                title=(
                    "Mean training and validation "
                    "Macro F1-score\n"
                    f"{title_suffix}"
                ),
                output_path=(
                    experiment_output
                    / "mean_training_validation_macro_f1.png"
                ),
                percentage=True,
            )

        plot_validation_by_fold(
            experiment=experiment,
            metric_column="validation_loss",
            ylabel="Validation loss",
            title=(
                "Validation loss by fold\n"
                f"{title_suffix}"
            ),
            output_path=(
                experiment_output
                / "validation_loss_by_fold.png"
            ),
        )

        plot_validation_by_fold(
            experiment=experiment,
            metric_column="validation_accuracy",
            ylabel="Validation accuracy (%)",
            title=(
                "Validation accuracy by fold\n"
                f"{title_suffix}"
            ),
            output_path=(
                experiment_output
                / "validation_accuracy_by_fold.png"
            ),
            percentage=True,
        )

        if "validation_f1_macro" in experiment.columns:
            plot_validation_by_fold(
                experiment=experiment,
                metric_column="validation_f1_macro",
                ylabel="Validation Macro F1-score (%)",
                title=(
                    "Validation Macro F1-score by fold\n"
                    f"{title_suffix}"
                ),
                output_path=(
                    experiment_output
                    / "validation_macro_f1_by_fold.png"
                ),
                percentage=True,
            )

        experiment.to_csv(
            experiment_output
            / "combined_fold_history.csv",
            index=False,
        )

        number_of_experiments += 1

    best_epochs = create_best_epoch_table(
        histories
    )

    best_epochs.to_csv(
        OUTPUT_DIR / "best_epoch_per_fold.csv",
        index=False,
    )

    experiment_summary = (
        create_experiment_summary(best_epochs)
    )

    experiment_summary.to_csv(
        OUTPUT_DIR
        / "training_experiment_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TRAINING HISTORY ANALYSIS COMPLETED")
    print("=" * 70)
    print(
        f"History files loaded: "
        f"{len(history_files)}"
    )
    print(
        f"Experiments analyzed: "
        f"{number_of_experiments}"
    )
    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
