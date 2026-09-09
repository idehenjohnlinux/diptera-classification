from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "scripts":
    PROJECT_ROOT = PROJECT_ROOT.parent

TRAINING_DIR = PROJECT_ROOT / "results" / "training"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "architecture_learning_figures_clean"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARCHITECTURES = [
    "efficientnet_b0",
    "resnet18",
    "mobilenet_v3_large",
]

ARCHITECTURE_NAMES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet18": "ResNet18",
    "mobilenet_v3_large": "MobileNetV3 Large",
}

LEVEL_NAMES = {
    "family": "Família",
    "genus": "Género",
}

VIEWS = [
    "FLT",
    "FLP",
    "FFF",
    "FDT",
]

LEVELS = [
    "family",
    "genus",
]

REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "validation_loss",
    "train_accuracy",
    "validation_accuracy",
}


# ============================================================
# FONT CONFIGURATION
# ============================================================

MAIN_TITLE_SIZE = 19
SUBPLOT_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 15
NOTE_SIZE = 15


# ============================================================
# DATA LOADING
# ============================================================

def load_architecture_history(
    architecture: str,
    taxonomic_level: str,
) -> pd.DataFrame:
    """
    Load the training histories for all anatomical views
    and all cross-validation folds of one architecture.
    """

    frames: list[pd.DataFrame] = []

    for view_code in VIEWS:
        view_dir = (
            TRAINING_DIR
            / architecture
            / taxonomic_level
            / view_code
        )

        history_files = sorted(
            view_dir.glob("fold_*/history.csv")
        )

        for history_file in history_files:
            frame = pd.read_csv(history_file)

            missing_columns = (
                REQUIRED_COLUMNS
                - set(frame.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"{history_file} is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            frame = frame.copy()

            frame["epoch"] = pd.to_numeric(
                frame["epoch"],
                errors="coerce",
            )

            frame = frame.dropna(
                subset=["epoch"],
            )

            frame["epoch"] = (
                frame["epoch"]
                .astype(int)
            )

            metric_columns = (
                REQUIRED_COLUMNS
                - {"epoch"}
            )

            for column in metric_columns:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

            frame["view_code"] = view_code
            frame["fold"] = history_file.parent.name

            frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            "No training histories were found for "
            f"{architecture}/{taxonomic_level}"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def mean_curve(
    histories: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Calculate one mean learning curve per architecture.

    First, the five folds are averaged within each anatomical
    view. Then, the four anatomical-view curves are averaged.
    """

    view_means = (
        histories
        .groupby(
            [
                "view_code",
                "epoch",
            ],
            as_index=False,
        )[columns]
        .mean()
    )

    overall_mean = (
        view_means
        .groupby(
            "epoch",
            as_index=False,
        )[columns]
        .mean()
        .sort_values("epoch")
    )

    return overall_mean


# ============================================================
# COMMON AXIS FORMATTING
# ============================================================

def format_axis(
    axis: plt.Axes,
) -> None:
    """Apply common formatting to a subplot."""

    axis.tick_params(
        axis="both",
        labelsize=TICK_LABEL_SIZE,
    )

    axis.grid(
        axis="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.25,
    )


# ============================================================
# FIGURE 1: ACCURACY ONLY
# ============================================================

def create_accuracy_figure(
    taxonomic_level: str,
) -> None:
    """
    Create three side-by-side accuracy panels,
    one for each CNN architecture.
    """

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for index, architecture in enumerate(
        ARCHITECTURES
    ):
        histories = load_architecture_history(
            architecture,
            taxonomic_level,
        )

        curve = mean_curve(
            histories,
            [
                "train_accuracy",
                "validation_accuracy",
            ],
        )

        epochs = curve["epoch"]

        axes[index].plot(
            epochs,
            curve["train_accuracy"] * 100,
            linewidth=2.6,
            label="Treino",
        )

        axes[index].plot(
            epochs,
            curve["validation_accuracy"] * 100,
            linewidth=2.6,
            label="Validação",
        )

        axes[index].set_title(
            ARCHITECTURE_NAMES[architecture],
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
            pad=10,
        )

        axes[index].set_xlabel(
            "Época",
            fontsize=AXIS_LABEL_SIZE,
            fontweight="bold",
        )

        axes[index].set_ylim(
            0,
            100,
        )

        format_axis(
            axes[index],
        )

        if index == 0:
            axes[index].set_ylabel(
                "Accuracy (%)",
                fontsize=AXIS_LABEL_SIZE,
                fontweight="bold",
            )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.93),
    )

    figure.suptitle(
        (
            "Classificação ao nível da "
            f"{LEVEL_NAMES[taxonomic_level]}: "
            "Comportamento Geral de Aprendizagem "
            "das Arquiteturas CNN"
        ),
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.015,
        (
            "Cada linha representa a média de quatro "
            "vistas anatómicas e cinco folds por vista."
        ),
        ha="center",
        fontsize=NOTE_SIZE,
    )

    figure.tight_layout(
        rect=[
            0,
            0.08,
            1,
            0.88,
        ]
    )

    stem = (
        f"{taxonomic_level}"
        "_overall_architecture_accuracy_clean"
    )

    for extension in (
        "png",
        "svg",
        "pdf",
    ):
        output_file = (
            OUTPUT_DIR
            / f"{stem}.{extension}"
        )

        figure.savefig(
            output_file,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(figure)


# ============================================================
# FIGURE 2: ACCURACY AND LOSS
# ============================================================

def create_accuracy_loss_grid(
    taxonomic_level: str,
) -> None:
    """
    Create a 2 × 3 figure containing accuracy and loss
    curves for the three CNN architectures.
    """

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 11),
        sharex="col",
    )

    for column_index, architecture in enumerate(
        ARCHITECTURES
    ):
        histories = load_architecture_history(
            architecture,
            taxonomic_level,
        )

        curve = mean_curve(
            histories,
            [
                "train_accuracy",
                "validation_accuracy",
                "train_loss",
                "validation_loss",
            ],
        )

        epochs = curve["epoch"]

        # ====================================================
        # ACCURACY PANEL
        # ====================================================

        accuracy_axis = axes[
            0,
            column_index,
        ]

        accuracy_axis.plot(
            epochs,
            curve["train_accuracy"] * 100,
            linewidth=2.6,
            label="Treino",
        )

        accuracy_axis.plot(
            epochs,
            curve["validation_accuracy"] * 100,
            linewidth=2.6,
            label="Validação",
        )

        accuracy_axis.set_title(
            (
                f"{ARCHITECTURE_NAMES[architecture]}\n"
                "Accuracy"
            ),
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
            pad=10,
        )

        accuracy_axis.set_ylim(
            0,
            100,
        )

        format_axis(
            accuracy_axis,
        )

        if column_index == 0:
            accuracy_axis.set_ylabel(
                "Accuracy (%)",
                fontsize=AXIS_LABEL_SIZE,
                fontweight="bold",
            )

        # ====================================================
        # LOSS PANEL
        # ====================================================

        loss_axis = axes[
            1,
            column_index,
        ]

        loss_axis.plot(
            epochs,
            curve["train_loss"],
            linewidth=2.6,
            label="Treino",
        )

        loss_axis.plot(
            epochs,
            curve["validation_loss"],
            linewidth=2.6,
            label="Validação",
        )

        loss_axis.set_title(
            (
                f"{ARCHITECTURE_NAMES[architecture]}\n"
                "Perda"
            ),
            fontsize=SUBPLOT_TITLE_SIZE,
            fontweight="bold",
            pad=10,
        )

        loss_axis.set_xlabel(
            "Época",
            fontsize=AXIS_LABEL_SIZE,
            fontweight="bold",
        )

        format_axis(
            loss_axis,
        )

        if column_index == 0:
            loss_axis.set_ylabel(
                "Perda",
                fontsize=AXIS_LABEL_SIZE,
                fontweight="bold",
            )

    handles, labels = (
        axes[0, 0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, 0.94),
    )

    figure.suptitle(
        (
            "Classificação ao nível da "
            f"{LEVEL_NAMES[taxonomic_level]}: "
            "Accuracy e Perda de Treino e Validação"
        ),
        fontsize=MAIN_TITLE_SIZE,
        fontweight="bold",
        y=0.995,
    )

    figure.text(
        0.5,
        0.015,
        (
            "Cada linha representa a média de quatro "
            "vistas anatómicas e cinco folds por vista."
        ),
        ha="center",
        fontsize=NOTE_SIZE,
    )

    figure.tight_layout(
        rect=[
            0,
            0.07,
            1,
            0.91,
        ]
    )

    stem = (
        f"{taxonomic_level}"
        "_architecture_accuracy_loss_clean"
    )

    for extension in (
        "png",
        "svg",
        "pdf",
    ):
        output_file = (
            OUTPUT_DIR
            / f"{stem}.{extension}"
        )

        figure.savefig(
            output_file,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Generate the learning-curve figures."""

    if not TRAINING_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: "
            f"{TRAINING_DIR}"
        )

    for level in LEVELS:
        create_accuracy_figure(
            level,
        )

        create_accuracy_loss_grid(
            level,
        )

    print("=" * 72)
    print("FIGURAS DE APRENDIZAGEM CRIADAS")
    print("=" * 72)
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(
        "Foram utilizadas curvas médias sem áreas "
        "de desvio-padrão ou intervalos de confiança."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
