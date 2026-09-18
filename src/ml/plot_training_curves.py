"""
Plot Training Curves
====================

Reads metrics.json files from model training folders and creates:
- loss_curve.png
- accuracy_f1_curve.png
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt


def plot_curves(metrics_file: Path) -> None:
    with open(metrics_file, "r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = [item["epoch"] for item in history["epochs"]]
    train_loss = [item["train_loss"] for item in history["epochs"]]
    test_loss = [item["loss"] for item in history["epochs"]]
    accuracy = [item["accuracy"] for item in history["epochs"]]
    f1 = [item["f1_weighted"] for item in history["epochs"]]

    output_dir = metrics_file.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, test_loss, marker="o", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracy, marker="o", label="Accuracy")
    plt.plot(epochs, f1, marker="o", label="Weighted F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_f1_curve.png", dpi=300)
    plt.close()

    print(f"Saved figures to: {output_dir}")


def main() -> None:
    metrics_files = sorted(Path("models").rglob("metrics.json"))

    if not metrics_files:
        raise FileNotFoundError("No metrics.json files found under models/")

    for metrics_file in metrics_files:
        plot_curves(metrics_file)


if __name__ == "__main__":
    main()
