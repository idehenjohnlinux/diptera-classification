"""
Loss Functions
==============
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


def compute_class_weights(
    csv_file: str | Path,
    classes: list[str],
    label_column: str = "label",
) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    data = pd.read_csv(csv_file)

    counts = data[label_column].value_counts().to_dict()
    total = len(data)
    num_classes = len(classes)

    weights = []

    for class_name in classes:
        count = counts.get(class_name, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def build_loss_function(
    csv_file: str | Path,
    classes: list[str],
    use_class_weights: bool = True,
    label_column: str = "label",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Build CrossEntropyLoss, optionally with class weights."""
    if use_class_weights:
        weights = compute_class_weights(
            csv_file=csv_file,
            classes=classes,
            label_column=label_column,
        ).to(device)

        return nn.CrossEntropyLoss(weight=weights)

    return nn.CrossEntropyLoss()
