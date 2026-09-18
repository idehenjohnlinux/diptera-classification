"""
Metrics
=======
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(y_true, y_pred) -> dict:
    """Compute standard classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def count_trainable_parameters(model) -> int:
    """Count trainable parameters."""
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def count_total_parameters(model) -> int:
    """Count all parameters."""
    return int(
        sum(parameter.numel() for parameter in model.parameters())
    )


def predictions_from_logits(logits):
    """Convert logits to predicted class indices."""
    return np.argmax(logits, axis=1)
