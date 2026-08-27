"""
Image transformations for the Brachycera classification pipeline.

This module provides separate transformation pipelines for:

- training
- validation
- prediction

Training transformations include conservative augmentation.
Validation and prediction transformations are deterministic.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import transforms


DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# ImageNet normalization values.
# These are appropriate for pretrained torchvision models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_training_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Return transformations used during model training.

    Conservative augmentation is used because strong transformations
    could distort important morphological characteristics.

    Parameters
    ----------
    image_size:
        Final image size as ``(height, width)``.

    Returns
    -------
    torchvision.transforms.Compose
        Training transformation pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize(image_size),

            # Small geometric variation
            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(
                degrees=10,
                fill=255,
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=255,
            ),

            # Small lighting variation
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),

            # Randomly hides a small region after tensor conversion
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.08),
                ratio=(0.5, 2.0),
                value="random",
            ),
        ]
    )


def get_validation_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Return deterministic transformations for validation images.

    No random augmentation is applied because validation results must
    remain reproducible.

    Parameters
    ----------
    image_size:
        Final image size as ``(height, width)``.

    Returns
    -------
    torchvision.transforms.Compose
        Validation transformation pipeline.
    """

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_prediction_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Return transformations for model prediction.

    Prediction transformations are identical to validation
    transformations.

    Parameters
    ----------
    image_size:
        Final image size as ``(height, width)``.

    Returns
    -------
    torchvision.transforms.Compose
        Prediction transformation pipeline.
    """

    return get_validation_transforms(image_size=image_size)


def denormalize_tensor(tensor):
    """
    Reverse ImageNet normalization.

    This function is useful for visualising transformed images.

    Parameters
    ----------
    tensor:
        Normalized PyTorch image tensor with shape ``[C, H, W]``.

    Returns
    -------
    torch.Tensor
        Denormalized tensor with values limited to the range [0, 1].
    """

    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)

    return (tensor * std + mean).clamp(0, 1)


if __name__ == "__main__":
    print("=" * 55)
    print("BRACHYCERA IMAGE TRANSFORMS")
    print("=" * 55)

    print("\nTraining transforms:")
    print(get_training_transforms())

    print("\nValidation transforms:")
    print(get_validation_transforms())

    print("\nPrediction transforms:")
    print(get_prediction_transforms())
