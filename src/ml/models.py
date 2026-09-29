"""
CNN model factory for Brachycera image classification.

"""
# import modules
from __future__ import annotations

import argparse
from typing import Final

import torch
from torch import nn
from torchvision import models

# CNN architectures evaluated in this project.
SUPPORTED_ARCHITECTURES: Final[tuple[str, ...]] = (
    "resnet18",
    "efficientnet_b0",
    "mobilenet_v3_large",
)


def normalize_architecture_name(architecture: str) -> str:
    """
    Normalize common architecture-name variations.
    This allows users to provide names like the architectures

    """

    normalized = architecture.strip().lower().replace("-", "_")

    aliases = {
        "resnet_18": "resnet18",
        "efficientnetb0": "efficientnet_b0",
        "mobilenetv3large": "mobilenet_v3_large",
        "mobilenet_v3large": "mobilenet_v3_large",
    }

    return aliases.get(normalized, normalized)


def create_resnet18(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a ResNet18 model with a custom output layer.
    """

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)
    # Number of features produced by the original ResNet18 backbone.
    input_features = model.fc.in_features
    # Replace the original ImageNet classification layer.
    # The new layer predicts the family or genus classes
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def create_efficientnet_b0(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create an EfficientNet-B0 model with a custom output layer.
    """

    weights = (
        models.EfficientNet_B0_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.efficientnet_b0(weights=weights)
    # EfficientNet-B0 stores its final linear classifier
    # at position 1 of model.classifier.
    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def create_mobilenet_v3_large(
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Create a MobileNetV3-Large model with a custom output layer.
    """

    weights = (
        models.MobileNet_V3_Large_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.mobilenet_v3_large(weights=weights)
    # The final linear layer of MobileNetV3 Large is stored
    # at position 3 of the classifier sequence.
    input_features = model.classifier[3].in_features
    # Adapt the output layer to the number of taxonomic classes.
    model.classifier[3] = nn.Linear(
        in_features=input_features,
        out_features=num_classes,
    )

    return model


def freeze_feature_extractor(model: nn.Module) -> None:
    """
    Freeze all model parameters except the classification head.

    This can be useful during the first stage of transfer learning.
    """
    # Freeze every parameter in the pretrained network.
    for parameter in model.parameters():
        parameter.requires_grad = False
    if isinstance(model, models.ResNet):
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    elif isinstance(model, models.EfficientNet):
        for parameter in model.classifier[1].parameters():
            parameter.requires_grad = True
    elif isinstance(model, models.MobileNetV3):
        for parameter in model.classifier[3].parameters():
            parameter.requires_grad = True

    else:
        raise AttributeError(
            "The model does not contain a recognized classification head."
        )


def unfreeze_all_layers(model: nn.Module) -> None:
    """
    Make all model parameters trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def create_model(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Create one of the supported CNN architectures.
    
        Model architecture name.
        Number of output classes.
        Whether to use ImageNet pretrained weights.
        Whether to freeze the feature extractor and train only the
        classification head initially.

    """

    if num_classes < 2:
        raise ValueError(
            "num_classes must be at least 2."
        )

    architecture = normalize_architecture_name(architecture)

    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)

        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported architectures: {supported}"
        )

    if architecture == "resnet18":
        model = create_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    elif architecture == "efficientnet_b0":
        model = create_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    else:
        model = create_mobilenet_v3_large(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    if freeze_backbone:
        freeze_feature_extractor(model)

    return model


def test_forward_pass(
    model: nn.Module,
    image_size: int = 224,
    batch_size: int = 2,
) -> tuple[int, ...]:
    """
    Test the model using a dummy image batch.
    """

    model.eval()

    dummy_batch = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
    )

    with torch.no_grad():
        output = model(dummy_batch)

    return tuple(output.shape)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Create and test Brachycera CNN models."
    )

    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="resnet18",
        help="CNN architecture to test.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Number of classification outputs.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image height and width.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Dummy batch size used for the forward-pass test.",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Create the model without pretrained weights.",
    )

    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the feature extractor.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all supported architectures.",
    )

    return parser


def print_model_summary(
    architecture: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
    image_size: int,
    batch_size: int,
) -> None:
    """
    Create one model and print a compact validation summary.
    """

    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    total_parameters, trainable_parameters = count_parameters(model)

    output_shape = test_forward_pass(
        model=model,
        image_size=image_size,
        batch_size=batch_size,
    )

    print("-" * 60)
    print(f"Architecture          : {architecture}")
    print(f"Number of classes     : {num_classes}")
    print(f"Pretrained            : {pretrained}")
    print(f"Backbone frozen       : {freeze_backbone}")
    print(f"Total parameters      : {total_parameters:,}")
    print(f"Trainable parameters  : {trainable_parameters:,}")
    print(f"Output shape          : {output_shape}")
    print("Status                : valid")


def main() -> None:
    """
    Command-line entry point.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("BRACHYCERA CNN MODEL FACTORY")
    print("=" * 60)

    architectures = (
        SUPPORTED_ARCHITECTURES
        if args.all
        else (args.architecture,)
    )

    for architecture in architectures:
        print_model_summary(
            architecture=architecture,
            num_classes=args.num_classes,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )

    print("-" * 60)
    print("All requested models were created successfully.")


if __name__ == "__main__":
    main()
