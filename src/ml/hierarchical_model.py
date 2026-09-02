"""Hierarchical EfficientNet model for family and genus prediction.


This creates a conditional hierarchical architecture in which genus
classification is informed by family classification.
"""
# import modules 
from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

# creates class for Hierarchical model(EfficientNet) 
class HierarchicalEfficientNet(nn.Module):
    """EfficientNet-B0 with conditional family-genus classification."""

    def __init__(
        self,
        number_of_families: int,
        number_of_genera: int,
        pretrained: bool = True,
        family_dropout: float = 0.2,
        genus_dropout: float = 0.3,
        genus_hidden_dimension: int = 512,
    ) -> None:
        super().__init__()

        if number_of_families < 2:
            raise ValueError(
                "At least two family classes are required."
            )

        if number_of_genera < 2:
            raise ValueError(
                "At least two genus classes are required."
            )

        if genus_hidden_dimension < 1:
            raise ValueError(
                "genus_hidden_dimension must be positive."
            )

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = efficientnet_b0(weights=weights)

        feature_dimension = backbone.classifier[1].in_features

        # Remove the original EfficientNet classifier.
        backbone.classifier = nn.Identity()

        self.backbone = backbone

        self.family_head = nn.Sequential(
            nn.Dropout(p=family_dropout),
            nn.Linear(
                feature_dimension,
                number_of_families,
            ),
        )

        # Genus classification uses both visual features and the
        # family probability distribution.
        genus_input_dimension = (
            feature_dimension + number_of_families
        )

        self.genus_head = nn.Sequential(
            nn.Linear(
                genus_input_dimension,
                genus_hidden_dimension,
            ),
            nn.BatchNorm1d(genus_hidden_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout(p=genus_dropout),
            nn.Linear(
                genus_hidden_dimension,
                number_of_genera,
            ),
        )

        self.number_of_families = number_of_families
        self.number_of_genera = number_of_genera
        self.feature_dimension = feature_dimension

    def forward(
        self,
        images: Tensor,
    ) -> dict[str, Tensor]:
        """Return family and conditional genus predictions."""

        features = self.backbone(images)

        family_logits = self.family_head(features)

        family_probabilities = torch.softmax(
            family_logits,
            dim=1,
        )

        genus_input = torch.cat(
            [
                features,
                family_probabilities,
            ],
            dim=1,
        )

        genus_logits = self.genus_head(genus_input)

        return {
            "family_logits": family_logits,
            "family_probabilities": family_probabilities,
            "genus_logits": genus_logits,
            "features": features,
        }


def create_hierarchical_model(
    number_of_families: int,
    number_of_genera: int,
    pretrained: bool = True,
    family_dropout: float = 0.2,
    genus_dropout: float = 0.3,
    genus_hidden_dimension: int = 512,
) -> HierarchicalEfficientNet:
    """Create the conditional hierarchical EfficientNet-B0 model."""

    return HierarchicalEfficientNet(
        number_of_families=number_of_families,
        number_of_genera=number_of_genera,
        pretrained=pretrained,
        family_dropout=family_dropout,
        genus_dropout=genus_dropout,
        genus_hidden_dimension=genus_hidden_dimension,
    )
