"""Taxonomically constrained inference for the hierarchical model.



The inference procedure then restricts the genus prediction to genera
belonging to the predicted family.


A genus belonging to another family cannot become the final prediction.
"""
# import modules
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from src.ml.hierarchical_model import create_hierarchical_model
from src.ml.transforms import get_validation_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "production"
    / "hierarchical"
    / "best_model.pt"
)


def select_device() -> torch.device:
    """Select CUDA, MPS or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def normalize_index_mapping(
    mapping: dict[Any, Any],
) -> dict[int, str]:
    """Convert checkpoint index mapping keys to integers."""

    return {
        int(index): str(label)
        for index, label in mapping.items()
    }


def normalize_genus_to_family_mapping(
    mapping: dict[Any, Any],
) -> dict[int, int]:
    """Convert genus-family mapping keys and values to integers."""

    return {
        int(genus_index): int(family_index)
        for genus_index, family_index in mapping.items()
    }


def build_taxonomy_mask(
    number_of_families: int,
    number_of_genera: int,
    genus_to_family: dict[int, int],
    device: torch.device,
) -> Tensor:
    """Build a family-by-genus Boolean taxonomy mask."""

    mask = torch.zeros(
        (
            number_of_families,
            number_of_genera,
        ),
        dtype=torch.bool,
        device=device,
    )

    for genus_index, family_index in genus_to_family.items():
        if not 0 <= family_index < number_of_families:
            raise ValueError(
                "Invalid family index in taxonomy mapping: "
                f"{family_index}"
            )

        if not 0 <= genus_index < number_of_genera:
            raise ValueError(
                "Invalid genus index in taxonomy mapping: "
                f"{genus_index}"
            )

        mask[family_index, genus_index] = True

    if not mask.any():
        raise ValueError(
            "The taxonomy mask contains no family-genus relationships."
        )

    genus_assignments = mask.sum(dim=0)

    missing_genera = torch.where(
        genus_assignments == 0
    )[0].detach().cpu().tolist()

    if missing_genera:
        raise ValueError(
            "Some genera are not assigned to a family: "
            f"{missing_genera}"
        )

    multiple_assignments = torch.where(
        genus_assignments > 1
    )[0].detach().cpu().tolist()

    if multiple_assignments:
        raise ValueError(
            "Some genera belong to multiple families: "
            f"{multiple_assignments}"
        )

    return mask


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load and validate the hierarchical checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "family_to_idx",
        "genus_to_idx",
        "genus_to_family_idx",
    }

    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return checkpoint

# creates class for Hierarchical Prediction
class HierarchicalPredictor:
    """Load a trained model and perform constrained predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or select_device()
        self.checkpoint_path = Path(checkpoint_path)

        checkpoint = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        self.checkpoint = checkpoint

        self.family_to_idx = {
            str(label): int(index)
            for label, index in checkpoint[
                "family_to_idx"
            ].items()
        }

        self.genus_to_idx = {
            str(label): int(index)
            for label, index in checkpoint[
                "genus_to_idx"
            ].items()
        }

        if "idx_to_family" in checkpoint:
            self.idx_to_family = normalize_index_mapping(
                checkpoint["idx_to_family"]
            )
        else:
            self.idx_to_family = {
                index: label
                for label, index in self.family_to_idx.items()
            }

        if "idx_to_genus" in checkpoint:
            self.idx_to_genus = normalize_index_mapping(
                checkpoint["idx_to_genus"]
            )
        else:
            self.idx_to_genus = {
                index: label
                for label, index in self.genus_to_idx.items()
            }

        self.genus_to_family = (
            normalize_genus_to_family_mapping(
                checkpoint["genus_to_family_idx"]
            )
        )

        configuration = checkpoint.get(
            "configuration",
            {},
        )

        self.image_size = int(
            configuration.get("image_size", 224)
        )

        self.view_code = checkpoint.get(
            "view_code",
            configuration.get("view_code", "FLP"),
        )

        self.model = create_hierarchical_model(
            number_of_families=len(
                self.family_to_idx
            ),
            number_of_genera=len(
                self.genus_to_idx
            ),
            pretrained=False,
            family_dropout=float(
                configuration.get(
                    "family_dropout",
                    0.2,
                )
            ),
            genus_dropout=float(
                configuration.get(
                    "genus_dropout",
                    0.3,
                )
            ),
            genus_hidden_dimension=int(
                configuration.get(
                    "genus_hidden_dimension",
                    512,
                )
            ),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.taxonomy_mask = build_taxonomy_mask(
            number_of_families=len(
                self.family_to_idx
            ),
            number_of_genera=len(
                self.genus_to_idx
            ),
            genus_to_family=self.genus_to_family,
            device=self.device,
        )

        self.transform = get_validation_transforms(
            image_size=(
                self.image_size,
                self.image_size,
            )
        )

    def allowed_genera_for_family(
        self,
        family_index: int,
    ) -> list[str]:
        """Return all genera allowed for one family."""

        allowed_indices = torch.where(
            self.taxonomy_mask[family_index]
        )[0].detach().cpu().tolist()

        return [
            self.idx_to_genus[index]
            for index in allowed_indices
        ]

    def constrained_predictions(
        self,
        family_logits: Tensor,
        genus_logits: Tensor,
    ) -> dict[str, Tensor]:
        """Apply hard family-to-genus constraints.

        The family prediction is selected first. The genus prediction
        is then selected only among genera belonging to that family.
        """

        family_probabilities = torch.softmax(
            family_logits,
            dim=1,
        )

        raw_genus_probabilities = torch.softmax(
            genus_logits,
            dim=1,
        )

        predicted_family_indices = (
            family_probabilities.argmax(dim=1)
        )

        allowed_genus_mask = self.taxonomy_mask[
            predicted_family_indices
        ]

        constrained_genus_probabilities = (
            raw_genus_probabilities
            * allowed_genus_mask.to(
                dtype=raw_genus_probabilities.dtype
            )
        )

        constrained_probability_sum = (
            constrained_genus_probabilities.sum(
                dim=1,
                keepdim=True,
            )
        )

        constrained_genus_probabilities = (
            constrained_genus_probabilities
            / constrained_probability_sum.clamp_min(
                1e-8
            )
        )

        predicted_genus_indices = (
            constrained_genus_probabilities.argmax(
                dim=1
            )
        )

        family_confidences = (
            family_probabilities.gather(
                1,
                predicted_family_indices.unsqueeze(1),
            ).squeeze(1)
        )

        genus_confidences = (
            constrained_genus_probabilities.gather(
                1,
                predicted_genus_indices.unsqueeze(1),
            ).squeeze(1)
        )

        raw_genus_indices = (
            raw_genus_probabilities.argmax(dim=1)
        )

        raw_genus_confidences = (
            raw_genus_probabilities.gather(
                1,
                raw_genus_indices.unsqueeze(1),
            ).squeeze(1)
        )

        raw_genus_family_indices = torch.tensor(
            [
                self.genus_to_family[
                    int(genus_index)
                ]
                for genus_index in (
                    raw_genus_indices
                    .detach()
                    .cpu()
                    .tolist()
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

        raw_prediction_is_consistent = (
            raw_genus_family_indices
            == predicted_family_indices
        )

        return {
            "family_probabilities": (
                family_probabilities
            ),
            "raw_genus_probabilities": (
                raw_genus_probabilities
            ),
            "constrained_genus_probabilities": (
                constrained_genus_probabilities
            ),
            "family_indices": (
                predicted_family_indices
            ),
            "genus_indices": (
                predicted_genus_indices
            ),
            "family_confidences": (
                family_confidences
            ),
            "genus_confidences": (
                genus_confidences
            ),
            "raw_genus_indices": (
                raw_genus_indices
            ),
            "raw_genus_confidences": (
                raw_genus_confidences
            ),
            "raw_prediction_is_consistent": (
                raw_prediction_is_consistent
            ),
        }

    def predict_tensor(
        self,
        image_tensor: Tensor,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Predict one or more already transformed images."""

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.ndim != 4:
            raise ValueError(
                "image_tensor must have shape "
                "[channels, height, width] or "
                "[batch, channels, height, width]."
            )

        image_tensor = image_tensor.to(self.device)

        with torch.inference_mode():
            outputs = self.model(image_tensor)

            predictions = self.constrained_predictions(
                family_logits=outputs[
                    "family_logits"
                ],
                genus_logits=outputs[
                    "genus_logits"
                ],
            )

        results: list[dict[str, Any]] = []

        batch_size = image_tensor.shape[0]

        for batch_index in range(batch_size):
            family_index = int(
                predictions["family_indices"][
                    batch_index
                ].item()
            )

            genus_index = int(
                predictions["genus_indices"][
                    batch_index
                ].item()
            )

            raw_genus_index = int(
                predictions["raw_genus_indices"][
                    batch_index
                ].item()
            )

            constrained_probabilities = predictions[
                "constrained_genus_probabilities"
            ][batch_index]

            allowed_count = int(
                self.taxonomy_mask[
                    family_index
                ].sum().item()
            )

            effective_top_k = min(
                top_k,
                allowed_count,
            )

            top_probabilities, top_indices = (
                torch.topk(
                    constrained_probabilities,
                    k=effective_top_k,
                )
            )

            top_genera = [
                {
                    "genus": self.idx_to_genus[
                        int(index)
                    ],
                    "confidence": float(
                        probability
                    ),
                }
                for probability, index in zip(
                    top_probabilities
                    .detach()
                    .cpu()
                    .tolist(),
                    top_indices
                    .detach()
                    .cpu()
                    .tolist(),
                )
            ]

            raw_genus_family_index = (
                self.genus_to_family[
                    raw_genus_index
                ]
            )

            result = {
                "family": self.idx_to_family[
                    family_index
                ],
                "family_index": family_index,
                "family_confidence": float(
                    predictions[
                        "family_confidences"
                    ][batch_index].item()
                ),
                "genus": self.idx_to_genus[
                    genus_index
                ],
                "genus_index": genus_index,
                "genus_confidence": float(
                    predictions[
                        "genus_confidences"
                    ][batch_index].item()
                ),
                "raw_genus": self.idx_to_genus[
                    raw_genus_index
                ],
                "raw_genus_confidence": float(
                    predictions[
                        "raw_genus_confidences"
                    ][batch_index].item()
                ),
                "raw_genus_family": (
                    self.idx_to_family[
                        raw_genus_family_index
                    ]
                ),
                "raw_prediction_was_consistent": bool(
                    predictions[
                        "raw_prediction_is_consistent"
                    ][batch_index].item()
                ),
                "allowed_genera": (
                    self.allowed_genera_for_family(
                        family_index
                    )
                ),
                "top_constrained_genera": top_genera,
                "view_code": self.view_code,
            }

            results.append(result)

        return results

    def predict_image(
        self,
        image_path: str | Path,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Predict family and genus for one image."""

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        result = self.predict_tensor(
            image_tensor=image_tensor,
            top_k=top_k,
        )[0]

        result["image_path"] = str(
            image_path.resolve()
        )

        return result


def print_prediction(
    prediction: dict[str, Any],
) -> None:
    """Print a human-readable prediction."""

    print("\nHierarchical taxonomic prediction")
    print("=" * 72)
    print(
        f"Image: {prediction['image_path']}"
    )
    print(
        f"Expected view: {prediction['view_code']}"
    )
    print("-" * 72)

    print(
        "Predicted family: "
        f"{prediction['family']} "
        f"({prediction['family_confidence']:.2%})"
    )

    print(
        "Final constrained genus: "
        f"{prediction['genus']} "
        f"({prediction['genus_confidence']:.2%})"
    )

    print(
        "Raw genus before constraint: "
        f"{prediction['raw_genus']} "
        f"({prediction['raw_genus_confidence']:.2%})"
    )

    print(
        "Family of raw genus: "
        f"{prediction['raw_genus_family']}"
    )

    print(
        "Raw prediction taxonomically consistent: "
        f"{prediction['raw_prediction_was_consistent']}"
    )

    if not prediction[
        "raw_prediction_was_consistent"
    ]:
        print(
            "\nThe raw genus was replaced because it did "
            "not belong to the predicted family."
        )

    print("\nAllowed genera for predicted family:")

    for genus in prediction["allowed_genera"]:
        print(f"  - {genus}")

    print("\nTop constrained genus predictions:")

    for rank, item in enumerate(
        prediction["top_constrained_genera"],
        start=1,
    ):
        print(
            f"  {rank}. {item['genus']}: "
            f"{item['confidence']:.2%}"
        )


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict Brachycera family and genus from "
            "an FLP image using hierarchical constraints."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the FLP image.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to the hierarchical model checkpoint."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of constrained genus candidates "
            "to display."
        ),
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help=(
            "Optional path for saving prediction as JSON."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Command-line inference entry point."""

    arguments = parse_arguments()

    if arguments.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    predictor = HierarchicalPredictor(
        checkpoint_path=arguments.checkpoint
    )

    prediction = predictor.predict_image(
        image_path=arguments.image,
        top_k=arguments.top_k,
    )

    print_prediction(prediction)

    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with arguments.json_output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                prediction,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            "\nPrediction saved to: "
            f"{arguments.json_output}"
        )


if __name__ == "__main__":
    main()
