"""Loss functions for hierarchical family-genus classification.


"""
# import modules
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# creates class for hierarchical classification.
class HierarchicalLoss(nn.Module):
    """Combined family, genus and taxonomy-consistency loss.

    """

    def __init__(
        self,
        number_of_families: int,
        number_of_genera: int,
        family_to_genus_indices: Mapping[int, Sequence[int]],
        family_weight: float = 1.0,
        genus_weight: float = 0.5,
        consistency_weight: float = 0.2,
        genus_ignore_index: int = -1,
        label_smoothing: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        if number_of_families < 2:
            raise ValueError(
                "number_of_families must be at least 2."
            )

        if number_of_genera < 2:
            raise ValueError(
                "number_of_genera must be at least 2."
            )

        for name, value in {
            "family_weight": family_weight,
            "genus_weight": genus_weight,
            "consistency_weight": consistency_weight,
        }.items():
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                "label_smoothing must be between 0 and 1."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        taxonomy_mask = self._build_taxonomy_mask(
            number_of_families=number_of_families,
            number_of_genera=number_of_genera,
            family_to_genus_indices=family_to_genus_indices,
        )

        # Registered buffers automatically move to CPU/GPU together
        # with the loss module and are included in its state dictionary.
        self.register_buffer(
            "taxonomy_mask",
            taxonomy_mask,
        )

        # True for families that have at least one known genus.
        self.register_buffer(
            "family_has_known_genera",
            taxonomy_mask.any(dim=1),
        )

        self.number_of_families = number_of_families
        self.number_of_genera = number_of_genera

        self.family_weight = float(family_weight)
        self.genus_weight = float(genus_weight)
        self.consistency_weight = float(
            consistency_weight
        )

        self.genus_ignore_index = genus_ignore_index
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon

    @staticmethod
    def _build_taxonomy_mask(
        number_of_families: int,
        number_of_genera: int,
        family_to_genus_indices: Mapping[
            int,
            Sequence[int],
        ],
    ) -> Tensor:
        """Build a Boolean family-by-genus taxonomy matrix."""

        mask = torch.zeros(
            (
                number_of_families,
                number_of_genera,
            ),
            dtype=torch.bool,
        )

        for family_index, genus_indices in (
            family_to_genus_indices.items()
        ):
            if not 0 <= family_index < number_of_families:
                raise ValueError(
                    "Family index is outside the expected range: "
                    f"{family_index}"
                )

            for genus_index in genus_indices:
                if not 0 <= genus_index < number_of_genera:
                    raise ValueError(
                        "Genus index is outside the expected range: "
                        f"{genus_index}"
                    )

                mask[family_index, genus_index] = True

        if not mask.any():
            raise ValueError(
                "The taxonomy mask contains no valid "
                "family-genus relationships."
            )

        # Every genus must belong to exactly one family.
        genus_family_counts = mask.sum(dim=0)

        unassigned_genera = torch.where(
            genus_family_counts == 0
        )[0].tolist()

        if unassigned_genera:
            raise ValueError(
                "Some genera are not assigned to a family: "
                f"{unassigned_genera}"
            )

        multiply_assigned_genera = torch.where(
            genus_family_counts > 1
        )[0].tolist()

        if multiply_assigned_genera:
            raise ValueError(
                "Some genera are assigned to multiple families: "
                f"{multiply_assigned_genera}"
            )

        return mask

    def _validate_inputs(
        self,
        family_logits: Tensor,
        genus_logits: Tensor,
        family_targets: Tensor,
        genus_targets: Tensor,
    ) -> None:
        """Validate tensor shapes and target ranges."""

        if family_logits.ndim != 2:
            raise ValueError(
                "family_logits must have shape "
                "[batch_size, number_of_families]."
            )

        if genus_logits.ndim != 2:
            raise ValueError(
                "genus_logits must have shape "
                "[batch_size, number_of_genera]."
            )

        batch_size = family_logits.shape[0]

        if family_logits.shape != (
            batch_size,
            self.number_of_families,
        ):
            raise ValueError(
                "Unexpected family_logits shape: "
                f"{tuple(family_logits.shape)}"
            )

        if genus_logits.shape != (
            batch_size,
            self.number_of_genera,
        ):
            raise ValueError(
                "Unexpected genus_logits shape: "
                f"{tuple(genus_logits.shape)}"
            )

        if family_targets.shape != (batch_size,):
            raise ValueError(
                "family_targets must have shape [batch_size]."
            )

        if genus_targets.shape != (batch_size,):
            raise ValueError(
                "genus_targets must have shape [batch_size]."
            )

        invalid_family_targets = (
            (family_targets < 0)
            | (family_targets >= self.number_of_families)
        )

        if invalid_family_targets.any():
            values = family_targets[
                invalid_family_targets
            ].detach().cpu().tolist()

            raise ValueError(
                "Invalid family targets found: "
                f"{values}"
            )

        labelled_genus_mask = (
            genus_targets != self.genus_ignore_index
        )

        invalid_genus_targets = labelled_genus_mask & (
            (genus_targets < 0)
            | (genus_targets >= self.number_of_genera)
        )

        if invalid_genus_targets.any():
            values = genus_targets[
                invalid_genus_targets
            ].detach().cpu().tolist()

            raise ValueError(
                "Invalid genus targets found: "
                f"{values}"
            )

    def calculate_consistency_loss(
        self,
        genus_logits: Tensor,
        family_targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Penalize genus probability outside the correct family.

       
        """

        genus_probabilities = torch.softmax(
            genus_logits,
            dim=1,
        )

        batch_taxonomy_mask = self.taxonomy_mask[
            family_targets
        ]

        valid_specimen_mask = (
            self.family_has_known_genera[family_targets]
        )

        valid_specimen_count = (
            valid_specimen_mask.sum()
        )

        if not valid_specimen_mask.any():
            zero_loss = genus_logits.sum() * 0.0

            return zero_loss, valid_specimen_count

        allowed_probability = (
            genus_probabilities
            * batch_taxonomy_mask.to(
                dtype=genus_probabilities.dtype
            )
        ).sum(dim=1)

        valid_allowed_probability = allowed_probability[
            valid_specimen_mask
        ]

        consistency_loss = -torch.log(
            valid_allowed_probability.clamp_min(
                self.epsilon
            )
        ).mean()

        return consistency_loss, valid_specimen_count

    def forward(
        self,
        family_logits: Tensor,
        genus_logits: Tensor,
        family_targets: Tensor,
        genus_targets: Tensor,
    ) -> dict[str, Tensor]:
        """Calculate the complete hierarchical training loss."""

        family_targets = family_targets.long()
        genus_targets = genus_targets.long()

        self._validate_inputs(
            family_logits=family_logits,
            genus_logits=genus_logits,
            family_targets=family_targets,
            genus_targets=genus_targets,
        )

        # All training specimens have a known family.
        family_loss = F.cross_entropy(
            family_logits,
            family_targets,
            label_smoothing=self.label_smoothing,
        )

        genus_label_mask = (
            genus_targets != self.genus_ignore_index
        )

        labelled_genus_count = genus_label_mask.sum()

        # Avoid CrossEntropyLoss producing NaN when an entire batch
        # contains only family-level annotations.
        if genus_label_mask.any():
            genus_loss = F.cross_entropy(
                genus_logits[genus_label_mask],
                genus_targets[genus_label_mask],
                label_smoothing=self.label_smoothing,
            )
        else:
            genus_loss = genus_logits.sum() * 0.0

        (
            consistency_loss,
            consistency_specimen_count,
        ) = self.calculate_consistency_loss(
            genus_logits=genus_logits,
            family_targets=family_targets,
        )

        weighted_family_loss = (
            self.family_weight * family_loss
        )

        weighted_genus_loss = (
            self.genus_weight * genus_loss
        )

        weighted_consistency_loss = (
            self.consistency_weight
            * consistency_loss
        )

        total_loss = (
            weighted_family_loss
            + weighted_genus_loss
            + weighted_consistency_loss
        )

        return {
            "total_loss": total_loss,
            "family_loss": family_loss,
            "genus_loss": genus_loss,
            "consistency_loss": consistency_loss,
            "weighted_family_loss": weighted_family_loss,
            "weighted_genus_loss": weighted_genus_loss,
            "weighted_consistency_loss": (
                weighted_consistency_loss
            ),
            "genus_labelled_count": (
                labelled_genus_count
            ),
            "consistency_specimen_count": (
                consistency_specimen_count
            ),
        }
