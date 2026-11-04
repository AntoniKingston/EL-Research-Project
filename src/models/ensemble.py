import copy
from typing import cast

import torch
from torch import nn

from src.models.base import TabularModel
from src.models.resnet import Resnet

EPS = 1e-8


def cosine_similarity_matrix(Z: torch.Tensor, Z_ref: torch.Tensor) -> torch.Tensor:
    """Squared functional cosine similarity between all member pairs.

    Z, Z_ref: batch-centered latents of shape (M, B, d). Returns (M, M) with entry [m, j]
    equal to Phi_Cos(Z[m], Z_ref[j]).
    """
    num = torch.einsum("mbk,jbk->mjk", Z, Z_ref)
    den = Z.norm(dim=1)[:, None] * Z_ref.norm(dim=1)[None] + EPS
    return (num / den).pow(2).mean(dim=-1)


def cka_matrix(Z: torch.Tensor, Z_ref: torch.Tensor) -> torch.Tensor:
    """Linear CKA between all member pairs, computed in feature space. Shapes as above."""
    cross = torch.einsum("mbk,jbl->mjkl", Z, Z_ref).pow(2).sum(dim=(-2, -1))
    self_norm = torch.einsum("mbk,mbl->mkl", Z, Z).flatten(1).norm(dim=-1)
    ref_norm = torch.einsum("mbk,mbl->mkl", Z_ref, Z_ref).flatten(1).norm(dim=-1)
    return cross / (self_norm[:, None] * ref_norm[None] + EPS)


SIMILARITY_METRICS = {"cosine": cosine_similarity_matrix, "cka": cka_matrix}


class Ensemble(TabularModel):
    """Additive ensemble Y = f_1(X) + nu * sum_{m>=2} f_m(X) of independent Tabular ResNets.

    mask:
        "none"             - unregularized ensemble (lambda = 0)
        "symmetric"        - pairwise repulsion with gradients flowing to both members
        "lower_triangular" - SFOvTSGM: member m is repelled from sg(Z_j) for j < m
    """

    mask: torch.Tensor
    member_weights: torch.Tensor

    def __init__(
        self,
        backbone: Resnet,
        ensemble_size: int,
        mask: str,
        orthogonality_weight: float,
        additive_coef: float,
        similarity_metric: str,
    ):
        super().__init__()
        if mask not in ["none", "symmetric", "lower_triangular"]:
            raise ValueError(f"Invalid mask: {mask}")
        if similarity_metric not in SIMILARITY_METRICS:
            raise ValueError(f"Invalid similarity metric: {similarity_metric}")

        self.output_dim = backbone.output_dim
        self.ensemble_size = ensemble_size
        self.mask_type = mask
        self.orthogonality_weight = orthogonality_weight
        self.additive_coef = additive_coef
        self.similarity_metric = similarity_metric

        self.members = nn.ModuleList([backbone])
        for _ in range(ensemble_size - 1):
            member = copy.deepcopy(backbone)
            for module in member.modules():
                if isinstance(module, nn.Linear | nn.BatchNorm1d):
                    module.reset_parameters()
            self.members.append(member)

        # Row m holds the members j < m that member m is compared against; normalizing by the
        # row count gives the 1 / (m - 1) weighting of the plan.
        if mask == "none":
            pair_mask = torch.zeros(ensemble_size, ensemble_size)
        else:
            pair_mask = torch.tril(torch.ones(ensemble_size, ensemble_size), diagonal=-1)
            pair_mask = pair_mask / pair_mask.sum(dim=1, keepdim=True).clamp_min(1)
        self.register_buffer("mask", pair_mask)

        weights = torch.full((ensemble_size,), float(additive_coef))
        weights[0] = 1.0
        self.register_buffer("member_weights", weights)

    def _members_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        members = cast("list[Resnet]", list(self.members))
        latents = [member.embed(x) for member in members]
        outputs = [member.head(z) for member, z in zip(members, latents, strict=True)]
        return torch.stack(latents), torch.stack(outputs)

    def _combine(self, outputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("m,mbo->bo", self.member_weights, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        _, outputs = self._members_forward(x)
        return self._combine(outputs)

    def orthogonality_penalty(self, latents: torch.Tensor) -> torch.Tensor:
        Z = latents - latents.mean(dim=1, keepdim=True)
        Z_ref = Z.detach() if self.mask_type == "lower_triangular" else Z
        similarity = SIMILARITY_METRICS[self.similarity_metric](Z, Z_ref)
        return (self.mask * similarity).sum()

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> torch.Tensor:
        latents, outputs = self._members_forward(x)
        loss = loss_fn(self._format_output(self._combine(outputs)), y)
        if self.mask_type != "none" and self.orthogonality_weight > 0:
            loss = loss + self.orthogonality_weight * self.orthogonality_penalty(latents)
        return loss
