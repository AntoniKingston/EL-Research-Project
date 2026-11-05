import torch
from torch import nn

from src.models.base import TabularModel
from src.models.resnet import Resnet


class BatchEnsembleLinear(nn.Module):
    """Shared weight W with per-member rank-1 fast weights: y_i = ((x * r_i) W^T) * s_i + b_i.

    Expects the batch to be the concatenation of `ensemble_size` equally sized member batches.
    """

    def __init__(self, linear: nn.Linear, ensemble_size: int, adapter_initialization: str):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = linear.weight
        self.r = nn.Parameter(torch.empty(ensemble_size, linear.in_features))
        self.s = nn.Parameter(torch.empty(ensemble_size, linear.out_features))
        bias = linear.bias.detach() if linear.bias is not None else torch.zeros(linear.out_features)
        self.bias = nn.Parameter(bias.repeat(ensemble_size, 1))

        with torch.no_grad():
            for p in (self.r, self.s):
                if adapter_initialization == "random_sign":
                    p.bernoulli_(0.5).mul_(2).sub_(1)
                elif adapter_initialization == "normal":
                    p.normal_(1.0, 0.5)
                else:
                    raise ValueError(f"Invalid adapter initialization: {adapter_initialization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        x = x.view(self.ensemble_size, -1, x.shape[-1])
        out = (x * self.r[:, None]) @ self.weight.T
        out = out * self.s[:, None] + self.bias[:, None]
        return out.reshape(-1, out.shape[-1])


class BatchEnsemble(TabularModel):
    """BatchEnsemble (Wen et al., 2020) over a Tabular ResNet: every nn.Linear of the backbone
    is replaced in place by a BatchEnsembleLinear. BatchNorm layers stay shared."""

    def __init__(
        self,
        is_classification: bool,
        backbone: Resnet,
        ensemble_size: int,
        adapter_initialization: str,
    ):
        super().__init__(is_classification)
        self.backbone = backbone
        self.output_dim = backbone.output_dim
        self.ensemble_size = ensemble_size
        self.adapter_initialization = adapter_initialization
        self._replace_linears(self.backbone)

    def _replace_linears(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                be = BatchEnsembleLinear(child, self.ensemble_size, self.adapter_initialization)
                setattr(module, name, be)
            else:
                self._replace_linears(child)

    def member_outputs(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x.repeat(self.ensemble_size, 1))
        return out.view(self.ensemble_size, len(x), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.member_outputs(x).mean(dim=0)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn) -> torch.Tensor:
        out = self.member_outputs(x).flatten(0, 1)
        return loss_fn(self._format_output(out), y.repeat(self.ensemble_size))
