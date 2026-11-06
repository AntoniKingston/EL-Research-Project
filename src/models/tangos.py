import torch
import torch.nn.functional as F

from src.models.base import TabularModel
from src.models.resnet import Resnet


class Tangos(TabularModel):
    """TANGOS (Jeffares et al., 2023): regularizes the input-gradient attributions of the
    penultimate latent units to be sparse and mutually orthogonal."""

    def __init__(
        self, is_classification: bool, backbone: Resnet, sparsity: float, orthogonality: float
    ):
        super().__init__(is_classification)
        self.backbone = backbone
        self.output_dim = backbone.output_dim
        self.sparsity = sparsity
        self.orthogonality = orthogonality

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.backbone(x)

    def training_loss(self, x: torch.Tensor, y: torch.Tensor, loss_fn=None) -> torch.Tensor:
        loss_fn = loss_fn or self.task_loss
        x = x.detach().requires_grad_(True)
        z = self.backbone.embed(x)
        task = loss_fn(self._format_output(self.backbone.head(z)), y)

        n_units = z.shape[1]
        # One backward pass per latent unit (vectorized): attributions[k, b] = dz[b, k] / dx[b].
        grad_outputs = torch.eye(n_units, device=z.device).unsqueeze(1).expand(-1, len(z), -1)
        (attributions,) = torch.autograd.grad(
            z, x, grad_outputs=grad_outputs, is_grads_batched=True, create_graph=True
        )

        sparsity_loss = attributions.abs().mean()

        a = F.normalize(attributions.transpose(0, 1), dim=-1)  # (B, n_units, input_dim)
        cos = a @ a.transpose(1, 2)
        i, j = torch.triu_indices(n_units, n_units, offset=1, device=z.device)
        orthogonality_loss = cos[:, i, j].abs().mean()

        return task + self.sparsity * sparsity_loss + self.orthogonality * orthogonality_loss
