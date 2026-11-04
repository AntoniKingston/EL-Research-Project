import torch
from torch import nn

from src.models.base import TabularModel


class ResBlock(nn.Module):
    """u + Drop2(W2 Drop1(ReLU(W1 BN(u) + b1)) + b2)"""

    def __init__(self, hidden_dim: int, dr: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dr)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dr)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        h = self.drop1(torch.relu(self.linear1(self.bn(u))))
        return u + self.drop2(self.linear2(h))


class Resnet(TabularModel):
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, dr):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dr = dr

        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dr) for _ in range(n_blocks)])
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate latent representation Z = ReLU(BN(U^(L)))."""
        return torch.relu(self.final_bn(self.blocks(self.stem(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.head(self.embed(x))
