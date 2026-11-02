from torch import nn


class Tangos:
    def __init__(self, backbone: nn.Module, sparsity: float, orthogonality: float):
        self.backbone = backbone
        self.sparsity = sparsity
        self.orthogonality = orthogonality

    def forward(self, x):
        return self.backbone(x)
