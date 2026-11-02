import torch


class Ensemble:
    def __init__(
        self,
        backbone,
        ensemble_size: int,
        mask: str,
        orthogonality_weight: float,
        additive_coef: float,
        similarity_metric: str,
    ):
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        if mask == "none":
            self.mask = torch.zeros(ensemble_size, ensemble_size)
        elif mask == "symmetric":
            self.mask = torch.ones(ensemble_size)
        elif mask == "lower_triangular":
            self.mask = torch.zeros((ensemble_size, ensemble_size))
            for i in range(1, ensemble_size):
                for j in range(i):
                    self.mask[i, j] = 1
        self.orthogonality_weight = orthogonality_weight
        self.additive_coef = additive_coef
        if similarity_metric not in ["cosine", "cka"]:
            raise ValueError(f"Invalid similarity metric: {similarity_metric}")
        self.similarity_metric = similarity_metric

    def forward(self, x):
        pass
