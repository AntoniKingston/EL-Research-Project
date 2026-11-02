class BatchEnsemble:
    def __init__(self, backbone, ensemble_size: int, adapter_initialization: str):
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        self.adapter_initialization = adapter_initialization

    def forward(self, x):
        pass
