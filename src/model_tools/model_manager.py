from abc import ABC, abstractmethod

import torch.nn as nn
from torch import optim

from src.dataset_tools.domain import DatasetSplits
from src.model_tools.utils import mean_score_by_folds
from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble
from src.models.resnet import Resnet
from src.models.tangos import Tangos


class IModelManager(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def hyperparameters_study(self, trial, device, X, Y, folds: list[DatasetSplits]):
        pass


class SingleResNet(IModelManager):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_dim: int = 10
        self.output_dim: int = 2
        self.epochs: int = 100
        self.patience: int = 16
        self.optimizer = optim.AdamW
        self.lr: float = 0.01
        self.wd: float = 0.01
        self.batch_size: int = 64
        self.loss_fn = nn.CrossEntropyLoss()

    def hyperparameters_study(self, trial, device, X, Y, folds: list[DatasetSplits]):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        n_blocks = trial.suggest_int("n_blocks", 2, 4)
        dr = trial.suggest_float("dr", 0.0, 0.5)

        def model_fun():
            return Resnet(self.input_dim, self.output_dim, hidden_dim, n_blocks, dr)

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            folds,
            self.epochs,
            self.patience,
            self.lr,
            self.wd,
            self.optimizer,
            self.loss_fn,
            self.batch_size,
            device,
        )


class TangosManager(IModelManager):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_dim: int = 10
        self.output_dim: int = 2
        self.hidden_dim: int = 128
        self.n_blocks: int = 3
        self.dr: float = 0.2
        self.epochs: int = 100
        self.patience: int = 16
        self.optimizer = optim.AdamW
        self.batch_size: int = 64
        self.loss_fn = nn.CrossEntropyLoss()

    def hyperparameters_study(self, trial, device, X, Y, folds: list[DatasetSplits]):
        lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
        sparsity = trial.suggest_float("sparsity", 1e-4, 1e-1, log=True)
        orthogonality = trial.suggest_float("orthogonality", 1e-3, 1.0, log=True)

        def model_fun():
            return Tangos(
                Resnet(self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                sparsity,
                orthogonality,
            )

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            folds,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.loss_fn,
            self.batch_size,
            device,
        )


class BatchEnsembleManager(IModelManager):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_dim: int = 10
        self.output_dim: int = 2
        self.hidden_dim: int = 128
        self.n_blocks: int = 3
        self.dr: float = 0.2
        self.ensemble_size: int = 4
        self.epochs: int = 100
        self.patience: int = 16
        self.optimizer = optim.AdamW
        self.batch_size: int = 64
        self.loss_fn = nn.CrossEntropyLoss()

    def hyperparameters_study(self, trial, device, X, Y, folds: list[DatasetSplits]):
        lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
        adapter_initialization = trial.suggest_categorical(
            "adapter_initialization", ["random_sign", "normal"]
        )

        def model_fun():
            return BatchEnsemble(
                Resnet(self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                self.ensemble_size,
                adapter_initialization,
            )

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            folds,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.loss_fn,
            self.batch_size,
            device,
        )


class EnsembleResnetManager(IModelManager):
    def __init__(self, name: str):
        super().__init__(name)
        self.input_dim: int = 10
        self.output_dim: int = 2
        self.hidden_dim: int = 128
        self.n_blocks: int = 3
        self.dr: float = 0.2
        self.ensemble_size: int = 4
        self.mask: str = "lower_triangular"  
        self.orthogonality_weight: float = 0.1
        self.epochs: int = 100
        self.patience: int = 16
        self.optimizer = optim.AdamW
        self.batch_size: int = 64
        self.loss_fn = nn.CrossEntropyLoss()

    def hyperparameters_study(self, trial, device, X, Y, folds: list[DatasetSplits]):
        lr = trial.suggest_float("lr", 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float("wd", 2.5 * 1e-3, 1e-2, log=True)
        additive_coef = trial.suggest_float("additive_coef", 0.0, 1.0)
        similarity_metric = trial.suggest_categorical("similarity_metric", ["cosine", "cka"])

        def model_fun():
            return Ensemble(
                Resnet(self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                self.ensemble_size,
                self.mask,
                self.orthogonality_weight,
                additive_coef,
                similarity_metric,
            )

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            folds,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.loss_fn,
            self.batch_size,
            device,
        )
