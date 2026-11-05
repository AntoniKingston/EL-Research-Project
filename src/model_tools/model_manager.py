from abc import ABC, abstractmethod

import torch.nn as nn
from torch import optim

from src.dataset_tools.domain import DatasetSplits
from src.model_tools.utils import mean_score_by_folds, train_model
from src.models.base import TabularModel
from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble
from src.models.resnet import Resnet
from src.models.tangos import Tangos


class IModelManager(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def hyperparameters_study(self, trial, device, X, Y, is_classification: bool, fold: DatasetSplits):
        pass

    @abstractmethod
    def train_optimal_model(self, params: dict, device, X, Y, is_classification: bool, fold: DatasetSplits) -> TabularModel:
        pass


# ===================================================
# -----------------SingleResne-----------------------
# ===================================================


class SingleResnetManager(IModelManager):
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

        self._hidden_dim_name = 'hidden_dim'
        self._n_blocks_name = 'n_blocks'
        self._dr_name = 'dr'

    def hyperparameters_study(self, trial, device, X, Y, is_classification: bool, fold: DatasetSplits):
        hidden_dim = trial.suggest_categorical(self._hidden_dim_name, [64, 128, 256])
        n_blocks = trial.suggest_int(self._n_blocks_name, 2, 4)
        dr = trial.suggest_float(self._dr_name, 0.0, 0.5)

        def model_fun():
            return Resnet(is_classification, self.input_dim, self.output_dim, hidden_dim, n_blocks, dr)

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            self.lr,
            self.wd,
            self.optimizer,
            self.batch_size,
            device,
        )

    def train_optimal_model(self, params: dict, device, X, Y, is_classification: bool, fold: DatasetSplits) -> TabularModel:
        hidden_dim = params.get(self._hidden_dim_name, 128)
        n_blocks = params.get(self._n_blocks_name, 3)
        dr = params.get(self._dr_name, 0.2)
        def model_fun():
            return Resnet(is_classification, self.input_dim, self.output_dim, hidden_dim, n_blocks, dr)
        return train_model(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            self.lr,
            self.wd,
            self.optimizer,
            self.batch_size,
            device,
        )


# ===================================================
# --------------------Tangos-------------------------
# ===================================================


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

        self._lr_name = 'lr'
        self._wd_name = 'wd'
        self._sparsity_name = 'sparsity'
        self._othodonality_name = 'orthogonality'

    def hyperparameters_study(self, trial, device, X, Y, is_classification: bool, fold: DatasetSplits):
        lr = trial.suggest_float(self._lr_name, 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float(self._wd_name, 2.5 * 1e-3, 1e-2, log=True)
        sparsity = trial.suggest_float(self._sparsity_name, 1e-4, 1e-1, log=True)
        orthogonality = trial.suggest_float(self._othodonality_name, 1e-3, 1.0, log=True)

        def model_fun():
            return Tangos(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                sparsity,
                orthogonality,
            )

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )

    def train_optimal_model(self, params: dict, device, X, Y, is_classification: bool, fold: DatasetSplits) -> TabularModel:
        lr = params.get(self._lr_name, 1e-3)
        wd = params.get(self._wd_name, 1e-2)
        sparsity = params.get(self._sparsity_name, 1e-3)
        orthogonality = params.get(self._othodonality_name, 1e-2)
        def model_fun():
            return Tangos(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                sparsity,
                orthogonality,
            )
        return train_model(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )


# ===================================================
# -----------------BatchEnsemble---------------------
# ===================================================


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

        self._lr_name = 'lr'
        self._wd_name = 'wd'
        self._adapter_initialization_name = 'adapter_initialization'

    def hyperparameters_study(self, trial, device, X, Y, is_classification: bool, fold: DatasetSplits):
        lr = trial.suggest_float(self._lr_name, 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float(self._wd_name, 2.5 * 1e-3, 1e-2, log=True)
        adapter_initialization = trial.suggest_categorical(
            self._adapter_initialization_name, ['random_sign', 'normal']
        )

        def model_fun():
            return BatchEnsemble(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                self.ensemble_size,
                adapter_initialization,
            )

        return mean_score_by_folds(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )
    
    def train_optimal_model(self, params: dict, device, X, Y, is_classification: bool, fold: DatasetSplits) -> TabularModel:
        lr = params.get(self._lr_name, 1e-3)
        wd = params.get(self._wd_name, 1e-2)
        adapter_initialization = params.get(self._adapter_initialization_name, 'random_sign')

        def model_fun():
            return BatchEnsemble(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                self.ensemble_size,
                adapter_initialization,
            )

        return train_model(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )


# ===================================================
# ----------------EnsembleResnet---------------------
# ===================================================


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

        self._lr_name = 'lr'
        self._wd_name = 'wd'
        self._additive_coef_name = 'additive_coef'
        self._similarity_metric = 'similarity_metric'

    def hyperparameters_study(self, trial, device, X, Y, is_classification: bool, fold: DatasetSplits):
        lr = trial.suggest_float(self._lr_name, 1e-4, 5 * 1e-2, log=True)
        wd = trial.suggest_float(self._wd_name, 2.5 * 1e-3, 1e-2, log=True)
        additive_coef = trial.suggest_float(self._additive_coef_name, 0.0, 1.0)
        similarity_metric = trial.suggest_categorical(self._similarity_metric, ["cosine", "cka"])

        def model_fun():
            return Ensemble(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
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
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )

    def train_optimal_model(self, params: dict, device, X, Y, is_classification: bool, fold: DatasetSplits) -> TabularModel:
        lr = params.get(self._lr_name, 1e-3)
        wd = params.get(self._wd_name, 1e-2)
        additive_coef = params.get(self._additive_coef_name, 0.5)
        similarity_metric = params.get(self._similarity_metric, 'cosine')

        def model_fun():
            return Ensemble(
                is_classification,
                Resnet(is_classification, self.input_dim, self.output_dim, self.hidden_dim, self.n_blocks, self.dr),
                self.ensemble_size,
                self.mask,
                self.orthogonality_weight,
                additive_coef,
                similarity_metric,
            )

        return train_model(
            model_fun,
            X,
            Y,
            fold,
            self.epochs,
            self.patience,
            lr,
            wd,
            self.optimizer,
            self.batch_size,
            device,
        )
