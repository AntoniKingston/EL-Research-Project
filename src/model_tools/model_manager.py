from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import optuna

from src.dataset_tools.preprocessing import FoldData
from src.model_tools.utils import RunContext, fold_score, set_seed, train_model
from src.models.base import TabularModel
from src.models.batchensemble import BatchEnsemble
from src.models.ensemble import Ensemble
from src.models.resnet import Resnet
from src.models.tangos import Tangos

STAGE1_MODEL = "single_resnet"
BACKBONE_KEYS = ("hidden_dim", "n_blocks", "dr")


def make_resnet(backbone: dict[str, Any], ctx: RunContext) -> Resnet:
    return Resnet(
        ctx.is_classification,
        ctx.input_dim,
        ctx.output_dim,
        backbone["hidden_dim"],
        backbone["n_blocks"],
        backbone["dr"],
    )


def suggest_optimization(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
    }


class IModelManager(ABC):
    """Defines one method: its search space (`suggest`) and how to build it (`build`).

    Tuning and refitting are shared. `backbone` is the Stage 1 architecture
    (hidden_dim, n_blocks, dr); the Stage 1 manager tunes it itself and receives None.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    def is_stage1(self) -> bool:
        return False

    @abstractmethod
    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        pass

    @abstractmethod
    def build(
        self, params: dict[str, Any], backbone: dict[str, Any] | None, ctx: RunContext, seed: int
    ) -> TabularModel:
        pass

    def optimization_params(self, params: dict[str, Any], ctx: RunContext) -> tuple[float, float]:
        return params["lr"], params["wd"]

    def hyperparameters_study(
        self,
        trial: optuna.Trial,
        ctx: RunContext,
        fold_data: FoldData,
        backbone: dict[str, Any] | None,
        seed: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> float:
        """Optuna objective: inner validation task loss. Stores the best epoch on the trial."""
        params = self.suggest(trial, ctx)
        lr, wd = self.optimization_params(params, ctx)
        set_seed(seed)
        score, best_epoch = fold_score(
            lambda: self.build(params, backbone, ctx, seed),
            fold_data,
            ctx,
            lr,
            wd,
            trial,
            should_stop,
        )
        trial.set_user_attr("best_epoch", best_epoch)
        return score

    def train_optimal_model(
        self,
        params: dict[str, Any],
        backbone: dict[str, Any] | None,
        ctx: RunContext,
        fold_data: FoldData,
        seed: int,
        epochs: int,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ) -> TabularModel:
        lr, wd = self.optimization_params(params, ctx)
        set_seed(seed)
        return train_model(
            lambda: self.build(params, backbone, ctx, seed),
            fold_data,
            ctx,
            lr,
            wd,
            epochs,
            checkpoint_path,
            checkpoint_every,
            should_stop,
        )


class SingleResnetManager(IModelManager):
    """Stage 1: tunes the backbone architecture with fixed optimization settings."""

    @property
    def is_stage1(self) -> bool:
        return True

    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", ctx.hidden_dims),
            "n_blocks": trial.suggest_int("n_blocks", 2, 4),
            "dr": trial.suggest_float("dr", 0.0, 0.5),
        }

    def optimization_params(self, params: dict[str, Any], ctx: RunContext) -> tuple[float, float]:
        return ctx.stage1_lr, ctx.stage1_wd

    def build(self, params, backbone, ctx, seed) -> TabularModel:
        return make_resnet(params, ctx)


class ResnetManager(IModelManager):
    """Stage 2 single-network baseline: the Stage 1 backbone with tuned optimization settings."""

    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        return suggest_optimization(trial)

    def build(self, params, backbone, ctx, seed) -> TabularModel:
        return make_resnet(backbone, ctx)


class TangosManager(IModelManager):
    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        return {
            **suggest_optimization(trial),
            "sparsity": trial.suggest_float("sparsity", 1e-4, 1e-1, log=True),
            "orthogonality": trial.suggest_float("orthogonality", 1e-3, 1.0, log=True),
        }

    def build(self, params, backbone, ctx, seed) -> TabularModel:
        return Tangos(make_resnet(backbone, ctx), params["sparsity"], params["orthogonality"])


class BatchEnsembleManager(IModelManager):
    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        return {
            **suggest_optimization(trial),
            "adapter_initialization": trial.suggest_categorical(
                "adapter_initialization", ["random_sign", "normal"]
            ),
        }

    def build(self, params, backbone, ctx, seed) -> TabularModel:
        return BatchEnsemble(
            make_resnet(backbone, ctx), ctx.ensemble_size, params["adapter_initialization"]
        )


class EnsembleResnetManager(IModelManager):
    """Additive ensemble; `mask` selects unregularized / symmetric / SFOvTSGM."""

    def __init__(self, name: str, mask: str, similarity_metric: str):
        super().__init__(name)
        self.mask = mask
        self.similarity_metric = similarity_metric

    def suggest(self, trial: optuna.Trial, ctx: RunContext) -> dict[str, Any]:
        params = {
            **suggest_optimization(trial),
            "additive_coef": trial.suggest_float("additive_coef", 0.05, 1.0, log=True),
        }
        if self.mask != "none":
            params["orthogonality_weight"] = trial.suggest_float(
                "orthogonality_weight", 1e-3, 10.0, log=True
            )
        return params

    def build(self, params, backbone, ctx, seed) -> TabularModel:
        return Ensemble(
            make_resnet(backbone, ctx),
            ctx.ensemble_size,
            self.mask,
            params.get("orthogonality_weight", 0.0),
            params["additive_coef"],
            self.similarity_metric,
            seed=seed,
        )


MODEL_REGISTRY: dict[str, Callable[[str], IModelManager]] = {
    STAGE1_MODEL: SingleResnetManager,
    "resnet": ResnetManager,
    "tangos": TangosManager,
    "batchensemble": BatchEnsembleManager,
    "ensemble_none": partial(EnsembleResnetManager, mask="none", similarity_metric="cosine"),
    "sym_cka": partial(EnsembleResnetManager, mask="symmetric", similarity_metric="cka"),
    "sym_cos": partial(EnsembleResnetManager, mask="symmetric", similarity_metric="cosine"),
    "sfo_cka": partial(EnsembleResnetManager, mask="lower_triangular", similarity_metric="cka"),
    "sfo_cos": partial(EnsembleResnetManager, mask="lower_triangular", similarity_metric="cosine"),
}
