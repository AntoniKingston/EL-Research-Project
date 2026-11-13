import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import optim

from src.dataset_tools.preprocessing import FoldData
from src.models.base import TabularModel

OPTIMIZERS: dict[str, type[optim.Optimizer]] = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
}


@dataclass
class RunContext:
    """Everything a model manager needs to build and train models for one dataset."""

    input_dim: int
    output_dim: int
    is_classification: bool
    device: str
    optimizer: type[optim.Optimizer]
    batch_size: int
    max_epochs: int
    patience: int
    stage1_lr: float
    stage1_wd: float
    hidden_dims: list[int]
    ensemble_size: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def eval_task_loss(model: TabularModel, X: np.ndarray, y: np.ndarray) -> float:
    pred = torch.as_tensor(model.predict(X))
    with torch.no_grad():
        return model.task_loss(pred, torch.as_tensor(y)).item()


def fold_score(
    model_fun: Callable[[], TabularModel],
    fold_data: FoldData,
    ctx: RunContext,
    lr: float,
    wd: float,
    trial: optuna.Trial | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[float, int]:
    """Trains on the inner train split with early stopping on the inner validation split.

    Returns (validation task loss of the restored best weights, best epoch). With `trial`,
    the validation loss is reported every epoch and the trial may be pruned.
    """

    def report(epoch: int, val_loss: float) -> None:
        if trial is None:
            return
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model = model_fun().to(ctx.device)
    model.fit(
        fold_data.X_train,
        fold_data.y_train,
        X_val=fold_data.X_eval,
        y_val=fold_data.y_eval,
        epochs=ctx.max_epochs,
        patience=ctx.patience,
        lr=lr,
        wd=wd,
        optimizer=ctx.optimizer,
        batch_size=ctx.batch_size,
        on_epoch_end=report,
        should_stop=should_stop,
    )
    return eval_task_loss(model, fold_data.X_eval, fold_data.y_eval), model.best_epoch


def train_model(
    model_fun: Callable[[], TabularModel],
    fold_data: FoldData,
    ctx: RunContext,
    lr: float,
    wd: float,
    epochs: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 0,
    should_stop: Callable[[], bool] | None = None,
) -> TabularModel:
    """Refits on the full outer training partition for a fixed number of epochs.

    The test partition (`fold_data.X_eval`) is never seen during training: the epoch count
    comes from the inner validation run of the selected configuration.
    """
    model = model_fun().to(ctx.device)
    model.fit(
        fold_data.X_train,
        fold_data.y_train,
        epochs=max(epochs, 1),
        lr=lr,
        wd=wd,
        optimizer=ctx.optimizer,
        batch_size=ctx.batch_size,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        should_stop=should_stop,
    )
    return model
