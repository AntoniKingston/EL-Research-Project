import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TrainingInterrupted(Exception):  # noqa: N818
    """Raised by `fit` when `should_stop()` fires; the epoch checkpoint has been saved."""


@dataclass
class _FitState:
    """Everything besides model/optimizer/RNG needed to resume `TabularModel.fit`."""

    epoch: int = 0
    best_loss: float = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs: int = 0
    best_epoch: int = 0
    loss_history: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "val_loss": []}
    )


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class TabularModel(nn.Module):
    """Shared training / inference loop.

    Subclasses set `self.output_dim` and implement `forward(x)` returning the raw model output
    (logits for classification). Models with additional regularization override
    `training_loss`; validation always monitors the plain task loss. A single output column
    is squeezed to shape (B,).
    """

    def __init__(self, is_classification: bool):
        super().__init__()
        self.is_classification = is_classification
        self.best_epoch = 0
        self.loss_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _format_output(self, out: torch.Tensor) -> torch.Tensor:
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def task_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.is_classification:
            return torch.nn.functional.cross_entropy(pred, target.long())
        return torch.nn.functional.mse_loss(pred.view_as(target), target.float())

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.task_loss(self._format_output(self(x)), y)

    def _to_tensors(self, X, y, device) -> tuple[torch.Tensor, torch.Tensor]:
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        y = np.asarray(y)
        y_dtype = torch.long if np.issubdtype(y.dtype, np.integer) else torch.float32
        y_t = torch.as_tensor(y, dtype=y_dtype, device=device)
        return X_t, y_t

    @torch.no_grad()
    def _predict_raw(self, X_t: torch.Tensor, batch_size: int) -> torch.Tensor:
        self.eval()
        outs = [self(X_t[i : i + batch_size]) for i in range(0, len(X_t), batch_size)]
        return self._format_output(torch.cat(outs))

    def fit(
        self,
        X_train,
        y_train,
        *,
        epochs: int,
        lr: float,
        wd: float,
        optimizer,
        batch_size: int,
        X_val=None,
        y_val=None,
        patience: int | None = None,
        on_epoch_end: Callable[[int, float], None] | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_every: int = 0,
        should_stop: Callable[[], bool] | None = None,
    ):
        """Mini-batch training.

        With (X_val, y_val): early stopping on the validation task loss (`patience`), the best
        weights are restored and the best epoch is stored in `best_epoch`.
        Without validation data: trains exactly `epochs` epochs and keeps the final weights.

        `on_epoch_end(epoch, monitored_loss)` is called after every epoch (may raise, e.g. to
        prune an Optuna trial). With `checkpoint_path`, training state (incl. RNG state) is
        saved every `checkpoint_every` epochs and training resumes from it if the file exists;
        the file is removed once training completes. If `should_stop()` returns True after an
        epoch, a checkpoint is saved and `TrainingInterrupted` is raised.
        """
        device = next(self.parameters()).device
        X_t, y_t = self._to_tensors(X_train, y_train, device)
        val = self._to_tensors(X_val, y_val, device) if X_val is not None else None
        opt = optimizer(self.parameters(), lr=lr, weight_decay=wd)

        n = len(X_t)
        # Drop the incomplete last batch: BatchNorm and batch-wise similarity
        # penalties are unstable on tiny batches.
        n_batches = max(n // batch_size, 1)
        state = _FitState()

        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        if ckpt is not None and ckpt.exists():
            saved = torch.load(ckpt, map_location=device, weights_only=False)
            self.load_state_dict(saved["model"])
            opt.load_state_dict(saved["optimizer"])
            state = _FitState(**saved["state"])
            _restore_rng_state(saved["rng"])

        def save_checkpoint() -> None:
            if ckpt is None:
                return
            payload = {
                "model": self.state_dict(),
                "optimizer": opt.state_dict(),
                "state": asdict(state),
                "rng": _capture_rng_state(),
            }
            atomic_torch_save(payload, ckpt)

        while state.epoch < epochs:
            epoch = state.epoch
            self.train()
            perm = torch.randperm(n, device=device)
            train_loss = 0.0
            for i in range(n_batches):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                opt.zero_grad()
                loss = self.training_loss(X_t[idx], y_t[idx])
                loss.backward()
                opt.step()
                train_loss += loss.item()

            mean_train_loss = train_loss / n_batches
            state.loss_history["train_loss"].append(mean_train_loss)
            state.epoch = epoch + 1

            if val is not None:
                monitored = self.task_loss(self._predict_raw(val[0], batch_size), val[1]).item()
                state.loss_history["val_loss"].append(monitored)
                if monitored < state.best_loss:
                    state.best_loss, state.bad_epochs = monitored, 0
                    state.best_epoch = epoch + 1
                    state.best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                else:
                    state.bad_epochs += 1
            else:
                monitored = mean_train_loss
                state.best_epoch = epoch + 1

            if on_epoch_end is not None:
                on_epoch_end(epoch, monitored)

            if val is not None and patience is not None and state.bad_epochs >= patience:
                break
            if should_stop is not None and should_stop():
                save_checkpoint()
                raise TrainingInterrupted(f"Stopped after epoch {epoch + 1}")
            if checkpoint_every and state.epoch % checkpoint_every == 0:
                save_checkpoint()

        if val is not None and state.best_state is not None:
            self.load_state_dict(state.best_state)
        self.best_epoch = state.best_epoch
        self.loss_history = state.loss_history
        if ckpt is not None:
            ckpt.unlink(missing_ok=True)
        return self

    def predict(self, X, batch_size: int = 1024) -> np.ndarray:
        device = next(self.parameters()).device
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        return self._predict_raw(X_t, batch_size).cpu().numpy()
