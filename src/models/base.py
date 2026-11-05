import math

import numpy as np
import torch
from torch import nn


class TabularModel(nn.Module):
    """Shared training / inference loop.

    Subclasses must set `self.output_dim` and implement `forward(x)` returning the raw model
    output. Models with additional regularization override `training_loss`; early stopping
    always monitors the plain task loss. A single output column is squeezed to shape (B,).
    """

    def __init__(self, is_classification: bool):
        super().__init__()
        self.is_classification = is_classification
        self.loss_history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _format_output(self, out: torch.Tensor) -> torch.Tensor:
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO implement
        # return loss_fn(self._format_output(self(x)), y)
        if self.is_classification:
            pass
        else:
            pass

    def training_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.task_loss(x, y)

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
        X_val,
        y_val,
        epochs,
        patience,
        lr,
        wd,
        optimizer,
        batch_size,
    ):
        """Mini-batch training with early stopping.

        Monitors task loss on (X_val, y_val) when given, otherwise the mean training loss.
        The best weights are restored at the end and the best epoch is stored in `best_epoch`.
        """
        device = next(self.parameters()).device
        X_t, y_t = self._to_tensors(X_train, y_train, device)
        val = self._to_tensors(X_val, y_val, device) if X_val is not None else None
        opt = optimizer(self.parameters(), lr=lr, weight_decay=wd)

        n = len(X_t)
        # Drop the incomplete last batch: BatchNorm and batch-wise similarity
        # penalties are unstable on tiny batches.
        n_batches = max(n // batch_size, 1)
        best_loss, best_state, bad_epochs = math.inf, None, 0
        self.best_epoch = 0
        self.loss_history = {"train_loss": [], "val_loss": []}
        for epoch in range(epochs):
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
            self.loss_history["train_loss"].append(mean_train_loss)

            if val is not None:
                monitored = self.task_loss(self._predict_raw(val[0], batch_size), val[1]).item()
                self.loss_history["val_loss"].append(monitored)
            else:
                monitored = mean_train_loss

            if monitored < best_loss:
                best_loss, bad_epochs, self.best_epoch = monitored, 0, epoch + 1
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    def predict(self, X, batch_size: int = 1024) -> np.ndarray:
        device = next(self.parameters()).device
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        return self._predict_raw(X_t, batch_size).cpu().numpy()
