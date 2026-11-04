import numpy as np
import torch
from src.dataset_tools.domain import DatasetSplits 

def mean_score_by_folds(
    model_fun,
    X,
    y,
    folds: list[DatasetSplits],
    epochs: int,
    patience: float,
    lr: float,
    wd: float,
    optimizer,
    loss_fn,
    batch_size: int,
    device,
):
    """Mean of `loss_fn(predictions, y_val)` over folds (lower is better).

    Predictions are raw model outputs (logits for classification), targets are class labels
    for classification (integer y) or values for regression, as in standard PyTorch losses.
    """
    scores = np.array([])
    for fold in folds:
        model = model_fun()
        model.to(device)
        X_train = X[fold.inner_train_idx]
        y_train = y[fold.inner_train_idx]
        X_val = X[fold.inner_val_idx]
        y_val = y[fold.inner_val_idx]
        model.fit(X_train, y_train, epochs, patience, lr, wd, optimizer, loss_fn, batch_size)
        y_pred = torch.as_tensor(model.predict(X_val))
        y_true = torch.as_tensor(y_val)
        if not np.issubdtype(y_val.dtype, np.integer):
            y_true = y_true.to(y_pred.dtype)
        with torch.no_grad():
            score = loss_fn(y_pred, y_true).item()
        scores = np.append(scores, score)
    return np.mean(scores)
