import numpy as np
import torch
from src.dataset_tools.dataset_service import DatasetService
from src.dataset_tools.domain import DatasetSplits


def fold_score(
    model_fun,
    X,
    y,
    fold: DatasetSplits,
    epochs: int,
    patience: float,
    lr: float,
    wd: float,
    optimizer,
    batch_size: int,
    device,
):
    """`loss_fn(predictions, y_val)` over one fold (lower is better).

    Predictions are raw model outputs (logits for classification), targets are class labels
    for classification (integer y) or values for regression, as in standard PyTorch losses.
    """
    model = model_fun()
    model.to(device)
    X_train = X[fold.inner_train_idx]
    y_train = y[fold.inner_train_idx]
    X_val = X[fold.inner_val_idx]
    y_val = y[fold.inner_val_idx]

    # Normalize fold data strictly using statistics from X_train to prevent data leakage
    X_train, X_val = DatasetService.normalize_data(X_train, X_val)

    model.fit(X_train, y_train, X_val, y_val, epochs, patience, lr, wd, optimizer, batch_size)
    y_pred = torch.as_tensor(model.predict(X_val))
    y_true = torch.as_tensor(y_val)
    if not np.issubdtype(y_val.dtype, np.integer):
        y_true = y_true.to(y_pred.dtype)
    with torch.no_grad():
        score = model.task_loss(y_pred, y_true).item()
    return score


def train_model(
    model_fun,
    X,
    Y,
    fold: DatasetSplits,
    epochs: int,
    patience: float,
    lr: float,
    wd: float,
    optimizer,
    batch_size: int,
    device,
):
    """Trains a model on the full outer training partition of a fold."""
    model = model_fun()
    model.to(device)

    X_train = X[fold.outer_train_idx]
    Y_train = Y[fold.outer_train_idx]
    X_val = X[fold.outer_test_idx]
    Y_val = Y[fold.outer_test_idx]

    # Normalize fold data strictly using statistics from X_train to prevent data leakage
    X_train, X_val = DatasetService.normalize_data(X_train, X_val)

    model.fit(X_train, Y_train, X_val, Y_val, epochs, patience, lr, wd, optimizer, batch_size)

    return model
