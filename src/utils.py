from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


@dataclass
class DatasetSplits:
    fold: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray


def get_nested_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    is_classification: bool = True,
    seed: int = 42,
) -> list[DatasetSplits]:
    """Generates deterministic outer and inner splits to be reused across all models."""
    if is_classification:
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        y_outer_train = y[outer_train_idx] if is_classification else None
        # Split 80% outer train into 80% inner_train and 20% inner_val
        inner_train_rel, inner_val_rel = train_test_split(
            np.arange(len(outer_train_idx)),
            test_size=0.20,
            shuffle=True,
            random_state=seed,
            stratify=y_outer_train,
        )
        inner_train_idx = outer_train_idx[inner_train_rel]
        inner_val_idx = outer_train_idx[inner_val_rel]
        splits.append(
            DatasetSplits(
                fold=fold,
                outer_train_idx=outer_train_idx,
                outer_test_idx=outer_test_idx,
                inner_train_idx=inner_train_idx,
                inner_val_idx=inner_val_idx,
            )
        )
    return splits


def mean_score_by_folds(
    model_fun,
    X,
    y,
    folds: list[DatasetSplits],
    epochs,
    patience,
    lr,
    wd,
    optimizer,
    loss_fn,
    batch_size,
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
