import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from src.dataset_splits import DatasetSplits


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
