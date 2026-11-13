from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .domain import Dataset

UNKNOWN_CATEGORY = "<UNK>"


@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    preprocessor: "FoldPreprocessor"

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


class FoldPreprocessor:
    """Fitted on a training partition only, then applied to any evaluation partition.

    Numerical: median imputation + standardization. Categorical: `<UNK>` imputation +
    one-hot (categories unseen in training map to all zeros). Regression targets are
    standardized with training statistics; classification labels pass through.
    """

    def __init__(self, dataset: Dataset):
        self.is_classification = dataset.is_classification
        numerical = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=UNKNOWN_CATEGORY)),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                ),
            ]
        )
        self.features = ColumnTransformer(
            [("num", numerical, dataset.numerical), ("cat", categorical, dataset.categorical)],
            sparse_threshold=0.0,
        )
        self.y_mean, self.y_std = 0.0, 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FoldPreprocessor":
        self.features.fit(X)
        if not self.is_classification:
            self.y_mean, self.y_std = float(y.mean()), float(y.std()) or 1.0
        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.features.transform(X), dtype=np.float32)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.is_classification:
            return y
        return ((y - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        if self.is_classification:
            return y
        return y * self.y_std + self.y_mean


def prepare_fold(dataset: Dataset, train_idx: np.ndarray, eval_idx: np.ndarray) -> FoldData:
    """Fits preprocessing on `train_idx` and applies it to both partitions.

    HPO: (inner_train_idx, inner_val_idx). Final evaluation: (outer_train_idx, outer_test_idx).
    """
    X_train, y_train = dataset.X.iloc[train_idx], dataset.y[train_idx]
    X_eval, y_eval = dataset.X.iloc[eval_idx], dataset.y[eval_idx]
    preprocessor = FoldPreprocessor(dataset).fit(X_train, y_train)
    return FoldData(
        X_train=preprocessor.transform_features(X_train),
        y_train=preprocessor.transform_target(y_train),
        X_eval=preprocessor.transform_features(X_eval),
        y_eval=preprocessor.transform_target(y_eval),
        preprocessor=preprocessor,
    )
