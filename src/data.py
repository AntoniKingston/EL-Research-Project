import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

CATEGORICAL_DTYPES = ["category", "object", "bool"]
UNKNOWN_CATEGORY = "<UNK>"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    is_classification: bool
    categorical: list[str]
    numerical: list[str]


@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
    preprocessor: "FoldPreprocessor"


def read_links(path: str | Path) -> dict[str, str]:
    links = pd.read_csv(path, skipinitialspace=True)
    return dict(zip(links["dataset_name"], links["link"], strict=True))


def parse_openml_id(link: str | int) -> int:
    """Accepts a bare ID, `https://www.openml.org/d/<id>` or `...search?type=data&id=<id>`."""
    match = re.search(r"(?:/d/|[?&]id=|^)(\d+)\s*$", str(link).strip())
    if match is None:
        raise ValueError(f"Cannot parse OpenML dataset id from: {link}")
    return int(match.group(1))


def download_dataset(name: str, link: str, data_dir: str | Path, force: bool = False) -> Path:
    """Downloads an OpenML dataset to `data_dir/raw/<name>.parquet` (+ `.json` metadata).

    Skipped if both files already exist, unless `force`.
    """
    raw_dir = Path(data_dir) / "raw"
    frame_path, meta_path = raw_dir / f"{name}.parquet", raw_dir / f"{name}.json"
    if frame_path.exists() and meta_path.exists() and not force:
        return frame_path

    openml_id = parse_openml_id(link)
    # The default pandas ARFF parser crashes on integer-valued nominal attributes (e.g. diamonds).
    bunch = fetch_openml(
        data_id=openml_id,
        as_frame=True,
        data_home=Path(data_dir) / "openml",
        parser="liac-arff",
    )
    target = bunch.target_names[0]
    frame = bunch.frame
    features = frame.drop(columns=target)
    is_classification = isinstance(frame[target].dtype, pd.CategoricalDtype) or frame[
        target
    ].dtype in (object, bool)

    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(frame_path)
    meta = {
        "openml_id": openml_id,
        "target": target,
        "is_classification": bool(is_classification),
        "categorical": features.select_dtypes(include=CATEGORICAL_DTYPES).columns.tolist(),
        "n_samples": len(frame),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return frame_path


def load_dataset(
    name: str,
    link: str,
    data_dir: str | Path,
    max_samples: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Loads a dataset, downloading it first if needed.

    If `max_samples` is set and smaller than the dataset, a seeded subsample of that size is
    drawn (stratified for classification); the cached file always keeps the full data.
    Class labels are mapped to 0..K-1 globally (a fixed relabeling, not a fitted statistic);
    features are returned raw and must go through `prepare_fold`.
    """
    frame_path = download_dataset(name, link, data_dir)
    meta = json.loads(frame_path.with_suffix(".json").read_text())
    frame = pd.read_parquet(frame_path)
    if max_samples is not None and len(frame) > max_samples:
        stratify = frame[meta["target"]] if meta["is_classification"] else None
        frame, _ = train_test_split(
            frame, train_size=max_samples, random_state=seed, stratify=stratify
        )
        frame = frame.reset_index(drop=True)

    X = frame.drop(columns=meta["target"])
    categorical = meta["categorical"]
    numerical = [c for c in X.columns if c not in categorical]
    X[categorical] = X[categorical].astype(object)

    y_raw = frame[meta["target"]]
    if meta["is_classification"]:
        y = LabelEncoder().fit_transform(y_raw.astype(str)).astype(np.int64)
    else:
        y = y_raw.to_numpy(dtype=np.float32)

    return Dataset(name, X, y, meta["is_classification"], categorical, numerical)


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
