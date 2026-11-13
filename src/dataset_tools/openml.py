import json
import re
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

CATEGORICAL_DTYPES = ["category", "object", "bool"]


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
    target_dtype = frame[target].dtype
    is_classification = isinstance(target_dtype, pd.CategoricalDtype) or target_dtype in (
        object,
        bool,
    )

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


def load_openml_frame(name: str, link: str, data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Returns the cached raw frame (downloading it first if needed) and its metadata."""
    frame_path = download_dataset(name, link, data_dir)
    meta = json.loads(frame_path.with_suffix(".json").read_text())
    return pd.read_parquet(frame_path), meta
