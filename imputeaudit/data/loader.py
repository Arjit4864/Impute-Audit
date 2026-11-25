"""CSV ingestion and schema validation."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from imputeaudit import config

SPLIT_COLUMN = "official_split"


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used. ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class ValidatedDataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    target: str
    file_hash: str
    class_names: list[str]
    official_split: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_csv_bytes(raw: bytes, max_mb: int = config.MAX_UPLOAD_MB) -> pd.DataFrame:
    if not raw:
        raise DatasetValidationError(["The file is empty."])
    if len(raw) > max_mb * 1024 * 1024:
        raise DatasetValidationError([f"The file is larger than the {max_mb} MB upload limit."])
    try:
        return pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError as error:
        raise DatasetValidationError(["The file is not UTF-8 text. Save it as a UTF-8 CSV and upload it again."]) from error
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise DatasetValidationError([f"The file could not be parsed as CSV: {str(error)[:200]}"]) from error


def validate_dataset(frame: pd.DataFrame, target: str, name: str = "uploaded",
                     raw_hash: str = "", max_rows: int = config.MAX_ROWS,
                     min_rows_per_class: int = config.MIN_ROWS_PER_CLASS,
                     allow_missing: bool = False) -> ValidatedDataset:
    problems: list[str] = []
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        problems.append(f"Duplicate column names: {', '.join(map(str, dupes))}.")
    if not target or target not in frame.columns:
        problems.append(f"Target column '{target}' was not found. Choose the target explicitly.")
        raise DatasetValidationError(problems)
    if len(frame) > max_rows:
        problems.append(f"The dataset has {len(frame):,} rows; the limit is {max_rows:,}.")

    split = None
    if SPLIT_COLUMN in frame.columns:
        split = frame[SPLIT_COLUMN].astype(str).to_numpy()
        if not set(np.unique(split)) <= {"train", "test"}:
            problems.append(f"Column '{SPLIT_COLUMN}' may only contain 'train' and 'test'.")
    predictors = frame.drop(columns=[c for c in (target, SPLIT_COLUMN) if c in frame.columns])
    if predictors.shape[1] < 2:
        problems.append("At least two predictor columns are required.")
    non_numeric = [c for c in predictors.columns if not pd.api.types.is_numeric_dtype(predictors[c])]
    if non_numeric:
        problems.append(f"Non-numeric predictors are not supported: {', '.join(map(str, non_numeric[:10]))}.")
    y = frame[target]
    if y.isna().any():
        problems.append("The target column contains missing values.")
    if not non_numeric and predictors.shape[1]:
        values = predictors.to_numpy(dtype=float)
        if np.isinf(values).any():
            problems.append("Predictors contain infinite values.")
        if not allow_missing and np.isnan(values).any():
            problems.append("Predictors already contain missing values. ImputeAudit simulates missingness on complete data.")
        constant = [c for c in predictors.columns if predictors[c].nunique(dropna=True) <= 1]
        if constant:
            problems.append(f"Constant predictors carry no information: {', '.join(map(str, constant[:10]))}.")
    counts = y.value_counts()
    if len(counts) < 2:
        problems.append("The target must have at least two classes.")
    small = counts[counts < min_rows_per_class]
    if len(small):
        problems.append(f"Classes with fewer than {min_rows_per_class} rows: {', '.join(map(str, small.index[:10]))}.")
    if problems:
        raise DatasetValidationError(problems)

    class_names = sorted(map(str, counts.index))
    return ValidatedDataset(
        name=name, X=predictors.astype(float).reset_index(drop=True),
        y=y.astype(str).reset_index(drop=True), target=target, file_hash=raw_hash,
        class_names=class_names, official_split=split,
    )


def load_dataset(path: str | Path, target: str, name: str | None = None) -> ValidatedDataset:
    path = Path(path)
    raw = path.read_bytes()
    frame = read_csv_bytes(raw, max_mb=max(config.MAX_UPLOAD_MB, len(raw) // (1024 * 1024) + 1))
    return validate_dataset(frame, target, name or path.stem, file_hash(raw))


def load_bundled(name: str) -> ValidatedDataset:
    spec = config.DATASETS[name]
    dataset = load_dataset(config.DATA_DIR / spec.file, spec.target, name)
    dataset.meta.update({"title": spec.title, "domain": spec.domain, "split": spec.split})
    return dataset
