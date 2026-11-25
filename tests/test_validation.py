import numpy as np
import pandas as pd
import pytest

from imputeaudit.data.loader import DatasetValidationError, read_csv_bytes, validate_dataset


def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    df["y"] = ["p", "q"] * 50
    return df


def test_missing_target():
    with pytest.raises(DatasetValidationError, match="was not found"):
        validate_dataset(frame(), "nope")


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.assign(c="text"), "Non-numeric"),
    (lambda d: d.assign(a=np.r_[np.inf, d.a.values[1:]]), "infinite"),
    (lambda d: d.assign(a=np.r_[np.nan, d.a.values[1:]]), "already contain missing"),
    (lambda d: d.assign(b=1.0), "Constant"),
    (lambda d: d.assign(y=["p"] * 100), "at least two classes"),
])
def test_each_problem_has_its_own_message(mutate, message):
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(mutate(frame()), "y")


def test_duplicate_columns():
    df = frame()
    df.columns = ["a", "a", "c", "y"]
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validate_dataset(df, "y")


def test_bad_files():
    with pytest.raises(DatasetValidationError, match="empty"):
        read_csv_bytes(b"")
    with pytest.raises(DatasetValidationError, match="upload limit"):
        read_csv_bytes(b"a,b\n" * 10, max_mb=0)
    with pytest.raises(DatasetValidationError, match="UTF-8"):
        read_csv_bytes("a,b\n\xe9,1\n".encode("latin-1") + b"\xff\xfe")


def test_valid_dataset():
    ds = validate_dataset(frame(), "y")
    assert ds.X.shape == (100, 3) and ds.class_names == ["p", "q"]
