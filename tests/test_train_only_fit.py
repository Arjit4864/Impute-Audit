import numpy as np
import pytest

from imputeaudit.pipeline.preprocess import fit_preprocessor, transform_test
from imputeaudit.pipeline.split import make_split, stratified_sample

rng = np.random.default_rng(0)
TRAIN = rng.normal(size=(500, 4))
TRAIN[rng.random(TRAIN.shape) < 0.2] = np.nan


@pytest.mark.parametrize("imputer", ["median", "knn", "iterative"])
def test_test_rows_do_not_change_fitted_state(imputer):
    pipe = fit_preprocessor(TRAIN, imputer, seed=0)
    medians_before = pipe.named_steps["scale"].mean_.copy()
    probe = np.full((3, 4), np.nan)
    probe[:, 0] = 1.0
    out_before = transform_test(pipe, probe)
    extreme = np.full((200, 4), 1e6)
    transform_test(pipe, extreme)
    assert np.allclose(pipe.named_steps["scale"].mean_, medians_before)
    assert np.allclose(transform_test(pipe, probe), out_before)


def test_fit_sees_only_training_rows():
    pipe = fit_preprocessor(TRAIN, "median", seed=0)
    assert pipe.named_steps["impute"].n_features_in_ == 4
    assert np.allclose(pipe.named_steps["scale"].mean_, np.nanmean(TRAIN, axis=0))


def test_split_reproducible():
    y = np.repeat(["a", "b", "c"], 100)
    a, b = make_split(y, "stratified", 7), make_split(y, "stratified", 7)
    assert np.array_equal(a.train_ids, b.train_ids) and np.array_equal(a.test_ids, b.test_ids)
    assert len(set(a.train_ids) & set(a.test_ids)) == 0


def test_official_split_preserved():
    official = np.array(["train"] * 70 + ["test"] * 30)
    s = make_split(np.zeros(100), "official", official=official)
    assert np.array_equal(s.test_ids, np.arange(70, 100))


def test_stratified_explanation_sample_covers_classes():
    y = np.array([0] * 500 + [1] * 30 + [2] * 8)
    pos = stratified_sample(y, 20, 0)
    counts = np.bincount(y[pos])
    assert counts.tolist() == [20, 20, 8]
