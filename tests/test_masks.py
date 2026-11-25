import numpy as np
import pytest
from scipy.special import expit

from imputeaudit.missingness.calibration import calibrate_intercept
from imputeaudit.missingness.generators import choose_anchor, make_missingness_mask

X = np.random.default_rng(1).normal(size=(10_000, 10))


def test_mcar_rate():
    m = make_missingness_mask(X, "MCAR", 0.2, seed=3)
    assert abs(m.achieved_rate - 0.2) < 0.01


def test_mar_anchor_observed_and_rate():
    m = make_missingness_mask(X, "MAR", 0.4, seed=3, anchor=4)
    assert not m.mask[:, 4].any()
    assert abs(m.achieved_rate - 0.4) < 0.02


def test_mar_depends_on_anchor():
    m = make_missingness_mask(X, "MAR", 0.3, seed=3, anchor=4)
    missing_rows = m.mask[:, 0]
    assert X[missing_rows, 4].mean() > X[~missing_rows, 4].mean() + 0.3


def test_mnar_hides_high_values():
    m = make_missingness_mask(X, "MNAR", 0.3, seed=3)
    assert X[m.mask].mean() > X[~m.mask].mean() + 0.3
    assert abs(m.achieved_rate - 0.3) < 0.02


def test_calibration_exact():
    z = np.random.default_rng(2).normal(size=5000)
    a = calibrate_intercept(z, 1.5, 0.37)
    assert abs(expit(a + 1.5 * z).mean() - 0.37) < 1e-6


def test_same_seed_same_mask():
    a = make_missingness_mask(X, "MNAR", 0.2, seed=11)
    b = make_missingness_mask(X, "MNAR", 0.2, seed=11)
    assert np.array_equal(a.mask, b.mask) and a.metadata() == b.metadata()


def test_rejects_incomplete_input_and_unknown_mechanism():
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        make_missingness_mask(bad, "MCAR", 0.2, 0)
    with pytest.raises(ValueError):
        make_missingness_mask(X, "XYZ", 0.2, 0)
    with pytest.raises(ValueError):
        make_missingness_mask(X, "MAR", 0.2, 0)


def test_anchor_is_most_connected():
    Z = np.random.default_rng(0).normal(size=(2000, 4))
    Z[:, 2] = Z[:, 0] + Z[:, 1] + Z[:, 3]
    assert choose_anchor(Z) == 2
