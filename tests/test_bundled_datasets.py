import numpy as np
import pytest

from imputeaudit import config
from imputeaudit.data.loader import load_bundled
from imputeaudit.data.profiler import profile
from imputeaudit.pipeline.split import make_split

EXPECTED = {"dry_bean": (13611, 16, 7), "pendigits": (10992, 16, 10), "satimage": (6435, 36, 6),
            "letter": (20000, 16, 26), "htru2": (17898, 8, 2)}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_bundled_profile(name):
    if not (config.DATA_DIR / config.DATASETS[name].file).exists():
        pytest.skip("run scripts/prepare_datasets.py first")
    ds = load_bundled(name)
    p = profile(ds)
    assert (p["rows"], p["features"], p["classes"]) == EXPECTED[name]
    assert p["missing_cells"] == 0
    assert ds.target not in ds.X.columns


def test_pendigits_official_split_sizes():
    if not (config.DATA_DIR / "pendigits.csv").exists():
        pytest.skip("run scripts/prepare_datasets.py first")
    ds = load_bundled("pendigits")
    s = make_split(ds.y, "official", official=ds.official_split)
    assert (len(s.train_ids), len(s.test_ids)) == (7494, 3498)
    assert np.all(ds.official_split[s.test_ids] == "test")
