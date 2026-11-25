from imputeaudit.services.experiment_service import ExperimentConfig, ExperimentService, prepare


def test_target_never_in_predictors(toy_dataset):
    assert "label" not in toy_dataset.X.columns
    assert toy_dataset.X.shape[1] == 6


def test_mask_shape_matches_predictors_only(toy_dataset, small_settings):
    prep = prepare(toy_dataset, small_settings)
    assert prep.X_train.shape[1] == toy_dataset.X.shape[1]
    rows = ExperimentService(small_settings).run_unit(prep, ExperimentConfig("toy", "MCAR", 0.2, 0, ("median",)))
    assert rows[0]["achieved_rate_train"] > 0
    assert set(prep.y_train) == {0, 1, 2}
