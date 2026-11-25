import numpy as np
import pytest

from imputeaudit.services.experiment_service import ExperimentConfig, ExperimentService, prepare
from imputeaudit.services.report_service import ReportService
from imputeaudit.storage.repository import Repository


def test_unit_runs_all_imputers_and_models(toy_dataset, small_settings):
    prep = prepare(toy_dataset, small_settings)
    rows = ExperimentService(small_settings).run_unit(
        prep, ExperimentConfig("toy", "MAR", 0.4, 1, ("median", "knn", "iterative"), ("random_forest", "xgboost")))
    assert len(rows) == 6
    for r in rows:
        assert 0 <= r["macro_f1"] <= 1 and -1 <= r["global_tau"] <= 1
        assert len(r["class_tau"]) == 3
        assert abs(r["achieved_rate_train"] - 0.4) < 0.05


def test_same_config_reproduces(toy_dataset, small_settings):
    cfg = ExperimentConfig("toy", "MNAR", 0.2, 3, ("iterative",))
    a = ExperimentService(small_settings).run_unit(prepare(toy_dataset, small_settings), cfg)[0]
    b = ExperimentService(small_settings).run_unit(prepare(toy_dataset, small_settings), cfg)[0]
    for key in ("macro_f1", "global_tau", "mean_class_tau", "nrmse"):
        assert a[key] == pytest.approx(b[key])
    assert cfg.config_id() == ExperimentConfig("toy", "MNAR", 0.2, 3, ("iterative",)).config_id()


def test_reference_rows_shared(toy_dataset, small_settings):
    service = ExperimentService(small_settings)
    prep = prepare(toy_dataset, small_settings)
    ids_before = prep.eval_pos.copy()
    service.run_unit(prep, ExperimentConfig("toy", "MCAR", 0.2, 0, ("median",)))
    service.run_unit(prep, ExperimentConfig("toy", "MNAR", 0.4, 0, ("knn",)))
    assert np.array_equal(ids_before, prep.eval_pos)
    assert list(prep.references) == [("random_forest", 0)]


def test_binary_metrics(toy_frame, small_settings):
    from imputeaudit.data.loader import validate_dataset
    frame = toy_frame.assign(label=np.where(toy_frame.label == "a", "yes", "no"))
    ds = validate_dataset(frame, "label", "bin")
    ds.meta["split"] = "stratified"
    row = ExperimentService(small_settings).run_unit(prepare(ds, small_settings),
                                                     ExperimentConfig("bin", "MCAR", 0.2, 0, ("median",)))[0]
    assert row["pr_auc"] is not None and row["mcc"] is not None


def test_repository_resume_and_export(tmp_path, toy_dataset, small_settings):
    repo = Repository(tmp_path)
    service = ExperimentService(small_settings)
    prep = prepare(toy_dataset, small_settings)
    cfg = ExperimentConfig("toy", "MCAR", 0.2, 0, ("median",))
    repo.start_experiment(cfg.config_id(), "toy", cfg.to_dict(), {})
    repo.save_results(cfg.config_id(), service.run_unit(prep, cfg))
    assert repo.exists(cfg.config_id())
    repo.start_experiment("broken", "toy", {}, {})
    repo.log_failure("broken", "ValueError: boom")
    assert not repo.exists("broken")
    assert repo.count_completed() == 1
    frame = repo.results_frame()
    assert len(frame) == 1
    bundle = ReportService(repo).export_bundle(frame, cfg.to_dict())
    import io, zipfile
    names = zipfile.ZipFile(io.BytesIO(bundle)).namelist()
    assert {"results.csv", "summary.csv", "config.json", "report.html"} <= set(names)
