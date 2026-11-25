"""Orchestration of leakage-safe experiment units.

One *unit* is (dataset, mechanism, rate, seed). Inside a unit the masks are drawn once and
shared by every imputer and model, so imputers are compared on identical missing cells.
The complete-data reference for (dataset, model, seed) uses the same split and the same
fixed explanation rows as every imputed condition.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from imputeaudit.config import AppSettings
from imputeaudit.data.loader import ValidatedDataset
from imputeaudit.evaluation.imputation_metrics import nrmse
from imputeaudit.evaluation.prediction_metrics import evaluate_model
from imputeaudit.explainability.rankings import rank_features
from imputeaudit.explainability.shap_engine import compute_shap
from imputeaudit.explainability.stability import compare_rankings
from imputeaudit.missingness.generators import choose_anchor, make_missingness_mask
from imputeaudit.pipeline.preprocess import fit_preprocessor, transform_test
from imputeaudit.pipeline.split import make_split, stratified_sample
from imputeaudit.pipeline.train import fit_model

RETRAIN_SEED_OFFSET = 1000
TEST_MASK_SEED_OFFSET = 50_000


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    mechanism: str
    rate: float
    seed: int
    imputers: tuple = ("median", "knn", "iterative")
    models: tuple = ("random_forest",)

    def validate(self) -> None:
        if self.mechanism not in ("MCAR", "MAR", "MNAR"):
            raise ValueError(f"Unsupported mechanism {self.mechanism}")
        if not 0 < self.rate < 1:
            raise ValueError("Missingness rate must be between 0 and 1")
        if not self.imputers or not self.models:
            raise ValueError("Choose at least one imputer and one model")

    def config_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedDataset:
    dataset: ValidatedDataset
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int
    train_mean: np.ndarray
    train_std: np.ndarray
    anchor: int
    eval_pos: np.ndarray
    clean_scaler: StandardScaler
    references: dict = field(default_factory=dict)


def prepare(dataset: ValidatedDataset, settings: AppSettings) -> PreparedDataset:
    split_kind = dataset.meta.get("split", "stratified")
    if split_kind == "official" and dataset.official_split is None:
        split_kind = "stratified"
    split = make_split(dataset.y, split_kind, settings.split_seed, settings.test_size, dataset.official_split)
    encoder = LabelEncoder().fit(dataset.y.iloc[split.train_ids])
    X = dataset.X.to_numpy(dtype=float)
    X_train, X_test = X[split.train_ids], X[split.test_ids]
    y_train = encoder.transform(dataset.y.iloc[split.train_ids])
    y_test = encoder.transform(dataset.y.iloc[split.test_ids])
    n_classes = len(encoder.classes_)
    per_class = max(settings.min_explain_rows_per_class, settings.explain_rows_total // n_classes)
    eval_pos = stratified_sample(y_test, per_class, settings.split_seed)
    return PreparedDataset(dataset, X_train, X_test, y_train, y_test, len(encoder.classes_),
                           X_train.mean(axis=0), X_train.std(axis=0), choose_anchor(X_train), eval_pos,
                           StandardScaler().fit(X_train))


class ExperimentService:
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings()

    def reference(self, prep: PreparedDataset, model: str, seed: int) -> dict:
        key = (model, seed)
        if key not in prep.references:
            Xtr = prep.clean_scaler.transform(prep.X_train)
            Xte = prep.clean_scaler.transform(prep.X_test)
            fitted = fit_model(model, Xtr, prep.y_train, seed, prep.n_classes, self.settings.models)
            phi = compute_shap(fitted, Xte[prep.eval_pos], prep.n_classes)
            prep.references[key] = {"model": fitted, "rankings": rank_features(phi),
                                    "metrics": evaluate_model(fitted, Xte, prep.y_test, prep.n_classes)}
        return prep.references[key]

    def clean_retrain_row(self, prep: PreparedDataset, model: str, seed: int) -> dict:
        """Noise floor: a second complete-data model with a different seed against the reference."""
        start = time.perf_counter()
        ref = self.reference(prep, model, seed)
        Xtr = prep.clean_scaler.transform(prep.X_train)
        Xte = prep.clean_scaler.transform(prep.X_test)
        other = fit_model(model, Xtr, prep.y_train, seed + RETRAIN_SEED_OFFSET, prep.n_classes, self.settings.models)
        stab = compare_rankings(ref["rankings"], rank_features(compute_shap(other, Xte[prep.eval_pos], prep.n_classes)),
                                self.settings.top_k)
        metrics = evaluate_model(other, Xte, prep.y_test, prep.n_classes)
        return self._row(prep, model, "none", 0.0, seed, "clean_retrain", None, None, 0.0, metrics, ref, stab,
                         None, time.perf_counter() - start)

    def run_unit(self, prep: PreparedDataset, cfg: ExperimentConfig, model_only: bool = True) -> list[dict]:
        cfg.validate()
        s = self.settings
        anchor = prep.anchor if cfg.mechanism == "MAR" else None
        common = dict(anchor=anchor, slope=s.slope, ref_mean=prep.train_mean, ref_std=prep.train_std)
        m_train = make_missingness_mask(prep.X_train, cfg.mechanism, cfg.rate, cfg.seed, **common)
        m_test = make_missingness_mask(prep.X_test, cfg.mechanism, cfg.rate, cfg.seed + TEST_MASK_SEED_OFFSET, **common)
        Xtr_masked, Xte_masked = m_train.apply(prep.X_train), m_test.apply(prep.X_test)

        rows = []
        for imputer in cfg.imputers:
            start = time.perf_counter()
            pipe = fit_preprocessor(Xtr_masked, imputer, cfg.seed, s.knn_neighbors, s.iterative_max_iter)
            Xtr_imp = pipe.transform(Xtr_masked)
            Xte_imp = transform_test(pipe, Xte_masked)
            scaler = pipe.named_steps["scale"]
            error = nrmse(scaler.transform(prep.X_test), Xte_imp, m_test.mask)
            impute_time = time.perf_counter() - start
            for model in cfg.models:
                t0 = time.perf_counter()
                ref = self.reference(prep, model, cfg.seed)
                fitted = fit_model(model, Xtr_imp, prep.y_train, cfg.seed, prep.n_classes, s.models)
                metrics = evaluate_model(fitted, Xte_imp, prep.y_test, prep.n_classes)
                phi = compute_shap(fitted, Xte_imp[prep.eval_pos], prep.n_classes)
                stab = compare_rankings(ref["rankings"], rank_features(phi), s.top_k)
                model_only_stab = None
                if model_only:
                    phi_clean = compute_shap(fitted, scaler.transform(prep.X_test)[prep.eval_pos], prep.n_classes)
                    model_only_stab = compare_rankings(ref["rankings"], rank_features(phi_clean), s.top_k)
                rows.append(self._row(prep, model, cfg.mechanism, cfg.rate, cfg.seed, imputer, m_train, m_test, error,
                                      metrics, ref, stab, model_only_stab, impute_time + time.perf_counter() - t0))
        return rows

    @staticmethod
    def _row(prep, model, mechanism, rate, seed, imputer, m_train, m_test, error, metrics, ref, stab,
             model_only_stab, runtime) -> dict:
        return {
            "dataset": prep.dataset.name, "model": model, "mechanism": mechanism, "rate": rate, "seed": seed,
            "imputer": imputer,
            "achieved_rate_train": m_train.achieved_rate if m_train else 0.0,
            "achieved_rate_test": m_test.achieved_rate if m_test else 0.0,
            "nrmse": error, "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "ref_macro_f1": ref["metrics"]["macro_f1"], "pr_auc": metrics.get("pr_auc"), "mcc": metrics.get("mcc"),
            **{k: stab[k] for k in ("global_tau", "mean_class_tau", "min_class_tau", "delta_tau", "global_top_k",
                                    "mean_class_top_k", "min_class_top_k", "class_tau", "class_top_k")},
            "model_only_global_tau": model_only_stab["global_tau"] if model_only_stab else None,
            "model_only_mean_class_tau": model_only_stab["mean_class_tau"] if model_only_stab else None,
            "class_recall": metrics["class_recall"], "runtime_s": round(runtime, 3),
        }
