"""ImputeAudit Streamlit interface.

The interface collects inputs and displays outputs only. All experiment logic lives in the
imputeaudit package; interactive runs use small settings, and the full matrix is read from
the offline runner's results.
"""
from __future__ import annotations

import io
import json
import traceback
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from imputeaudit import config
from imputeaudit.config import AppSettings, ModelSettings
from imputeaudit.data.loader import DatasetValidationError
from imputeaudit.logging_setup import get_logger
from imputeaudit.missingness.generators import make_missingness_mask
from imputeaudit.services.dataset_service import DatasetService
from imputeaudit.services.experiment_service import ExperimentConfig, ExperimentService, prepare
from imputeaudit.services.report_service import ReportService
from imputeaudit.storage.repository import Repository

warnings.filterwarnings("ignore")
log = get_logger("imputeaudit.app")
st.set_page_config(page_title="ImputeAudit", layout="wide")

PAGES = ["Dashboard", "Dataset Upload and Profiling", "Missingness Simulation", "Imputation Comparison",
         "Model Evaluation", "Global SHAP Analysis", "Class-Wise SHAP Analysis", "Results and Export"]
INTERACTIVE = AppSettings(explain_rows_total=150, min_explain_rows_per_class=8,
                          models=ModelSettings(rf_trees=60, rf_max_depth=10, xgb_trees=80, n_jobs=-1))
PRETTY = {"median": "Median", "knn": "KNN", "iterative": "Iterative", "clean_retrain": "Clean retrain"}


@st.cache_resource
def services():
    repo = Repository(config.RUNS_DIR)
    return repo, DatasetService(repo), ReportService(repo)


repo, dataset_service, report_service = services()


@st.cache_data(show_spinner=False)
def batch_results() -> pd.DataFrame:
    return report_service.load_results()


def current_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = dataset_service.load_bundled("dry_bean")
    return st.session_state.dataset


def card_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


with st.sidebar:
    st.title("ImputeAudit")
    st.caption("Leakage-safe missing-data imputation and explanation-stability audit")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    ds = current_dataset()
    st.divider()
    st.caption(f"Active dataset: **{ds.meta.get('title', ds.name)}**")


def page_dashboard():
    st.header("Dashboard")
    st.write("Simulate missing values on complete tabular data, impute them without leaking test information, "
             "train a classifier and check whether its global and class-wise SHAP feature rankings still agree "
             "with a complete-data reference.")
    res = batch_results()
    exps = repo.experiments_frame()
    card_row([("Bundled datasets", len(config.DATASETS)), ("Stored result rows", f"{len(res):,}"),
              ("Completed units", int((exps.status == "completed").sum()) if len(exps) else 0),
              ("Failed units", int((exps.status == "failed").sum()) if len(exps) else 0)])
    st.subheader("Workflow")
    st.markdown("1. **Upload and validate** a complete numeric CSV and choose the target.  \n"
                "2. **Freeze the split** (stratified, or the dataset's official split).  \n"
                "3. **Mask predictors** under MCAR, MAR or MNAR at a calibrated rate.  \n"
                "4. **Impute** with a pipeline fitted on training rows only.  \n"
                "5. **Train and evaluate** Random Forest or XGBoost.  \n"
                "6. **Explain** the same held-out rows with TreeSHAP, globally and per class.  \n"
                "7. **Compare** with the complete-data reference (Kendall tau-b, Top-5 Jaccard) and export.")
    if len(res):
        imp = res[res.imputer != "clean_retrain"]
        fig = px.box(imp, x="dataset", y=["global_tau", "mean_class_tau"], title="Batch study: ranking agreement by dataset")
        st.plotly_chart(fig, width="stretch")


def page_dataset():
    st.header("Dataset Upload and Profiling")
    left, right = st.columns([2, 1])
    with left:
        choice = st.selectbox("Bundled benchmark", list(config.DATASETS),
                              format_func=lambda n: config.DATASETS[n].title)
        if st.button("Load bundled dataset"):
            st.session_state.dataset = dataset_service.load_bundled(choice)
            st.session_state.pop("last_run", None)
    with right:
        upload = st.file_uploader(f"Or upload a CSV (max {config.MAX_UPLOAD_MB} MB)", type=["csv"])
        if upload is not None:
            raw = upload.getvalue()
            header = pd.read_csv(io.BytesIO(raw), nrows=0).columns.tolist() if raw else []
            target = st.selectbox("Target column", header) if header else None
            if st.button("Validate upload"):
                try:
                    st.session_state.dataset = dataset_service.load_upload(raw, target, upload.name.rsplit(".", 1)[0])
                    st.session_state.pop("last_run", None)
                    st.success("Dataset accepted.")
                except DatasetValidationError as error:
                    for problem in error.problems:
                        st.error(problem)
                    log.info("upload rejected: %s", error)
    ds = current_dataset()
    prof = dataset_service.profile(ds)
    card_row([("Rows", f"{prof['rows']:,}"), ("Features", prof["features"]), ("Classes", prof["classes"]),
              ("Missing cells", prof["missing_cells"])])
    c1, c2 = st.columns(2)
    counts = pd.Series(prof["class_counts"]).sort_index()
    c1.plotly_chart(px.bar(x=counts.index.astype(str), y=counts.values, labels={"x": "class", "y": "rows"},
                           title=f"Class distribution (imbalance ratio {prof['imbalance_ratio']:.1f})"),
                    width="stretch")
    feature = c2.selectbox("Feature distribution", ds.X.columns)
    c2.plotly_chart(px.histogram(ds.X, x=feature, nbins=40), width="stretch")
    st.dataframe(prof["feature_summary"].round(3), width="stretch")


def page_missingness():
    st.header("Missingness Simulation")
    ds = current_dataset()
    c = st.columns(4)
    mech = c[0].selectbox("Mechanism", config.MECHANISMS, index=1)
    rate = c[1].select_slider("Rate", options=[0.1, 0.2, 0.3, 0.4, 0.5], value=0.2)
    seed = c[2].number_input("Mask seed", 0, 10_000, 1)
    X = ds.X.to_numpy(float)
    from imputeaudit.missingness.generators import choose_anchor
    anchor = choose_anchor(X)
    c[3].markdown(f"**MAR anchor (always observed)**  `{ds.X.columns[anchor]}`")
    m = make_missingness_mask(X, mech, rate, int(seed), anchor=anchor if mech == "MAR" else None)
    card_row([("Requested rate", f"{rate:.0%}"), ("Achieved rate", f"{m.achieved_rate:.2%}"),
              ("Masked cells", f"{int(m.mask.sum()):,}"), ("Seed", int(seed))])
    st.caption(f"Missingness is more likely for {m.direction}." if m.direction != "none" else "Every cell has the same probability of being missing.")
    sample = np.random.default_rng(0).choice(len(X), size=min(300, len(X)), replace=False)
    order = np.argsort(X[sample, anchor]) if mech == "MAR" else np.arange(len(sample))
    st.plotly_chart(px.imshow(m.mask[sample][order].astype(int).T, aspect="auto", color_continuous_scale=["#f4f6fa", "#1f3a5f"],
                              labels={"x": "sampled rows", "y": "feature", "color": "missing"},
                              y=list(ds.X.columns), title="Mask heatmap (rows sorted by anchor value for MAR)"),
                    width="stretch")
    per_feature = pd.DataFrame({"feature": ds.X.columns, "missing_share": m.mask.mean(axis=0)})
    st.plotly_chart(px.bar(per_feature, x="feature", y="missing_share", title="Achieved missing share per feature"),
                    width="stretch")
    st.json(m.metadata())


def run_controls():
    ds = current_dataset()
    with st.form("run"):
        c = st.columns(5)
        mech = c[0].selectbox("Mechanism", config.MECHANISMS, index=2)
        rate = c[1].selectbox("Rate", config.RATES, index=1)
        seed = c[2].number_input("Seed", 0, 1000, 0)
        imputers = c[3].multiselect("Imputers", config.IMPUTERS, default=list(config.IMPUTERS))
        models = c[4].multiselect("Models", config.MODELS, default=["random_forest"])
        go_run = st.form_submit_button("Run lightweight experiment")
    if go_run:
        try:
            cfg = ExperimentConfig(ds.name, mech, float(rate), int(seed), tuple(imputers), tuple(models))
            cfg.validate()
            with st.spinner("Masking, imputing, training and explaining..."):
                service = ExperimentService(INTERACTIVE)
                prep = prepare(ds, INTERACTIVE)
                rows = service.run_unit(prep, cfg, model_only=False)
                rows += [service.clean_retrain_row(prep, m, int(seed)) for m in models]
                st.session_state.last_run = {"cfg": cfg.to_dict(), "rows": rows, "prep": prep}
            log.info("interactive run %s finished", cfg.config_id())
        except Exception as error:
            log.error("interactive run failed: %s", traceback.format_exc())
            st.error(f"The experiment could not be completed: {error}")
    return st.session_state.get("last_run")


def page_imputation():
    st.header("Imputation Comparison")
    run = run_controls()
    if not run:
        st.info("Configure and run a lightweight experiment. Imputers are fitted on training rows only.")
        return
    df = pd.DataFrame(run["rows"])
    df = df[df.imputer != "clean_retrain"].drop_duplicates("imputer")
    st.plotly_chart(px.bar(df, x=df.imputer.map(PRETTY), y="nrmse", labels={"x": "imputer", "nrmse": "NRMSE on hidden test cells"},
                           title="Reconstruction error of artificially hidden cells (training-standardized units)"),
                    width="stretch")
    st.dataframe(df[["imputer", "achieved_rate_train", "achieved_rate_test", "nrmse"]].round(4), width="stretch")


def page_models():
    st.header("Model Evaluation")
    run = run_controls()
    if not run:
        return
    df = pd.DataFrame(run["rows"])
    ref = df.ref_macro_f1.iloc[0]
    card_row([("Reference macro-F1", f"{ref:.3f}"), ("Best imputed macro-F1", f"{df[df.imputer != 'clean_retrain'].macro_f1.max():.3f}"),
              ("Runs", len(df)), ("Mechanism", run["cfg"]["mechanism"])])
    st.plotly_chart(px.bar(df, x=df.imputer.map(PRETTY), y=["macro_f1", "balanced_accuracy"], barmode="group",
                           facet_col="model", labels={"x": "condition", "value": "score"}), width="stretch")
    prep = run["prep"]
    recall = pd.DataFrame([r["class_recall"] for r in run["rows"]], columns=prep.dataset.class_names, index=[f"{r['model']} / {r['imputer']}" for r in run["rows"]])
    st.subheader("Per-class recall")
    st.dataframe(recall.round(3), width="stretch")


def ranking_frames(run):
    prep = run["prep"]
    names = list(prep.dataset.X.columns)
    model = run["cfg"]["models"][0]
    ref = prep.references[(model, run["cfg"]["seed"])]["rankings"]
    return prep, names, model, ref


def page_global_shap():
    st.header("Global SHAP Analysis")
    run = run_controls()
    if not run:
        return
    prep, names, model, ref = ranking_frames(run)
    df = pd.DataFrame(run["rows"])
    df = df[(df.model == model) & (df.imputer != "clean_retrain")]
    imp = pd.Series(ref["global"], index=names).sort_values()
    st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h")).update_layout(
        title="Reference ranking: macro-averaged mean |SHAP| (equal class weights)", height=450), width="stretch")
    card_row([(f"{PRETTY[r.imputer]} global tau-b", f"{r.global_tau:.3f}") for r in df.itertuples()])
    st.dataframe(df[["imputer", "global_tau", "global_top_k", "macro_f1"]].round(3), width="stretch")


def page_classwise_shap():
    st.header("Class-Wise SHAP Analysis")
    run = run_controls()
    if not run:
        return
    prep, names, model, ref = ranking_frames(run)
    df = pd.DataFrame(run["rows"])
    df = df[df.model == model]
    classes = prep.dataset.class_names
    heat = pd.DataFrame([r for r in df.class_tau], index=df.imputer.map(PRETTY), columns=classes)
    st.plotly_chart(px.imshow(heat, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu",
                              title="Kendall tau-b per class against the reference"), width="stretch")
    card_row([(f"{PRETTY[r.imputer]} delta (global - mean class)", f"{r.delta_tau:+.3f}") for r in df.itertuples()])
    cls = st.selectbox("Class", classes)
    ci = classes.index(cls)
    st.dataframe(pd.DataFrame({"feature": names, "reference importance": ref["per_class"][:, ci]})
                 .sort_values("reference importance", ascending=False).round(4), width="stretch")


def page_results():
    st.header("Results and Export")
    res = batch_results()
    if res.empty:
        st.info("No batch results yet. Run `python research_runner.py` to execute the experiment matrix offline.")
    else:
        dataset = st.selectbox("Dataset", sorted(res.dataset.unique()))
        model = st.selectbox("Model", sorted(res[res.dataset == dataset].model.unique()))
        view = res[(res.dataset == dataset) & (res.model == model) & (res.imputer != "clean_retrain")]
        st.plotly_chart(report_service.tau_comparison_figure(view), width="stretch")
        st.dataframe(report_service.summary_table(view), width="stretch")
        st.download_button("Export report bundle (CSV, JSON, HTML)", report_service.export_bundle(view, {"dataset": dataset, "model": model}),
                           file_name=f"imputeaudit_{dataset}_{model}.zip")
    st.subheader("Experiment history")
    st.dataframe(repo.experiments_frame()[["config_id", "status", "started_at", "finished_at", "error_message"]].head(200),
                 width="stretch")
    if "last_run" in st.session_state:
        st.download_button("Export last interactive run (JSON)",
                           json.dumps({"config": st.session_state.last_run["cfg"], "rows": st.session_state.last_run["rows"]}, default=float, indent=1),
                           file_name="imputeaudit_interactive_run.json")


{"Dashboard": page_dashboard, "Dataset Upload and Profiling": page_dataset, "Missingness Simulation": page_missingness,
 "Imputation Comparison": page_imputation, "Model Evaluation": page_models, "Global SHAP Analysis": page_global_shap,
 "Class-Wise SHAP Analysis": page_classwise_shap, "Results and Export": page_results}[page]()


