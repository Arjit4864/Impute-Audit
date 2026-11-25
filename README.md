# ImputeAudit

Code, data and results for the study *Missingness Mechanisms and Class-Wise SHAP Stability in Tabular
Classification* (Arjit Singh, Maharaja Agrasen Institute of Technology, Delhi).

ImputeAudit takes a complete tabular classification dataset, hides values under a controlled MCAR, MAR
or MNAR mechanism, imputes them with pipelines fitted on training rows only, trains Random Forest or
XGBoost, and compares the global and class-wise TreeSHAP feature rankings of the imputed-data model with
those of a complete-data reference. A second complete-data model trained with another seed gives the
noise floor of every agreement measure.

The repository contains:

* the `imputeaudit` Python package and a Streamlit interface (`app.py`);
* the offline experiment runner used for the study (`research_runner.py`);
* the five benchmark datasets (`data/samples`, CC BY 4.0, see `data/README.md`);
* the raw results of all 1,140 runs (`results/units`) and the analysis that turns them into every table,
  figure and number of the article (`analysis/analyze.py`, outputs in `results/tables` and `results/figures`);
* 42 automated tests (`tests/`).

## Requirements

Python 3.12 or newer; the published results were produced with Python 3.14.3 and the versions pinned in
`requirements.txt` (scikit-learn 1.9.1, SHAP 0.54.1, XGBoost 3.4.1). No GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
```

## Reproduce the article from the published results

```bash
python analysis/analyze.py
```

This reads `results/units/*.parquet` and rewrites `results/tables/` and `results/figures/`.
`results/tables/facts.json` holds the summary numbers quoted in the text.

## Re-run the experiments

```bash
./run_full_matrix.sh             # Windows: .\run_full_matrix.ps1
```

The runner executes 5 datasets × 3 mechanisms × 2 rates × 10 seeds, with three imputers per unit,
XGBoost in addition to Random Forest on Dry Bean, and a complete-data retrain per dataset, model and seed
(1,080 imputed runs and 60 noise-floor runs). On a 24-thread desktop CPU with 18 workers it took 73 minutes.
Every finished unit is written atomically to `runs/units/`; an interrupted run resumes where it stopped.
The analysis of a new run is written to `runs/analysis/`.

A quick pilot:

```bash
python research_runner.py --seeds 2 --datasets satimage --jobs 2
```

## Interactive application

```bash
python scripts/load_published_results.py    # optional: show the published batch results in the app
streamlit run app.py                         # http://localhost:8501
```

Pages: dashboard, dataset upload and profiling, missingness simulation, imputation comparison, model
evaluation, global SHAP analysis, class-wise SHAP analysis, results and export. Screenshots are in
`docs/screenshots/`. Uploaded CSV files are validated (target column, numeric predictors, no missing or
infinite values, at least 20 rows per class) and processed in memory.

With Docker:

```bash
docker compose up --build
```

## Protocol

| Step | Rule |
|---|---|
| Split | Made once and stored as row identifiers; official partitions for Pen-Digits (writer-independent), Satimage and Letter, stratified 80/20 for Dry Bean and HTRU2 |
| Masks | Predictors only; MCAR Bernoulli, MAR logistic on an always-observed anchor (highest mean absolute correlation), MNAR self-masking logistic on the cell's own value; slope 1.5; intercepts calibrated by bisection to 20% or 40%; independent train and test masks from the same rule |
| Imputation | NaN-aware `StandardScaler` + `SimpleImputer(median)`, `KNNImputer(k=5)` or `IterativeImputer(BayesianRidge, 10 rounds)`, fitted on masked training rows only |
| Models | Random Forest (100 trees, depth 12, min. leaf 2) on all datasets; XGBoost (200 trees, depth 6, learning rate 0.1) on Dry Bean |
| Explanations | `shap.TreeExplainer`, tree-path-dependent, on one fixed class-stratified sample of max(10, ⌊300/C⌋) test rows per class |
| Rankings | Class-wise importance = mean absolute SHAP per class; global importance = equal-weight mean over classes |
| Agreement | Kendall τ_b and top-5 Jaccard, globally and per class, against the complete-data reference of the same seed |

## Layout

```
imputeaudit/
  config.py               defaults, dataset registry, IMPUTEAUDIT_* environment overrides
  data/                   CSV ingestion, validation, profiling
  missingness/            MCAR, MAR and MNAR generators, intercept calibration
  pipeline/               frozen splits, train-only preprocessing, model training
  explainability/         TreeSHAP adapter, rankings, Kendall tau-b, top-k Jaccard
  evaluation/             prediction metrics, NRMSE, bootstrap CI, Wilcoxon, Benjamini-Hochberg
  storage/                SQLite run index
  services/               dataset, experiment and report services
app.py                    Streamlit interface
research_runner.py        parallel, checkpointed experiment runner
analysis/analyze.py       statistics, tables and figures
scripts/                  dataset download and preparation, result loading, screenshots
data/samples/             benchmark datasets
results/                  published unit results, tables, figures
tests/                    unit and integration tests
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `IMPUTEAUDIT_RUNS_DIR` | `runs/` | SQLite index and unit checkpoints |
| `IMPUTEAUDIT_LOG_DIR` | `logs/` | Rotating log file |
| `IMPUTEAUDIT_MAX_UPLOAD_MB` | 50 | Upload size limit |
| `IMPUTEAUDIT_MAX_ROWS` | 200000 | Row limit for uploads |
| `IMPUTEAUDIT_MIN_ROWS_PER_CLASS` | 20 | Minimum rows per class for uploads |
| `IMPUTEAUDIT_STORE_RAW_UPLOADS` | false | Keep uploaded files after processing |

## Limitations

The masks simulate named mechanisms on complete data; the tool does not identify the mechanism of real
missing data. SHAP values are model-specific attributions and the complete-data ranking is a reference,
not ground truth. Only numeric predictors and classification targets are supported.

## Acknowledgement of tools

Parts of the code and documentation were written with the help of Kiro, an AI coding assistant. All
experiments were run, checked and analysed by the author.

## Licence

Code: MIT (see `LICENSE`). Datasets: CC BY 4.0, credited in `data/README.md`.
