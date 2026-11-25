"""Statistical analysis of the experiment matrix.

Reads the per-unit Parquet files and writes tables (CSV and LaTeX), figures and facts.json.

    python analysis/analyze.py                                  # published results
    python analysis/analyze.py --units runs/units --out runs/analysis   # after a new run
"""
from __future__ import annotations

import argparse
import re
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imputeaudit.evaluation.statistics import benjamini_hochberg, bootstrap_ci, paired_wilcoxon  # noqa: E402

TITLES = {"dry_bean": "Dry Bean", "pendigits": "Pen-Digits", "satimage": "Satimage", "letter": "Letter", "htru2": "HTRU2"}
ORDER = ["dry_bean", "pendigits", "satimage", "letter", "htru2"]
MULTI = ["dry_bean", "pendigits", "satimage", "letter"]
IMP = {"median": "Median", "knn": "KNN", "iterative": "Iterative"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def load(units: Path = ROOT / "results" / "units") -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(Path(units).glob("*.parquet"))]
    if not frames:
        raise SystemExit(f"No unit files found in {units}")
    df = pd.concat(frames, ignore_index=True)
    for col in ("class_tau", "class_top_k", "class_recall"):
        df[col] = df[col].apply(json.loads)
    df["delta_f1"] = df["macro_f1"] - df["ref_macro_f1"]
    df["concealed"] = (df["global_top_k"] == 1.0) & (df["min_class_top_k"] < 1.0)
    return df


def fmt(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def ci_str(values):
    lo, hi = bootstrap_ci(values)
    return f"{np.mean(values):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def latex_table(frame: pd.DataFrame, path: Path, colspec: str) -> None:
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", " & ".join(frame.columns) + r" \\", r"\midrule"]
    prev = None
    for row in frame.itertuples(index=False):
        first = row[0]
        if prev is not None and first != prev and first != "":
            lines.append(r"\midrule")
        prev = first if first != "" else prev
        lines.append(" & ".join(str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines)
    parts = re.split(r"(\$[^$]*\$)", text)
    parts = [p if p.startswith("$") else re.sub(r"(?<!\\)%", r"\\%", p.replace("_", r"\_")) for p in parts]
    path.write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default=str(ROOT / "results" / "units"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load(Path(args.units))
    rf = df[df.model == "random_forest"]
    imp = df[df.imputer != "clean_retrain"]
    clean = df[df.imputer == "clean_retrain"]
    datasets = [d for d in ORDER if d in set(df.dataset)]
    facts: dict = {"n_rows": int(len(df)), "n_imputed_runs": int(len(imp)), "n_clean_retrain": int(len(clean)),
                   "seeds": int(df.seed.nunique()), "datasets": datasets}

    nf_rows = []
    for (d, m), g in clean.groupby(["dataset", "model"]):
        nf_rows.append({"Dataset": TITLES[d], "Model": "RF" if m == "random_forest" else "XGB",
                        "Global $\\tau$": fmt(g.global_tau.mean()), "Mean class $\\tau$": fmt(g.mean_class_tau.mean()),
                        "Worst class $\\tau$": fmt(g.min_class_tau.mean()), "$\\Delta$": fmt(g.delta_tau.mean()),
                        "Concealed": f"{g.concealed.mean():.0%}"})
    nf = pd.DataFrame(nf_rows).sort_values(["Dataset", "Model"], key=lambda s: s.map(
        {TITLES[k]: i for i, k in enumerate(ORDER)}) if s.name == "Dataset" else s)
    nf.to_csv(out / "tables" / "noise_floor.csv", index=False)
    latex_table(nf, out / "tables" / "noise_floor.tex", "llrrrrr")

    main_rows = []
    for d in datasets:
        for mech in ("MCAR", "MAR", "MNAR"):
            for rate in (0.2, 0.4):
                g = rf[(rf.dataset == d) & (rf.mechanism == mech) & (rf.rate == rate)]
                if g.empty:
                    continue
                main_rows.append({"Dataset": TITLES[d] if (mech == "MCAR" and rate == 0.2) else "",
                                  "Mech.": mech, "Rate": f"{int(rate * 100)}%",
                                  "$\\Delta$F1": f"{g.delta_f1.mean():+.3f}",
                                  "Global $\\tau$": fmt(g.global_tau.mean()), "Mean class $\\tau$": fmt(g.mean_class_tau.mean()),
                                  "Worst class $\\tau$": fmt(g.min_class_tau.mean()),
                                  "Global top-5": fmt(g.global_top_k.mean(), 2), "Worst class top-5": fmt(g.min_class_top_k.mean(), 2)})
    main = pd.DataFrame(main_rows)
    main.to_csv(out / "tables" / "main.csv", index=False)
    latex_table(main, out / "tables" / "main.tex", "lllrrrrrr")

    tests = []
    for (d, m, mech, rate, im), g in imp.groupby(["dataset", "model", "mechanism", "rate", "imputer"]):
        base = clean[(clean.dataset == d) & (clean.model == m)].set_index("seed")
        g = g.set_index("seed").loc[lambda x: x.index.isin(base.index)]
        excess = g.delta_tau - base.loc[g.index, "delta_tau"]
        w = paired_wilcoxon(g.delta_tau.values, base.loc[g.index, "delta_tau"].values)
        w0 = paired_wilcoxon(g.delta_tau.values, np.zeros(len(g)))
        tests.append({"dataset": d, "model": m, "mechanism": mech, "rate": rate, "imputer": im, "n": len(g),
                      "delta": g.delta_tau.mean(), "delta_noise": base.delta_tau.mean(), "excess": excess.mean(),
                      "excess_lo": bootstrap_ci(excess.values)[0], "excess_hi": bootstrap_ci(excess.values)[1],
                      "p_excess": w["p"], "r_excess": w["rank_biserial"], "p_delta0": w0["p"],
                      "mean_class_tau": g.mean_class_tau.mean(), "global_tau": g.global_tau.mean(),
                      "concealed": g.concealed.mean()})
    tests = pd.DataFrame(tests)
    tests["q_excess"] = benjamini_hochberg(tests.p_excess)
    tests["q_delta0"] = benjamini_hochberg(tests.p_delta0)
    tests.to_csv(out / "tables" / "condition_tests.csv", index=False)
    multi = tests[tests.dataset.isin(MULTI)]
    facts.update({
        "conditions": int(len(tests)), "conditions_multi": int(len(multi)),
        "delta_pos_sig_multi": int(((multi.q_delta0 < 0.05) & (multi.delta > 0)).sum()),
        "delta_neg_sig_multi": int(((multi.q_delta0 < 0.05) & (multi.delta < 0)).sum()),
        "excess_pos_sig_multi": int(((multi.q_excess < 0.05) & (multi.excess > 0)).sum()),
        "excess_neg_sig_multi": int(((multi.q_excess < 0.05) & (multi.excess < 0)).sum()),
        "delta_positive_share_multi": float((multi.delta > 0).mean()),
    })

    ex_rows = []
    for d in datasets:
        base = clean[(clean.dataset == d) & (clean.model == "random_forest")].set_index("seed").delta_tau
        for mech in ("MCAR", "MAR", "MNAR"):
            for rate in (0.2, 0.4):
                g = rf[(rf.dataset == d) & (rf.mechanism == mech) & (rf.rate == rate)]
                if g.empty:
                    continue
                per_seed = g.groupby("seed").delta_tau.mean()
                excess = per_seed - base.loc[per_seed.index]
                w = paired_wilcoxon(per_seed.values, base.loc[per_seed.index].values)
                conc = g.concealed.mean()
                ex_rows.append({"d": d, "Dataset": TITLES[d] if (mech == "MCAR" and rate == 0.2) else "", "Mech.": mech,
                                "Rate": f"{int(rate * 100)}%", "$\\Delta$ imputed": f"{per_seed.mean():+.3f}",
                                "$\\Delta$ noise floor": f"{base.mean():+.3f}", "Excess [95\\% CI]": ci_str(excess.values),
                                "p": w["p"], "Concealed": f"{conc:.0%}"})
    ex = pd.DataFrame(ex_rows)
    ex["q"] = benjamini_hochberg(ex.p)
    ex["$q$"] = ex.q.map(lambda q: "--" if pd.isna(q) else (f"{q:.3f}" if q >= 0.001 else "<0.001"))
    ex_out = ex.drop(columns=["d", "p", "q"])
    ex_out.to_csv(out / "tables" / "excess.csv", index=False)
    latex_table(ex_out, out / "tables" / "excess.tex", "lllrrlrr")
    ex_multi = ex[ex.d.isin(MULTI)].copy()
    ex_multi["q"] = benjamini_hochberg(ex_multi.p)
    ex_multi["$q$"] = ex_multi.q.map(lambda q: "--" if pd.isna(q) else (f"{q:.3f}" if q >= 0.001 else "<0.001"))
    latex_table(ex_multi.drop(columns=["d", "p", "q"]), out / "tables" / "excess_multi.tex", "lllrrlrr")
    facts["excess_rf_sig_pos"] = int(((ex.q < 0.05) & ex["Excess [95\\% CI]"].str.startswith("+")).sum())
    facts["excess_rf_rows"] = int(len(ex))

    facts["concealed_imputed_multi"] = float(imp[imp.dataset.isin(MULTI)].concealed.mean())
    facts["concealed_clean_multi"] = float(clean[clean.dataset.isin(MULTI)].concealed.mean())
    facts["global_top5_unchanged_multi"] = float((imp[imp.dataset.isin(MULTI)].global_top_k == 1).mean())
    facts["worst_top5_changed_given_global_unchanged"] = float(
        (imp[imp.dataset.isin(MULTI) & (imp.global_top_k == 1)].min_class_top_k < 1).mean())

    mech_rows = []
    for d in datasets:
        for rate in (0.2, 0.4):
            g = rf[(rf.dataset == d) & (rf.rate == rate)].pivot_table(index=["seed", "imputer"], columns="mechanism",
                                                                        values="mean_class_tau")
            for other in ("MAR", "MNAR"):
                if {"MCAR", other} <= set(g.columns):
                    w = paired_wilcoxon(g[other].values, g["MCAR"].values)
                    mech_rows.append({"dataset": d, "rate": rate, "contrast": f"{other}-MCAR",
                                      "diff": float((g[other] - g["MCAR"]).mean()), "p": w["p"], "r": w["rank_biserial"]})
    mech = pd.DataFrame(mech_rows, columns=["dataset", "rate", "contrast", "diff", "p", "r"])
    mech["q"] = benjamini_hochberg(mech.p)
    mech.to_csv(out / "tables" / "mechanism_tests.csv", index=False)
    qfmt = lambda q: "<0.001" if q < 0.001 else f"{q:.3f}"
    mech_lines = []
    for d in datasets:
        for rate in (0.2, 0.4):
            cells = [TITLES[d] if rate == 0.2 else "", f"{int(rate * 100)}\\%"]
            for contrast in ("MAR-MCAR", "MNAR-MCAR"):
                r = mech[(mech.dataset == d) & (mech.rate == rate) & (mech.contrast == contrast)]
                if r.empty:
                    cells += ["--", "--", "--"]
                else:
                    r = r.iloc[0]
                    cells += [f"{r['diff']:+.3f}", f"{r['r']:+.2f}", qfmt(r["q"])]
            mech_lines.append(" & ".join(cells) + r" \\")
        if d != datasets[-1]:
            mech_lines.append(r"\midrule")
    (out / "tables" / "mechanism_body.tex").write_text("\n".join(mech_lines), encoding="utf-8")

    imp_rows = []
    for d in datasets:
        g = rf[(rf.dataset == d) & (rf.imputer != "clean_retrain")]
        piv = g.pivot_table(index=["seed", "mechanism", "rate"], columns="imputer", values="mean_class_tau")
        row = {"Dataset": TITLES[d]}
        for k, name in IMP.items():
            sub = g[g.imputer == k]
            row[f"{name} NRMSE"] = fmt(sub.nrmse.mean(), 2)
            row[f"{name} $\\bar\\tau_c$"] = fmt(sub.mean_class_tau.mean())
            row[f"{name} $\\Delta$F1"] = f"{sub.delta_f1.mean():+.3f}"
        w = paired_wilcoxon(piv["iterative"].values, piv["median"].values)
        row["p_iter_vs_median"] = w["p"]
        row["diff_iter_vs_median"] = float((piv["iterative"] - piv["median"]).mean())
        imp_rows.append(row)
    impt = pd.DataFrame(imp_rows)
    impt["q"] = benjamini_hochberg(impt.p_iter_vs_median)
    impt.to_csv(out / "tables" / "imputers.csv", index=False)
    latex_table(impt[["Dataset"] + [c for c in impt.columns if any(n in c for n in IMP.values())]],
                out / "tables" / "imputers.tex", "l" + "rrr" * 3)
    cols = [c for c in impt.columns if any(n in c for n in IMP.values())]
    lines = [" & ".join([row["Dataset"]] + [str(row[c]) for c in cols]) + r" \\" for _, row in impt.iterrows()]
    (out / "tables" / "imputers_body.tex").write_text("\n".join(lines), encoding="utf-8")

    dec = imp.dropna(subset=["model_only_mean_class_tau"])
    facts["decomp_mean_class_tau_imputed_rows"] = float(dec.mean_class_tau.mean())
    facts["decomp_mean_class_tau_clean_rows"] = float(dec.model_only_mean_class_tau.mean())
    facts["decomp_global_tau_imputed_rows"] = float(dec.global_tau.mean())
    facts["decomp_global_tau_clean_rows"] = float(dec.model_only_global_tau.mean())
    dec_rows = []
    for d in datasets:
        for mech in ("MCAR", "MAR", "MNAR"):
            g = dec[(dec.dataset == d) & (dec.model == "random_forest") & (dec.mechanism == mech) & (dec.rate == 0.4)]
            if len(g):
                dec_rows.append({"Dataset": TITLES[d] if mech == "MCAR" else "", "Mech.": mech,
                                 "Imputed rows": fmt(g.mean_class_tau.mean()), "Complete rows": fmt(g.model_only_mean_class_tau.mean()),
                                 "Noise floor": fmt(clean[(clean.dataset == d) & (clean.model == "random_forest")].mean_class_tau.mean())})
    latex_table(pd.DataFrame(dec_rows), out / "tables" / "decomposition.tex", "llrrr")

    cond = imp.groupby(["dataset", "model", "mechanism", "rate", "imputer"]).agg(
        delta_f1=("delta_f1", "mean"), mean_class_tau=("mean_class_tau", "mean"), global_tau=("global_tau", "mean"),
        min_class_tau=("min_class_tau", "mean")).reset_index()
    cm = cond[cond.dataset.isin(MULTI)]
    rho = spearmanr(cm.delta_f1, cm.mean_class_tau)
    facts["rq3_spearman"] = float(rho.statistic)
    facts["rq3_p"] = float(rho.pvalue)
    small = cm[cm.delta_f1 > -0.02]
    facts["rq3_small_f1_drop_conditions"] = int(len(small))
    facts["rq3_small_f1_drop_mean_class_tau"] = float(small.mean_class_tau.mean()) if len(small) else float("nan")
    facts["rq3_small_f1_drop_min_class_tau"] = float(small.min_class_tau.mean()) if len(small) else float("nan")

    cls_rows = []
    for d in MULTI:
        if d not in datasets:
            continue
        g = rf[(rf.dataset == d) & (rf.imputer != "clean_retrain")]
        taus = np.vstack(g.class_tau.values).mean(axis=0)
        recalls = np.vstack(g.class_recall.values).mean(axis=0)
        cls_rows += [{"dataset": d, "class": i, "tau": t, "recall": r} for i, (t, r) in enumerate(zip(taus, recalls))]
    cls = pd.DataFrame(cls_rows)
    cls.to_csv(out / "tables" / "class_level.csv", index=False)
    if len(cls) > 3:
        r2 = spearmanr(cls.tau, cls.recall)
        facts["class_tau_vs_recall_spearman"] = float(r2.statistic)
        facts["class_tau_vs_recall_p"] = float(r2.pvalue)

    xg = df[(df.dataset == "dry_bean") & (df.imputer != "clean_retrain")]
    if (xg.model == "xgboost").any():
        for m in ("random_forest", "xgboost"):
            s = xg[xg.model == m]
            facts[f"dry_bean_{m}_delta"] = float(s.delta_tau.mean())
            facts[f"dry_bean_{m}_mean_class_tau"] = float(s.mean_class_tau.mean())
            facts[f"dry_bean_{m}_global_tau"] = float(s.global_tau.mean())
            facts[f"dry_bean_{m}_concealed"] = float(s.concealed.mean())
        xc = clean[(clean.dataset == "dry_bean")]
        for m in ("random_forest", "xgboost"):
            facts[f"dry_bean_{m}_noise_delta"] = float(xc[xc.model == m].delta_tau.mean())

    facts["achieved_rate_max_abs_error"] = float((imp.achieved_rate_train - imp.rate).abs().max())
    facts["achieved_rate_test_max_abs_error"] = float((imp.achieved_rate_test - imp.rate).abs().max())
    facts["runtime_total_hours"] = float(df.runtime_s.sum() / 3600)
    (out / "tables" / "facts.json").write_text(json.dumps(facts, indent=2, default=float), encoding="utf-8")
    figures(df, rf, clean, cond, cls, out / "figures")
    print(json.dumps(facts, indent=2, default=float))


def figures(df, rf, clean, cond, cls, figdir: Path):
    datasets = [d for d in ORDER if d in set(df.dataset)]
    colors = {"global_tau": "#1f4e79", "mean_class_tau": "#c55a11", "min_class_tau": "#7f7f7f"}
    labels = {"global_tau": "global", "mean_class_tau": "mean class", "min_class_tau": "worst class"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.35), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, datasets):
        g = rf[(rf.dataset == d) & (rf.imputer != "clean_retrain")]
        conds = [(m, r) for m in ("MCAR", "MAR", "MNAR") for r in (0.2, 0.4)]
        x = np.arange(len(conds))
        for k, (metric, c) in enumerate(colors.items()):
            means = [g[(g.mechanism == m) & (g.rate == r)][metric].mean() for m, r in conds]
            ax.plot(x, means, marker="o", ms=2.5, lw=1, color=c, label=labels[metric])
            nfv = clean[(clean.dataset == d) & (clean.model == "random_forest")][metric].mean()
            ax.axhline(nfv, color=c, lw=0.6, ls=":")
        ax.set_xticks(x, [f"{m} {int(r*100)}%" for m, r in conds], fontsize=5.5, rotation=90)
        ax.set_title(TITLES[d])
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Kendall $\\tau_b$ vs. reference")
    axes[-1].legend(loc="lower left", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(figdir / "fig_tau_conditions.pdf")
    fig.savefig(figdir / "fig_tau_conditions.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    markers = {"MCAR": "o", "MAR": "s", "MNAR": "^"}
    palette = {"dry_bean": "#1f4e79", "pendigits": "#c55a11", "satimage": "#548235", "letter": "#7030a0", "htru2": "#7f7f7f"}
    for (d, mech), g in cond.groupby(["dataset", "mechanism"]):
        ax.scatter(g.delta_f1, g.min_class_tau, s=9, marker=markers[mech], color=palette[d], alpha=0.8, lw=0)
    for d in datasets:
        ax.scatter([], [], color=palette[d], s=9, label=TITLES[d])
    for m, mk in markers.items():
        ax.scatter([], [], color="black", marker=mk, s=9, label=m)
    ax.set_xlabel("$\\Delta$ macro-F1 (imputed $-$ reference)")
    ax.set_ylabel("worst-class Kendall $\\tau_b$")
    ax.grid(lw=0.3, alpha=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=6, loc="lower center", columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "fig_f1_vs_tau.pdf")
    fig.savefig(figdir / "fig_f1_vs_tau.png", dpi=200)
    plt.close(fig)

    if "dry_bean" in datasets:
        from imputeaudit.data.loader import load_bundled
        names = load_bundled("dry_bean").class_names
        g = rf[(rf.dataset == "dry_bean") & (rf.rate == 0.4) & (rf.imputer != "clean_retrain")]
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        width = 0.2
        for k, mech in enumerate(("MCAR", "MAR", "MNAR")):
            vals = np.vstack(g[g.mechanism == mech].class_tau.values).mean(axis=0)
            ax.bar(np.arange(len(names)) + (k - 1) * width, vals, width, label=mech,
                   color=["#9dc3e6", "#f4b183", "#a9d18e"][k], edgecolor="black", lw=0.3)
        nf = np.vstack(clean[(clean.dataset == "dry_bean") & (clean.model == "random_forest")].class_tau.values).mean(axis=0)
        ax.scatter(np.arange(len(names)) + 0.3, nf, marker="_", s=60, color="black", label="noise floor", zorder=3)
        ax.set_xticks(np.arange(len(names)), [n.title() for n in names], rotation=30, fontsize=6.5)
        ax.set_ylabel("class-wise Kendall $\\tau_b$")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        fig.tight_layout(pad=0.3)
        fig.savefig(figdir / "fig_drybean_classes.pdf")
        fig.savefig(figdir / "fig_drybean_classes.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()


