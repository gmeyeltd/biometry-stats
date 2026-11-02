"""
Step 2: run the Moussa Research Suite engine (../github/biometry_stats.py, unmodified) on the SAME
bootstrap indices that WHWK/R used, then compare every statistic and p-value with the R output.

STUDY-ONLY WHWK OVERLAY (applied in memory; biometry_stats.py on disk is not changed):
  * bootstrap-t p-value  = #{|t*| >= |t_obs|} / B          (WHWK ydbt)   [package default: (1+b)/(B+1)]
  * MedAE percentile p   = 2*min(T, 1-T), T = P(d*>0) + 0.5 P(d*=0)        (WHWK dmedpb)
  * MPE uses B = 599 (WHWK oph.dep.commean default); MAE/RMSE/MedAE use B = 1999
  * multiplicity: Hommel per metric across the 15 pairs; McNemar: Holm across the 15 pairs per threshold
  * eye-level resampling (one eye per patient)

Writes outputs/Python_results.csv, outputs/Table2_agreement.csv, outputs/S1_Table_all_comparisons.xlsx,
       outputs/Fig_p_agreement.tif, outputs/run_report.md
"""
import hashlib
import importlib.util
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels


def md(df):
    return "```\n" + df.to_string(index=False) + "\n```"
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
PKG = next(p for p in (BASE.parent / "biometry_stats.py", BASE.parent / "github" / "biometry_stats.py") if p.exists())
PREP, OUT = BASE / "prepared", BASE / "outputs"
FORM = ["HQ", "H1", "Haigis", "SRK/T", "BU2", "K6"]
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
B_MAIN, B_MPE = 1999, 599

# ---------------------------------------------------------------- load the package, apply overlay
spec = importlib.util.spec_from_file_location("biometry_stats", PKG)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)

def whwk_bootstrap_p(null_abs_t, obs_abs_t):
    return float(np.mean(null_abs_t >= obs_abs_t))

def whwk_percentile_p(boot_diffs):
    t = np.mean(boot_diffs > 0) + 0.5 * np.mean(boot_diffs == 0)
    return float(2 * min(t, 1 - t))

bs._bootstrap_p = whwk_bootstrap_p
bs._calculate_percentile_bootstrap_p = whwk_percentile_p


def engine(e1, e2, idx0):
    """Call the package's bootstrap engine at eye level with externally supplied 0-based indices."""
    n = len(e1)
    return bs.perform_advanced_bootstrap(e1, e2, np.arange(n), n, idx0, np.zeros_like(idx0), use_studentized=True)


def load_idx(did, B):
    m = pd.read_csv(OUT / "indices" / f"{did}_B{B}.csv.gz", header=None).to_numpy(dtype=np.int64) - 1  # R is 1-based
    assert m.shape[0] == B
    return m


def run_python(ds):
    rows, trows, timing = [], [], []
    for did in ds.id:
        x = pd.read_csv(PREP / f"{did}.csv").to_numpy(dtype=float)
        idx1999, idx599 = load_idx(did, B_MAIN), load_idx(did, B_MPE)
        assert idx1999.shape[1] == len(x)
        t0 = time.perf_counter()
        for j, k in combinations(range(6), 2):
            e1, e2 = x[:, j], x[:, k]
            r = engine(e1, e2, idx1999)
            r599 = engine(e1, e2, idx599)
            base = dict(dataset=did, f1=FORM[j], f2=FORM[k])
            rows += [
                {**base, "metric": "MAE", "est1": np.mean(np.abs(e1)), "est2": np.mean(np.abs(e2)), "p": r["P_MAE"]},
                {**base, "metric": "RMSE", "est1": np.sqrt(np.mean(e1**2)), "est2": np.sqrt(np.mean(e2**2)), "p": r["P_RMSE"]},
                {**base, "metric": "MPE", "est1": np.mean(e1), "est2": np.mean(e2), "p": r599["P_MPE"]},
                {**base, "metric": "MedAE", "est1": np.median(np.abs(e1)), "est2": np.median(np.abs(e2)), "p": r["P_MedAE"]},
                {**base, "metric": "SD", "est1": np.std(e1, ddof=1), "est2": np.std(e2, ddof=1), "p": bs.comdvar(e1, e2)},
            ]
            for thr in THRESHOLDS:
                m1, m2 = np.abs(e1) <= thr, np.abs(e2) <= thr
                rows.append({**base, "metric": "McNemar", "threshold": thr, "est1": m1.mean(), "est2": m2.mean(),
                             "p": bs.mcnemar_test(m1, m2)})
            trows.append({**base, "t_MAE": r["RW_MAE_obs"], "t_RMSE": r["RW_RMSE_obs"], "t_MPE": r599["RW_MPE_obs"]})
        timing.append({"dataset": did, "n": len(x), "seconds": time.perf_counter() - t0})
    res = pd.DataFrame(rows)
    # multiplicity, exactly as WHWK: Hommel within each metric x dataset; Holm for McNemar within each threshold
    res["p_adj"] = np.nan
    for (did, met, thr), g in res.groupby(["dataset", "metric", "threshold"], dropna=False):
        method = "holm" if met == "McNemar" else "hommel"
        res.loc[g.index, "p_adj"] = multipletests(g.p.values, method=method)[1]
    return res, pd.DataFrame(trows), pd.DataFrame(timing)


def compare(rr, pr, rts, pts):
    key = ["dataset", "metric", "threshold", "f1", "f2"]
    for d in (rr, pr):
        d["threshold"] = d["threshold"].fillna(-1).round(2)
    m = rr.merge(pr, on=key, suffixes=("_R", "_Py"), how="outer", indicator=True)
    assert (m["_merge"] == "both").all(), "R and Python comparisons do not line up"
    m = m.drop(columns="_merge")
    m["threshold"] = m["threshold"].replace(-1, np.nan)
    for c in ["est1", "est2", "p", "p_adj"]:
        m[f"absdiff_{c}"] = (m[f"{c}_R"] - m[f"{c}_Py"]).abs()
    for a in (0.05, 0.01):
        m[f"sig{a}_R"] = m.p_adj_R < a
        m[f"sig{a}_Py"] = m.p_adj_Py < a
    # observed test statistics
    t = rts.merge(pts, on=["dataset", "f1", "f2"], suffixes=("_R", "_Py"))
    tdiff = {met: float((t[f"t_{met}_R"].abs() - t[f"t_{met}_Py"]).abs().max()) for met in ["MAE", "RMSE", "MPE"]}

    order = ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar"]
    tab = []
    for met in order:
        g = m[m.metric == met]
        tab.append({
            "Metric": met,
            "WHWK function": {"MAE": "oph.dep.comMeanAE", "RMSE": "oph.dep.comRMSAE", "MPE": "oph.dep.commean",
                              "MedAE": "oph.dep.comMedAE", "SD": "oph.dep.comvar", "McNemar": "oph.mcnemar"}[met],
            "Method": {"MAE": "Bootstrap-t, B=1999", "RMSE": "Bootstrap-t (MSE), B=1999", "MPE": "Bootstrap-t, B=599",
                       "MedAE": "Percentile bootstrap, B=1999", "SD": "Morgan-Pitman, HC4",
                       "McNemar": "McNemar chi-square (continuity corrected), 8 thresholds"}[met],
            "Adjustment": "Holm" if met == "McNemar" else "Hommel",
            "Comparisons": len(g),
            "Identical raw p": int((g.absdiff_p == 0).sum()),
            "Max |dp| raw": g.absdiff_p.max(),
            "Identical adjusted p": int((g.absdiff_p_adj == 0).sum()),
            "Max |dp| adjusted": g.absdiff_p_adj.max(),
            "Max |d t_obs|": tdiff.get(met, np.nan),
            "Max |d estimate|": max(g.absdiff_est1.max(), g.absdiff_est2.max()),
            "Significant (R, a=0.05)": int(g["sig0.05_R"].sum()),
            "Discordant decisions a=0.05": int((g["sig0.05_R"] != g["sig0.05_Py"]).sum()),
            "Discordant decisions a=0.01": int((g["sig0.01_R"] != g["sig0.01_Py"]).sum()),
        })
    return m, pd.DataFrame(tab)


def figure(m, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar"]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.2), dpi=300, sharex=True, sharey=True)
    for ax, met, lab in zip(axes.flat, order, "ABCDEF"):
        g = m[m.metric == met]
        ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, zorder=1)
        ax.scatter(g.p_adj_R, g.p_adj_Py, s=9, color="#2a6f97", alpha=0.75, lw=0, zorder=2)
        ax.axvline(0.05, color="#c44e52", lw=0.6, ls="--"); ax.axhline(0.05, color="#c44e52", lw=0.6, ls="--")
        ax.set_title(f"{lab}  {met} (n={len(g)})", fontsize=8, loc="left")
        ax.tick_params(labelsize=7); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.supxlabel("WHWK (R) adjusted p-value", fontsize=8)
    fig.supylabel("Moussa Research Suite (Python) adjusted p-value", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    ds = pd.read_csv(PREP / "datasets.csv")
    pr, pts, timing = run_python(ds)
    pr.to_csv(OUT / "Python_results.csv", index=False)
    rr = pd.read_csv(OUT / "R_results.csv")
    rts = pd.read_csv(OUT / "R_teststats.csv")
    m, tab = compare(rr, pr, rts, pts)
    tab.to_csv(OUT / "Table2_agreement.csv", index=False)
    with pd.ExcelWriter(OUT / "S1_Table_all_comparisons.xlsx") as w:
        tab.to_excel(w, sheet_name="Summary", index=False)
        ds.to_excel(w, sheet_name="Datasets", index=False)
        m.to_excel(w, sheet_name="All comparisons", index=False)
    # figures are drawn separately (TIFF writing crashed on some Windows/conda set-ups)

    sha = hashlib.sha256(PKG.read_bytes()).hexdigest()
    rchk = pd.read_csv(OUT / "R_index_check.csv")
    lines = [
        "# WHWK vs Moussa: shared-index run report", "",
        f"- Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"statsmodels {statsmodels.__version__}, pandas {pd.__version__}",
        f"- biometry_stats.py SHA-256: {sha}",
        f"- Eyes analysed: {int(ds.n_analysed.sum())} in {len(ds)} datasets; total comparisons: {len(m)}",
        "", "## Agreement by metric", "", md(tab), "",
        "## R index check (saved indices reproduce R's own p)", "", md(rchk), "",
        "## Python runtime (seconds)", "", md(timing), "",
        "## Largest raw-p differences", "",
        md(m.sort_values("absdiff_p", ascending=False).head(10)[
            ["dataset", "metric", "threshold", "f1", "f2", "p_R", "p_Py", "absdiff_p"]]),
    ]
    (OUT / "run_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string(index=False))
    print(f"\nReport written to {OUT / 'run_report.md'}")


if __name__ == "__main__":
    main()
