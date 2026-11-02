"""
Step 3b: runtime of the Moussa Research Suite on the same eight datasets (run AFTER the R benchmark, not
alongside it). Two timings per dataset, each the median of 3 repeats:

  python_1core : the same workload as WHWK, one process, pairs run one after another
                 (MAE/RMSE/MedAE at B=1999, MPE at B=599, Morgan-Pitman SD, McNemar at 8 thresholds,
                  Hommel/Holm), including generation of the bootstrap indices
  python_suite : the package's analysis (run_biometry_analysis + adjust_pvalues), Hommel, B = 1,999, all cores
  python_default: the same with the package's default number of resamples (4,999)

Writes outputs/benchmark_Python.csv and outputs/benchmark_summary.csv (joins benchmark_R.csv).
"""
import importlib.util
import os
import platform
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
PKG = next(p for p in (BASE.parent / "biometry_stats.py", BASE.parent / "github" / "biometry_stats.py") if p.exists())
PREP, OUT = BASE / "prepared", BASE / "outputs"
FORM = ["HQ", "H1", "Haigis", "SRKT", "BU2", "K6"]
THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
REPEATS = 5
THREADS = []   # thread scaling no longer informative: bootstrap tests now run in one matrix step

spec = importlib.util.spec_from_file_location("biometry_stats", PKG)
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def whwk_workload(x):
    n = len(x)
    rng = np.random.default_rng(42)
    i1999 = rng.integers(0, n, size=(1999, n))
    i599 = rng.integers(0, n, size=(599, n))
    p = {m: [] for m in ["MAE", "RMSE", "MPE", "MedAE", "SD"]}
    mcn = {t: [] for t in THRESHOLDS}
    r_all = bs.all_pairs_bootstrap(x, np.arange(n), n, i1999)          # all 15 pairs at once
    r599_all = bs.all_pairs_bootstrap(x, np.arange(n), n, i599)
    for j, k in combinations(range(6), 2):
        e1, e2 = x[:, j], x[:, k]
        r, r599 = r_all[(j, k)], r599_all[(j, k)]
        p["MAE"].append(r["P_MAE"]); p["RMSE"].append(r["P_RMSE"]); p["MedAE"].append(r["P_MedAE"])
        p["MPE"].append(r599["P_MPE"]); p["SD"].append(bs.comdvar(e1, e2))
        for t in THRESHOLDS:
            mcn[t].append(bs.mcnemar_test(np.abs(e1) <= t, np.abs(e2) <= t))
    for v in p.values():
        multipletests(v, method="hommel")
    for v in mcn.values():
        multipletests(v, method="holm")


def suite_default(df, B=1999):
    settings = {"fwer_method": "hommel"}
    if B is not None:
        settings["bootstrap_iterations"] = B
    res, rw, _ = bs.run_biometry_analysis(df, group_name="bench", settings=settings)
    bs.adjust_pvalues(res, rw_stats=rw, method="hommel")


_TPE = bs.ThreadPoolExecutor


def suite_threads(df, k):
    """Package analysis with k worker threads (package default is 8)."""
    bs.ThreadPoolExecutor = lambda max_workers=None: _TPE(max_workers=k)
    try:
        suite_default(df)
    finally:
        bs.ThreadPoolExecutor = _TPE


try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def one_core(fn, *a):
    if threadpool_limits is None:
        return fn(*a)
    with threadpool_limits(limits=1):
        return fn(*a)


def median_time(fn, *a):
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); fn(*a); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ds = pd.read_csv(PREP / "datasets.csv")
    rows = []
    for did in ds.id:
        df = pd.read_csv(PREP / f"{did}.csv")
        x = df[FORM].to_numpy(float)
        t1 = median_time(one_core, whwk_workload, x)
        t2 = median_time(suite_default, df, 1999)            # all cores, WHWK number of resamples
        t3 = median_time(suite_default, df, None)            # all cores, package default resamples
        rows.append({"dataset": did, "n": len(x), "python_1core": t1, "python_suite": t2, "python_default": t3,
                     "default_B": bs.BOOTSTRAP_ITERATIONS})
        print(f"{did:10s} n={len(x):5d}  1-core {t1:7.2f} s   all cores {t2:7.2f} s   default B {t3:7.2f} s", flush=True)
    # thread scaling on the largest dataset
    py_threads = "threadpoolctl" if threadpool_limits else "not limited (threadpoolctl missing)"
    print(f"Single-core limit for BLAS: {py_threads}", flush=True)
    py = pd.DataFrame(rows)
    py["python"] = platform.python_version()
    py["numpy"] = np.__version__
    py["cpu"] = platform.processor()
    py["logical_cpus"] = os.cpu_count()
    py.to_csv(OUT / "benchmark_Python.csv", index=False)

    r = pd.read_csv(OUT / "benchmark_R.csv")
    s = r[["dataset", "n", "MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar", "total"]].rename(columns={"total": "R_total"})
    s = s.merge(py[["dataset", "python_1core", "python_suite", "python_default"]], on="dataset")
    s["speedup_1core"] = s.R_total / s.python_1core
    s["speedup_suite"] = s.R_total / s.python_suite
    s["speedup_default"] = s.R_total / s.python_default
    tot = {"dataset": "All 8 datasets", "n": s.n.sum(),
           **{c: s[c].sum() for c in ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar", "R_total", "python_1core", "python_suite", "python_default"]}}
    tot["speedup_1core"] = tot["R_total"] / tot["python_1core"]
    tot["speedup_suite"] = tot["R_total"] / tot["python_suite"]
    tot["speedup_default"] = tot["R_total"] / tot["python_default"]
    s = pd.concat([s, pd.DataFrame([tot])], ignore_index=True)
    s.to_csv(OUT / "benchmark_summary.csv", index=False)
    print("\n" + s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
