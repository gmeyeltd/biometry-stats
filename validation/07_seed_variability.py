"""
Step 7: how much do the bootstrap results depend on the random seed?
Python only (equivalence with R was shown in step 2). For each seed, every bootstrap comparison
(MAE, RMSE, MPE, MedAE; 15 formula pairs x 8 datasets = 480) is run twice:
  WHWK      : WHWK settings - 1,999 resamples (599 for MPE), p = b/B, dmedpb rule for MedAE
  IMPROVED  : the suite's improved settings - 4,999 resamples for every outcome, p = (1 + b)/(B + 1)
Each setting uses its own independent stream of seeds. Hommel's correction across the 15 pairs, as in WHWK.
Run after steps 0-2 (needs prepared/). Usage: python 07_seed_variability.py [n_seeds] [n_workers]
Writes outputs/seed_variability_long.csv.gz and outputs/seed_variability_summary.csv
"""
import importlib.util, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PKG = next(p for p in (BASE.parent / "biometry_stats.py", BASE.parent / "github" / "biometry_stats.py") if p.exists())
PREP, OUT = BASE / "prepared", BASE / "outputs"
FORM = ["HQ", "H1", "Haigis", "SRK/T", "BU2", "K6"]
BOOT = ["MAE", "RMSE", "MPE", "MedAE"]
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
N_WORK = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) // 2 - 1)


def load_pkg():
    spec = importlib.util.spec_from_file_location("biometry_stats", PKG)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def whwk_t_p(null, obs):                      # WHWK ydbt: b/B
    return float(np.mean(null >= obs))


def whwk_pct(d):                              # WHWK dmedpb: 2 min(T, 1 - T), T = P(>0) + 0.5 P(=0)
    t = np.mean(d > 0) + 0.5 * np.mean(d == 0)
    return float(2 * min(t, 1 - t))


def one_seed(seed):
    """All 480 bootstrap comparisons for one seed, both settings. Single-threaded inside the worker."""
    try:
        from threadpoolctl import threadpool_limits
        ctx = threadpool_limits(limits=1)
    except ImportError:
        ctx = None
    bs = load_pkg()
    from statsmodels.stats.multitest import multipletests
    ds = pd.read_csv(PREP / "datasets.csv")
    rows = []
    for did in ds.id:
        x = pd.read_csv(PREP / f"{did}.csv").to_numpy(float); n = len(x); cl = np.arange(n)
        # WHWK settings: own resamples, seed stream 1
        rng = np.random.default_rng([1, seed])
        i1999, i599 = rng.integers(0, n, size=(1999, n)), rng.integers(0, n, size=(599, n))
        saved = bs._bootstrap_p, bs._calculate_percentile_bootstrap_p
        bs._bootstrap_p, bs._calculate_percentile_bootstrap_p = whwk_t_p, whwk_pct
        try:
            r1999 = bs.all_pairs_bootstrap(x, cl, n, i1999)
            r599 = bs.all_pairs_bootstrap(x, cl, n, i599)
        finally:
            bs._bootstrap_p, bs._calculate_percentile_bootstrap_p = saved
        # improved settings: 4,999 resamples for every outcome, seed stream 2
        rng2 = np.random.default_rng([2, seed])
        r4999 = bs.all_pairs_bootstrap(x, cl, n, rng2.integers(0, n, size=(4999, n)))
        for setting, get in (("WHWK", lambda jk, met: (r599 if met == "MPE" else r1999)[jk][f"P_{met}"]),
                             ("IMPROVED", lambda jk, met: r4999[jk][f"P_{met}"])):
            for met in BOOT:
                pairs = list(combinations(range(6), 2))
                p = np.array([get(jk, met) for jk in pairs])
                padj = multipletests(p, method="hommel")[1]
                rows += [dict(seed=seed, setting=setting, dataset=did, metric=met, f1=FORM[j], f2=FORM[k], p=pi, p_adj=pa)
                         for (j, k), pi, pa in zip(pairs, p, padj)]
    return rows


if __name__ == "__main__":
    t0 = time.time()
    print(f"{N_SEEDS} seeds on {N_WORK} worker processes ...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORK) as ex:
        for i, r in enumerate(ex.map(one_seed, range(1, N_SEEDS + 1)), 1):
            rows += r
            if i % 10 == 0 or i == N_SEEDS:
                print(f"  {i}/{N_SEEDS} seeds done ({time.time() - t0:.0f} s)", flush=True)
    L = pd.DataFrame(rows)
    L.to_csv(OUT / "seed_variability_long.csv.gz", index=False)
    g = L.groupby(["setting", "dataset", "metric", "f1", "f2"]).p_adj
    S = g.agg(mean="mean", sd="std", min="min", max="max").reset_index()
    S["n_sig"] = g.apply(lambda v: int((v < 0.05).sum())).values
    S["seed_dependent"] = (S.n_sig > 0) & (S.n_sig < N_SEEDS)
    S.to_csv(OUT / "seed_variability_summary.csv", index=False)
    print("\nComparisons whose decision at 0.05 depends on the seed (of 480):")
    print(S.groupby("setting").seed_dependent.sum().to_string())
    print("\nMedian SD of adjusted p across seeds, by setting:")
    print(S.groupby("setting").sd.median().round(4).to_string())
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in {OUT}")
