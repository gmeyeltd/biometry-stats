"""
Step 4: effect of the package's deliberate deviations from WHWK, and of routine use with its own random seed.
Run after steps 0-2 (needs prepared/, outputs/R_results.csv and outputs/indices/*_B1999.csv.gz).

Deviations (package default -> WHWK):
  D1  bootstrap p = (1 + b)/(B + 1)             vs  b/B                         (MAE, RMSE, MPE)
  D2  MPE uses the same B as other tests (1,999) vs  599                          (MPE)
  D3  MedAE percentile p = 2(1 + min(#>0,#<0))/(B + 1)  vs  2 min(T, 1-T), T = P(>0) + 0.5 P(=0)
  D4  Hommel for McNemar                          vs  Holm
  RANDOM   WHWK settings and B, but the package's own resamples (seed 42): Monte Carlo randomness only
  ALL all four together, same resamples as WHWK
  ROUTINE  package defaults (all deviations, Hommel) with its own seed (42) and its default B (4,999)
Writes outputs/deviations_summary.csv and outputs/deviations_all.csv
"""
import importlib.util
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent
PKG = next(p for p in (BASE.parent / "biometry_stats.py", BASE.parent / "github" / "biometry_stats.py") if p.exists())
PREP, OUT = BASE / "prepared", BASE / "outputs"
FORM = ["HQ", "H1", "Haigis", "SRK/T", "BU2", "K6"]
B = 1999

spec = importlib.util.spec_from_file_location("biometry_stats", PKG)
bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)   # package defaults, untouched

R = pd.read_csv(OUT / "R_results.csv")
R["threshold"] = R["threshold"].round(2)
ds = pd.read_csv(PREP / "datasets.csv")


def whwk_t_p(null, obs):  # WHWK b/B
    return float(np.mean(null >= obs))


def whwk_pct(d):  # WHWK dmedpb rule
    t = np.mean(d > 0) + 0.5 * np.mean(d == 0)
    return float(2 * min(t, 1 - t))


def engine_whwk(e1, e2, idx):
    """Package engine with WHWK p-value conventions (temporarily swapped in memory)."""
    saved = bs._bootstrap_p, bs._calculate_percentile_bootstrap_p
    bs._bootstrap_p, bs._calculate_percentile_bootstrap_p = whwk_t_p, whwk_pct
    try:
        return bs.perform_advanced_bootstrap(e1, e2, np.arange(len(e1)), len(e1), idx, np.zeros_like(idx))
    finally:
        bs._bootstrap_p, bs._calculate_percentile_bootstrap_p = saved


rows = []
for did in ds.id:
    x = pd.read_csv(PREP / f"{did}.csv").to_numpy(float); n = len(x)
    idx = pd.read_csv(OUT / "indices" / f"{did}_B{B}.csv.gz", header=None).to_numpy(np.int64) - 1
    # routine use: the package exactly as shipped (own RNG, seed 42, B = 2000), Hommel
    df = pd.DataFrame(x, columns=FORM)
    res, rw, _ = bs.run_biometry_analysis(df, group_name=did, settings={"fwer_method": "hommel"})   # package default number of resamples
    routine = {(a, b_): r for a, b_, r in zip(res["Formula A"], res["Formula B"], res.to_dict("records"))}
    # randomness only: WHWK settings and B, but the package's own resamples (numpy, seed 42)
    rng = np.random.default_rng(42)
    own1999 = rng.choice(n, size=(B, n), replace=True); own599 = rng.choice(n, size=(599, n), replace=True)
    for j, k in combinations(range(6), 2):
        e1, e2 = x[:, j], x[:, k]
        rw1, rw5 = engine_whwk(e1, e2, own1999), engine_whwk(e1, e2, own599)
        own = {"MAE": rw1["P_MAE"], "RMSE": rw1["P_RMSE"], "MPE": rw5["P_MPE"], "MedAE": rw1["P_MedAE"], "SD": bs.comdvar(e1, e2)}
        r = bs.perform_advanced_bootstrap(e1, e2, np.arange(n), n, idx, np.zeros_like(idx))   # package conventions
        medae_ties = int(np.sum([np.median(np.abs(e1[i])) == np.median(np.abs(e2[i])) for i in idx]))
        rr = routine[(FORM[j], FORM[k])]
        base = dict(dataset=did, f1=FORM[j], f2=FORM[k])
        rows += [{**d, "p_random_whwk": own[d["metric"]]} for d in [
            {**base, "metric": "MAE", "p_default_shared": r["P_MAE"], "p_mpeB_whwk": np.nan, "p_routine": rr["Boot (MAE) p"]},
            {**base, "metric": "RMSE", "p_default_shared": r["P_RMSE"], "p_mpeB_whwk": np.nan, "p_routine": rr["Boot (RMSE) p"]},
            {**base, "metric": "MPE", "p_default_shared": r["P_MPE"], "p_mpeB_whwk": whwk_t_p(r["RW_MPE_null"], r["RW_MPE_obs"]), "p_routine": rr["Boot (MPE) p"]},
            {**base, "metric": "MedAE", "p_default_shared": r["P_MedAE"], "p_mpeB_whwk": np.nan, "p_routine": rr["Boot (MedAE) p"], "medae_ties": medae_ties},
            {**base, "metric": "SD", "p_default_shared": bs.comdvar(e1, e2), "p_mpeB_whwk": np.nan, "p_routine": rr["Morgan-Pitman (SD) p"]},
        ]]
P = pd.DataFrame(rows)
m = R.merge(P, on=["dataset", "metric", "f1", "f2"], how="left")

def adj(df, col, method_fn):
    out = pd.Series(np.nan, index=df.index)
    for key, g in df.groupby(["dataset", "metric", "threshold"], dropna=False):
        v = g[col]
        if v.notna().all():
            out.loc[g.index] = multipletests(v.values, method=method_fn(key[1]))[1]
    return out

hom = lambda met: "hommel"
whwk_method = lambda met: "holm" if met == "McNemar" else "hommel"
Bm = np.where(m.metric == "MPE", 599, B)
# D1: +1 rule applied to WHWK counts
m["p_D1"] = np.where(m.metric.isin(["MAE", "RMSE", "MPE"]), (1 + m.p * Bm) / (Bm + 1), m.p)
# D2: MPE at B = 1999 with WHWK b/B rule
m["p_D2"] = np.where(m.metric == "MPE", m.p_mpeB_whwk, m.p)
# D3: package MedAE rule on WHWK resamples
m["p_D3"] = np.where(m.metric == "MedAE", m.p_default_shared, m.p)
# ALL: package conventions on shared resamples (McNemar raw p unchanged)
m["p_ALL"] = np.where(m.metric == "McNemar", m.p, m.p_default_shared)
m["p_ROUTINE"] = np.where(m.metric == "McNemar", m.p, m.p_routine)
m["p_RANDOM"] = np.where(m.metric == "McNemar", m.p, m.p_random_whwk)
m["p_RANDOM_adj"] = adj(m, "p_RANDOM", whwk_method)
for c in ["p_D1", "p_D2", "p_D3"]:
    m[c + "_adj"] = adj(m, c, whwk_method)
m["p_D4_adj"] = adj(m, "p", hom)
m["p_ALL_adj"] = adj(m, "p_ALL", hom)
m["p_ROUTINE_adj"] = adj(m, "p_ROUTINE", hom)

summ = []
for sc, metrics in [("D1", ["MAE", "RMSE", "MPE"]), ("D2", ["MPE"]), ("D3", ["MedAE"]), ("D4", ["McNemar"]),
                    ("RANDOM", ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar"]), ("ALL", ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar"]), ("ROUTINE", ["MAE", "RMSE", "MPE", "MedAE", "SD", "McNemar"])]:
    for met in metrics:
        g = m[m.metric == met]
        d = (g[f"p_{sc}_adj"] - g.p_adj).abs()
        summ.append({"scenario": sc, "metric": met, "comparisons": len(g), "max_abs_dp_adj": d.max(), "median_abs_dp_adj": d.median(),
                     "changed_0.05": int(((g.p_adj < .05) != (g[f"p_{sc}_adj"] < .05)).sum()),
                     "changed_0.01": int(((g.p_adj < .01) != (g[f"p_{sc}_adj"] < .01)).sum()),
                     "sig_WHWK_0.05": int((g.p_adj < .05).sum()), "sig_scn_0.05": int((g[f"p_{sc}_adj"] < .05).sum())})
S = pd.DataFrame(summ)
S.to_csv(OUT / "deviations_summary.csv", index=False)
m.to_csv(OUT / "deviations_all.csv", index=False)
mt = m[m.metric == "MedAE"].medae_ties
print(S.round(5).to_string(index=False))
print(f"\nMedAE: comparisons with any tied bootstrap medians {int((mt > 0).sum())}/120; median ties per comparison {mt.median():.0f} of {B}; max {mt.max()}")
