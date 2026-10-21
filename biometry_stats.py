"""
biometry_stats: compare the prediction errors of IOL power formulas, with one or both eyes per patient.

https://github.com/gmeyeltd/biometry-stats
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
from concurrent.futures import ThreadPoolExecutor

# Defaults. Most can be overridden through the settings dict passed to analyze_prediction_errors.
BOOTSTRAP_ITERATIONS = 4999   # with p = (1 + b)/(B + 1), p-values fall on a grid of 1/5000

# Fixed seed so results are reproducible; set to None for a random seed
BOOTSTRAP_SEED = 42

if BOOTSTRAP_SEED is None:
    BOOTSTRAP_SEED = int(np.random.default_rng().integers(0, 2**32))
    print(f"--- [Biometry Stats] Random Bootstrap Seed Generated: {BOOTSTRAP_SEED} ---")
else:
    print(f"--- [Biometry Stats] Using Fixed Bootstrap Seed: {BOOTSTRAP_SEED} ---")

# Used when the settings dict does not supply a value
YUEN_TRIM_PROPORTION = 0.2
MCNEMAR_THRESHOLD = 0.50
MULTIPLE_TEST_METHOD = 'romano_wolf'
FALLBACK_TEST_METHOD = 'hommel'
INVALID_THRESHOLD = 4.0
AXL_SHORT_CUTOFF = 22.0
AXL_LONG_CUTOFF = 26.0
MIN_GROUP_SIZE = 5      # a group (All, Short, Medium, Long) with fewer valid eyes than this is not tested
MEDIAN_WINDOW = 5       # bootstrap medians: half-width of the search window, in sqrt(n) eyes (speed only; results are exact)
RUN_GROUPS_IN_PARALLEL = True   # test All / Short / Medium / Long side by side (speed only; results are identical)

# Which tests to run
USE_STUDENTIZED_BS = True
ENFORCE_INDEPENDENCE = False
RUN_YUEN_TEST = False
RUN_SD_TEST = True
REPORT_CI = False    # report each difference between formulas with a bootstrap confidence interval
CI_LEVEL = 0.95

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def _calculate_percentile_bootstrap_p(boot_diffs):
    """Two-sided percentile bootstrap p-value (MedAE, or any metric when studentising is off)."""
    valid_diffs = boot_diffs[~np.isnan(boot_diffs)]
    if len(valid_diffs) == 0:
        return 1.0
    tail = min(np.sum(valid_diffs > 0), np.sum(valid_diffs < 0))
    p = 2 * (1 + tail) / (len(valid_diffs) + 1)
    return min(p, 1.0)

def _bootstrap_p(null_stats, obs_stat):
    """Bootstrap p-value (1 + #{null >= obs}) / (B + 1); never exactly 0 (Davison & Hinkley 1997)."""
    return (1 + np.sum(null_stats >= obs_stat)) / (len(null_stats) + 1)

def mcnemar_test(mask1, mask2):
    """McNemar test with continuity correction (one eye per patient)."""
    b = np.sum(mask1 & ~mask2)
    c = np.sum(~mask1 & mask2)
    n = b + c
    if b == c: return 1.0
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    p_val = stats.chi2.sf(chi2_stat, 1)
    return min(1.0, float(p_val))

def clustered_mcnemar_test(mask1, mask2, cluster_ids):
    """Yang, Sun & Hardin (2010) modified Obuchowski test for bilateral (clustered) data.

    Per cluster k: D_k = b_k - c_k (discordant-pair difference), n_k eyes.
    The variance averages the centred (Obuchowski) and null-centred (D_k^2) forms:
        X2 = ((K-1)/K) * (sum D_k)^2 / (0.5 * sum[(D_k - n_k*(p1 - p2))^2 + D_k^2])
    referred to chi-square with 1 df (no continuity correction).
    """
    d = mask1.astype(int) - mask2.astype(int)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    if K < 2: return 1.0

    D_k = np.bincount(cluster_idx, weights=d)
    n_k = np.bincount(cluster_idx)
    sum_D = np.sum(D_k)
    if sum_D == 0: return 1.0

    p_diff = sum_D / np.sum(n_k)
    denom = 0.5 * np.sum((D_k - n_k * p_diff)**2 + D_k**2)
    if denom == 0: return 1.0

    chi2_stat = ((K - 1) / K) * sum_D**2 / denom
    return min(1.0, float(stats.chi2.sf(chi2_stat, 1)))

def wild_cluster_sd_all_pairs(X, cluster_idx, K, W_g, pairs=None, chunk=500):
    """Morgan-Pitman test of equal SDs with a wild cluster bootstrap-t, for every formula pair at once.

    For each pair, the standardized difference and sum of the two prediction errors are regressed on each other;
    the slope is tested with a wild cluster bootstrap restricted to the null (Cameron, Gelbach & Miller 2008),
    Rademacher weights W_g (B x K) and a CR0 standard error. With weights of +/-1, every bootstrap slope and
    standard error is a function of cluster sums, so all resamples and all pairs come from one matrix product.
    Returns {(j, k): (p, |t_obs|, |t*|)}.
    """
    X = np.asarray(X, float)
    n, J = X.shape
    B = W_g.shape[0]
    pairs = pairs or list(combinations(range(J), 2))
    out, cols, meta = {}, [], {}
    for (j, k) in pairs:
        diff, summ = X[:, j] - X[:, k], X[:, j] + X[:, k]
        var_d, var_s = np.var(diff, ddof=1), np.var(summ, ddof=1)
        if var_d == 0 or var_s == 0 or K < 3:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        x = (diff - np.mean(diff)) / np.sqrt(var_d)
        y = (summ - np.mean(summ)) / np.sqrt(var_s)
        x_bar, y_bar = np.mean(x), np.mean(y)
        dx = x - x_bar
        sxx = np.sum(dx ** 2)
        if sxx == 0:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        beta_hat = np.sum(dx * y) / sxx
        eps_hat = y - (y_bar + beta_hat * dx)
        var_b = np.sum(np.bincount(cluster_idx, weights=dx * eps_hat, minlength=K) ** 2) / sxx ** 2
        if var_b == 0:
            out[(j, k)] = (1.0, 0.0, np.zeros(B)); continue
        t_obs = beta_hat / np.sqrt(var_b)
        e_t = y - y_bar                                                     # null-restricted residuals
        A = np.bincount(cluster_idx, weights=dx * e_t, minlength=K)        # per-cluster sums
        E = np.bincount(cluster_idx, weights=e_t, minlength=K)
        D = np.bincount(cluster_idx, weights=dx, minlength=K)
        Q = np.bincount(cluster_idx, weights=dx ** 2, minlength=K)
        meta[(j, k)] = dict(c=len(cols), t_obs=t_obs, y_bar=y_bar, sxx=sxx, sdx=np.sum(dx),
                            DD=np.sum(D * D), AA=np.sum(A * A), QQ=np.sum(Q * Q), DQ=np.sum(D * Q))
        cols += [A, E, D * A, A * Q]
    if cols:
        V = np.column_stack(cols)
        G = np.empty((B, V.shape[1]))
        for i in range(0, B, chunk):
            G[i:i + chunk] = W_g[i:i + chunk].astype(float) @ V
        for key, m in meta.items():
            c = m["c"]
            wA, wE, wDA, wAQ = G[:, c], G[:, c + 1], G[:, c + 2], G[:, c + 3]
            beta_s = (m["y_bar"] * m["sdx"] + wA) / m["sxx"]                # bootstrap slope
            a = -wE / n                                                      # y_bar - bootstrap mean of y
            ss = (a ** 2 * m["DD"] + m["AA"] + beta_s ** 2 * m["QQ"]
                  + 2 * a * wDA - 2 * a * beta_s * m["DQ"] - 2 * beta_s * wAQ)  # sum of squared cluster scores
            var_s = np.maximum(ss, 0) / m["sxx"] ** 2
            t_s = np.zeros(B); ok = var_s > 0
            t_s[ok] = beta_s[ok] / np.sqrt(var_s[ok])
            out[key] = (float(_bootstrap_p(np.abs(t_s), np.abs(m["t_obs"]))), np.abs(m["t_obs"]), np.abs(t_s))
    return out

def comdvar_wild_cluster(x, y, cluster_idx, K, W_g):
    """Morgan-Pitman test of equal SDs with a wild cluster bootstrap p-value (one pair)."""
    return wild_cluster_sd_all_pairs(np.column_stack((x, y)), cluster_idx, K, W_g)[(0, 1)]

def olshc4(x, y):
    """OLS slope test with HC4 standard errors. Adapted from Wilcox's olshc4 (Rallfun v45)."""
    n = len(y)
    if n < 3: return 1.0

    X = np.column_stack((np.ones(n), x))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 1.0

    h = np.sum((X @ XtX_inv) * X, axis=1)
    beta = XtX_inv @ X.T @ y
    res = y - X @ beta

    sum_h = np.sum(h)
    d = (n * h) / sum_h if sum_h != 0 else np.zeros_like(h)
    d = np.minimum(4, d)

    denom = np.power(1 - h, d)
    denom[denom < 1e-10] = 1e-10
    omega = (res**2) / denom

    middle = (X.T * omega) @ X
    Sigma = XtX_inv @ middle @ XtX_inv

    se_slope = np.sqrt(Sigma[1, 1])
    if se_slope == 0: return 1.0

    t_stat = beta[1] / se_slope
    df = n - 2
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

def comdvar(x, y):
    """Morgan-Pitman test of equal SDs with HC4 standard errors (one eye per patient).
    Adapted from Wilcox's comdvar (Rallfun v45)."""
    diff = x - y
    summ = x + y
    var_x, var_y = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_x == 0 or var_y == 0: return 1.0
    z1 = (diff - np.mean(diff)) / np.sqrt(var_x)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_y)
    return olshc4(z1, z2)

def _resample_counts(idx_chunk, K):
    """Count matrix: how often each cluster (patient or eye) appears in each resample of the block."""
    c = idx_chunk.shape[0]
    flat = (idx_chunk + (np.arange(c)[:, None] * K)).ravel()
    return np.bincount(flat, minlength=c * K).reshape(c, K).astype(np.float64)


def all_pairs_bootstrap(X, cluster_idx, K, b_indices, use_studentized=True, pairs=None, chunk=None, ci_level=None, prop_threshold=None):
    """Bootstrap tests for MPE, MAE, RMSE and MedAE for every formula pair at once.

    X is an n x J matrix of prediction errors (one column per formula). Returns {(j, k): result} for j < k,
    each result having the same keys as perform_advanced_bootstrap. Resamples, statistics and p-value rules are
    the same as a pair-by-pair calculation; only the arithmetic is organized differently. Each block of resamples becomes a count matrix W (resamples x clusters), so every
    bootstrap sum the bootstrap-t needs is one matrix product W @ V, and each formula's bootstrap median is read
    from running totals of W over that formula's sorted errors (one median per formula, not per pair).

    If ci_level is given (e.g. 0.95), each result also holds confidence intervals for the difference between the two
    formulas, from the same resamples: bootstrap-t for MPE and MAE (as their tests), percentile for RMSE, MedAE, SD and,
    when prop_threshold is given, the proportion of eyes with |PE| <= prop_threshold. They are not adjusted for
    multiple comparisons, and the p-values are unaffected.
    """
    X = np.asarray(X, float)
    n, J = X.shape
    B = b_indices.shape[0]
    pairs = pairs or list(combinations(range(J), 2))
    N_g = np.bincount(cluster_idx, minlength=K).astype(float)
    N_tot = N_g.sum()

    # per-cluster sums for each formula and each family (abs error, squared error, signed error)
    fam = {"MAE": np.abs(X), "MSE": X ** 2, "MPE": X}
    S = {f: np.stack([np.bincount(cluster_idx, weights=v[:, j], minlength=K) for j in range(J)], axis=1) for f, v in fam.items()}
    means = {f: S[f].sum(0) / N_tot for f in fam}                      # observed per-formula means
    C = {f: S[f] - np.outer(N_g, means[f]) for f in fam}               # centred per-cluster sums (better precision)

    # columns of V: for each family, centred sums (J), and cross-products for every pair incl. squares
    tri = [(a, b) for a in range(J) for b in range(a, J)]
    cols, index = [N_g], {}
    for f in fam:
        index[(f, "s")] = len(cols); cols += [C[f][:, j] for j in range(J)]
        index[(f, "x")] = len(cols); cols += [C[f][:, a] * C[f][:, b] for a, b in tri]
        index[(f, "n")] = len(cols); cols += [C[f][:, j] * N_g for j in range(J)]
    index["nn"] = len(cols); cols.append(N_g ** 2)
    index[("MSE", "raw")] = len(cols); cols += [S["MSE"][:, j] for j in range(J)]   # for RMSE differences
    if ci_level:
        index["SQ"] = len(cols)                                                    # sums of squared deviations (SD)
        cols += [np.bincount(cluster_idx, weights=(X[:, j] - means["MPE"][j]) ** 2, minlength=K) for j in range(J)]
        if prop_threshold is not None:
            index["PROP"] = len(cols)                                              # eyes with |PE| <= threshold
            cols += [np.bincount(cluster_idx, weights=(np.abs(X[:, j]) <= prop_threshold).astype(float), minlength=K) for j in range(J)]
    V = np.column_stack(cols)
    tpos = {t: i for i, t in enumerate(tri)}

    # observed statistics, computed exactly as the original engine does
    def cr1_se(Svec, mean_val):
        res = Svec - mean_val * N_g
        return np.sqrt(max(0.0, (K / (K - 1)) * np.sum(res ** 2) / N_tot ** 2)) if K > 1 else 0.0

    # medians: one per formula per resample, read from running totals of the counts over each formula's sorted values
    absX = np.abs(X)
    order = [np.argsort(absX[:, j], kind="stable") for j in range(J)]
    sorted_vals = [absX[order[j], j] for j in range(J)]
    eye_cluster = [cluster_idx[order[j]] for j in range(J)]
    max_c = int(N_g.max())

    # A resample's median almost always lies close to the middle of the original sorted errors, so the running
    # totals are only needed over a window of sorted positions around the middle (+/- 5 sqrt(n) + 16 eyes): the eyes
    # drawn below the window come from one product with per-cluster counts. A resample whose middle rank falls outside
    # the window is redone over all the values, so every median is exactly the one the full calculation gives.
    n_eyes = len(cluster_idx)
    half = int(np.ceil(MEDIAN_WINDOW * np.sqrt(n_eyes))) + 16
    w_start, w_end = max(0, n_eyes // 2 - half), min(n_eyes, n_eyes // 2 + half + 1)
    windowed = (w_end - w_start) < n_eyes
    if windowed:
        below = [np.bincount(eye_cluster[j][:w_start], minlength=K).astype(float) for j in range(J)]
        in_window = [eye_cluster[j][w_start:w_end] for j in range(J)]

    def _positions(Wj, lo, hi, base=None):
        """Sorted positions of the lo-th and hi-th drawn eye (running totals over the columns of Wj)."""
        c = Wj.shape[0]
        cum = np.cumsum(Wj, axis=1)                                      # eyes drawn up to each sorted value
        if base is not None:
            cum += base[:, None]
        big = int(max(cum[:, -1].max(), hi.max())) + 1
        offs = np.arange(c) * big
        flat = (cum + offs[:, None]).ravel()
        start = np.arange(c) * cum.shape[1]
        pos_lo = np.searchsorted(flat, lo + offs, side="left") - start
        pos_hi = np.searchsorted(flat, hi + offs, side="left") - start
        return pos_lo, pos_hi

    def _medians(W):
        c = W.shape[0]
        out = np.empty((c, J))
        nb = W @ N_g                                                     # eyes in each resample
        lo = np.floor((nb + 1) / 2).astype(np.int64)                     # ranks of the middle value(s), 1-based
        hi = np.floor(nb / 2).astype(np.int64) + 1
        for j in range(J):
            if windowed:
                base = W @ below[j]                                      # eyes drawn below the window
                Wj = W[:, in_window[j]]
                pos_lo, pos_hi = _positions(Wj, lo, hi, base)
                inside = (base < lo) & (base + Wj.sum(axis=1) >= hi)
                pos_lo = np.clip(pos_lo, 0, w_end - w_start - 1) + w_start
                pos_hi = np.clip(pos_hi, 0, w_end - w_start - 1) + w_start
                if not inside.all():                                     # rare: redo these resamples in full
                    redo = ~inside
                    pos_lo[redo], pos_hi[redo] = _positions(W[redo][:, eye_cluster[j]], lo[redo], hi[redo])
            else:
                pos_lo, pos_hi = _positions(W[:, eye_cluster[j]], lo, hi)
            out[:, j] = (sorted_vals[j][pos_lo] + sorted_vals[j][pos_hi]) / 2
        return out

    chunk = chunk or max(1, min(500, int(2e7 // max(1, int(N_tot)))))
    sums = np.empty((B, V.shape[1]))
    med = np.empty((B, J))
    for i in range(0, B, chunk):
        idx = b_indices[i:i + chunk]
        W = _resample_counts(idx, K)
        sums[i:i + chunk] = W @ V
        med[i:i + chunk] = _medians(W)

    bN = sums[:, 0]
    a_ci = (1 - ci_level) / 2 if ci_level else None
    qci = lambda v: tuple(float(x) for x in np.quantile(v, [a_ci, 1 - a_ci]))
    out = {}
    for (j, k) in pairs:
        r = {}
        for f, key in (("MPE", "MPE"), ("MAE", "MAE"), ("MSE", "RMSE")):
            d_mean_obs = means[f][j] - means[f][k]
            Sd = S[f][:, j] - S[f][:, k]
            se_obs = cr1_se(Sd, d_mean_obs)
            t_obs = d_mean_obs / se_obs if se_obs > 0 else 0.0
            # bootstrap: centred difference sums and CR1 variance from the matrix products
            s = index[(f, "s")]; x = index[(f, "x")]; nn_ = index[(f, "n")]
            sd = sums[:, s + j] - sums[:, s + k]                                    # sum_g w (Cj - Ck)
            sdd = sums[:, x + tpos[(j, j)]] + sums[:, x + tpos[(k, k)]] - 2 * sums[:, x + tpos[(min(j, k), max(j, k))]]
            sdn = sums[:, nn_ + j] - sums[:, nn_ + k]
            snn = sums[:, index["nn"]]
            m = sd / bN                                                              # boot mean minus observed mean
            ssq = sdd - 2 * m * sdn + m ** 2 * snn                                   # sum_g w (S'_g - m N_g)^2
            var = (K / (K - 1)) * np.maximum(ssq, 0) / bN ** 2 if K > 1 else np.zeros(B)
            se = np.sqrt(var)
            if ci_level:
                if key == "RMSE":                                                    # percentile, on the RMSE scale
                    q = index[("MSE", "raw")]
                    r[f"CI_{key}"] = qci(np.sqrt(np.maximum(0, sums[:, q + j] / bN)) - np.sqrt(np.maximum(0, sums[:, q + k] / bN)))
                elif use_studentized:                                                # bootstrap-t
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ts = np.where(se > 0, m / se, 0.0)
                    lo_q, hi_q = qci(ts)
                    r[f"CI_{key}"] = (d_mean_obs - hi_q * se_obs, d_mean_obs - lo_q * se_obs)
                else:
                    r[f"CI_{key}"] = qci(m + d_mean_obs)
            if use_studentized:
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_star = np.where(se > 0, m / se, 0.0)
                r[f"P_{key}"] = _bootstrap_p(np.abs(t_star), np.abs(t_obs))
                r[f"RW_{key}_obs"] = np.abs(t_obs); r[f"RW_{key}_null"] = np.abs(t_star)
            else:
                if key == "RMSE":
                    q = index[("MSE", "raw")]
                    boot = np.sqrt(np.maximum(0, sums[:, q + j] / bN)) - np.sqrt(np.maximum(0, sums[:, q + k] / bN))
                    obs = np.sqrt(means["MSE"][j]) - np.sqrt(means["MSE"][k])
                else:
                    boot, obs = m + d_mean_obs, d_mean_obs
                r[f"P_{key}"] = _calculate_percentile_bootstrap_p(boot)
                r[f"RW_{key}_obs"] = abs(obs); r[f"RW_{key}_null"] = np.abs(boot - obs)
        boot_medae = med[:, j] - med[:, k]
        obs_medae = np.median(absX[:, j]) - np.median(absX[:, k])
        r["P_MedAE"] = _calculate_percentile_bootstrap_p(boot_medae)
        if ci_level:
            r["CI_MedAE"] = qci(boot_medae)
            s0, sq = index[("MPE", "s")], index["SQ"]
            with np.errstate(divide="ignore", invalid="ignore"):
                sd_j = np.sqrt(np.maximum(0, (sums[:, sq + j] - sums[:, s0 + j] ** 2 / bN) / (bN - 1)))
                sd_k = np.sqrt(np.maximum(0, (sums[:, sq + k] - sums[:, s0 + k] ** 2 / bN) / (bN - 1)))
            r["CI_SD"] = qci(sd_j - sd_k)
            if prop_threshold is not None:
                pc = index["PROP"]
                r["CI_PROP"] = qci((sums[:, pc + j] - sums[:, pc + k]) / bN)
        if use_studentized:
            se_m = max(np.std(boot_medae, ddof=1), 1e-6)
            r["RW_MedAE_obs"] = abs(obs_medae) / se_m; r["RW_MedAE_null"] = np.abs(boot_medae - obs_medae) / se_m
        else:
            r["RW_MedAE_obs"] = abs(obs_medae); r["RW_MedAE_null"] = np.abs(boot_medae - obs_medae)
        out[(j, k)] = r
    return out


def perform_advanced_bootstrap(e1, e2, cluster_idx, K, b_indices, W_g=None, use_studentized=True):
    """Patient (cluster) bootstrap for MPE, MAE, RMSE and MedAE for one formula pair.

    MPE, MAE and RMSE use the bootstrap-t with a CR1 standard error (RMSE is tested through the MSE);
    MedAE uses the percentile bootstrap. Also returns the statistics the Romano-Wolf correction needs.
    """
    return all_pairs_bootstrap(np.column_stack((e1, e2)), cluster_idx, K, b_indices, use_studentized=use_studentized)[(0, 1)]



def yuen_t_test(x, y, tr=0):
    """Yuen's paired test on trimmed means (optional, off by default)."""
    x, y = np.array(x), np.array(y)
    d = x - y
    n = len(d)

    g = int(tr * n)
    d_sort = np.sort(d)
    d_win = d_sort.copy()
    if g > 0:
        d_win[:g] = d_win[g]
        d_win[-g:] = d_win[-g-1]

    win_var = np.var(d_win, ddof=1)
    h = n - 2 * g
    se_tr = np.sqrt((n - 1) * win_var / (h * (h - 1)))

    trimmed_mean_diff = np.mean(d_sort[g:-g]) if g > 0 else np.mean(d)
    t_stat = trimmed_mean_diff / se_tr if se_tr != 0 else 0
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), h - 1)), trimmed_mean_diff

def romano_wolf_correction(obs_stats, boot_null_stats):
    """Romano-Wolf step-down adjusted p-values from the bootstrap null distributions of |t|."""
    M = len(obs_stats)
    if M == 0: return np.array([])

    adj_p = np.zeros(M)
    order = np.argsort(obs_stats)[::-1]
    sorted_nulls = boot_null_stats[:, order]

    max_nulls = np.maximum.accumulate(sorted_nulls[:, ::-1], axis=1)[:, ::-1]

    for i in range(M):
        p_val = _bootstrap_p(max_nulls[:, i], obs_stats[order[i]])
        adj_p[order[i]] = p_val

    for i in range(1, M):
        adj_p[order[i]] = max(adj_p[order[i]], adj_p[order[i-1]])

    return adj_p

# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------

def run_biometry_analysis(df_input, group_name="All", settings=None):
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    bs_iterations = settings.get('bootstrap_iterations', BOOTSTRAP_ITERATIONS)
    invalid_threshold = settings.get('invalid_threshold', INVALID_THRESHOLD)
    mcnemar_threshold = settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD)
    report_ci = settings.get('report_ci', REPORT_CI)
    ci_level = float(settings.get('ci_level', CI_LEVEL)) if report_ci else None

    numeric_df = df_input.select_dtypes(include=[np.number])
    formulas = []

    patient_id_variants = ['patient_id', 'id_patient', 'patientid', 'pt_id', 'id', 'hosp_id', 'hospid', 'ptid']

    # biometry columns that are not prediction errors
    ignore_variants = ['axl', 'al', 'acd', 'acdepth', 'k1', 'k2', 'meank', 'age', 'gender', 'gender_num',
                       'eye', 'side', 'lt', 'lensthickness', 'cct', 'wtw', 'hcd', 'iol_implanted',
                       'postoperative_refraction', 'corneal_refractive_index']

    patient_col = None
    for c in df_input.columns:
        c_lower = str(c).lower()
        if c_lower in patient_id_variants:
            patient_col = c
            continue

        # skip metadata, predicted refraction and lens-constant columns
        if c_lower in ignore_variants or c_lower.startswith('predict_') or c_lower.endswith('_const'):
            continue

        if c in numeric_df.columns:
            formulas.append(c)

    if len(formulas) < 2: return pd.DataFrame(), {}, {}

    valid_mask = pd.Series(True, index=numeric_df.index)
    for f in formulas:
        valid_mask &= ~np.isnan(numeric_df[f])
        valid_mask &= np.abs(numeric_df[f]) <= invalid_threshold

    if patient_col:
        valid_mask &= df_input[patient_col].notna()
        if ENFORCE_INDEPENDENCE:
            rng_indep = np.random.default_rng(BOOTSTRAP_SEED)
            valid_patients = df_input.loc[valid_mask, patient_col]
            shuffled_idx = rng_indep.permutation(valid_patients.index)
            selected_idx = valid_patients.loc[shuffled_idx].drop_duplicates().index
            new_mask = pd.Series(False, index=numeric_df.index)
            new_mask.loc[selected_idx] = True
            valid_mask &= new_mask
            print(f"[{group_name}] ENFORCE_INDEPENDENCE Active: Dropped to 1 eye per patient. Analyzing {len(selected_idx)} independent rows.")
        else: print(f"[{group_name}] Cluster Bootstrap Active: Resampling by '{patient_col}'.")
    else: print(f"[{group_name}] Standard Bootstrap Active: Resampling independently.")

    valid_df = numeric_df[valid_mask]
    n_global = len(valid_df)

    min_group_size = max(2, int(settings.get('min_group_size', MIN_GROUP_SIZE)))
    if n_global < min_group_size:
        print(f"[{group_name}] Not tested: {n_global} valid eye(s), fewer than the minimum of {min_group_size}.")
        return pd.DataFrame(), {}, {}

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # IDs as strings so that mixed types sort
    cluster_ids = df_input.loc[valid_mask, patient_col].astype(str).values if patient_col else np.arange(n_global)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    has_clusters = K < n_global

    # resample indices and wild-bootstrap weights, generated in blocks and stored compactly (int32 / int8);
    # the values are identical to drawing them in one call
    global_b_indices = np.empty((bs_iterations, K), dtype=np.int32)
    for i in range(0, bs_iterations, 500):
        global_b_indices[i:i + 500] = rng.choice(K, size=(min(500, bs_iterations - i), K), replace=True)
    global_W_g = np.empty((bs_iterations, K), dtype=np.int8)
    for i in range(0, bs_iterations, 500):
        global_W_g[i:i + 500] = rng.choice([-1, 1], size=(min(500, bs_iterations - i), K))

    needs_bootstrap_nulls = fwer_method == 'romano_wolf'
    needs_wcb = RUN_SD_TEST and (has_clusters or needs_bootstrap_nulls)

    pairs = list(combinations(formulas, 2))

    bs_type = "Studentized (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized" if USE_STUDENTIZED_BS else "Percentile"
    print(f"[{group_name}] Processing {len(pairs)} comparisons ({bs_type}, N={n_global}, Iterations={bs_iterations})...")

    # bootstrap tests for every formula pair at once (matrix form; identical to pair-by-pair results)
    f_index = {f: i for i, f in enumerate(formulas)}
    boot_all = all_pairs_bootstrap(valid_df[formulas].values, cluster_idx, K, global_b_indices, use_studentized=USE_STUDENTIZED_BS,
                                   ci_level=ci_level, prop_threshold=mcnemar_threshold)
    sd_all = wild_cluster_sd_all_pairs(valid_df[formulas].values, cluster_idx, K, global_W_g) if (RUN_SD_TEST and needs_wcb) else None

    # remaining per-pair tests (SD, McNemar) run in threads
    def process_pair_task(pair):
        f1, f2 = pair
        e1, e2 = valid_df[f1].values, valid_df[f2].values

        boot_res = boot_all[(f_index[f1], f_index[f2])]

        if RUN_SD_TEST:
            if needs_wcb: p_sd, obs_sd, null_sd = sd_all[(f_index[f1], f_index[f2])]
            else: p_sd = comdvar(e1, e2); obs_sd = 0.0; null_sd = np.zeros(bs_iterations)
        else: p_sd = np.nan; obs_sd = 0.0; null_sd = np.zeros(bs_iterations)

        mask1, mask2 = np.abs(e1) <= mcnemar_threshold, np.abs(e2) <= mcnemar_threshold
        if has_clusters: p_mcnemar = clustered_mcnemar_test(mask1, mask2, cluster_ids)
        else: p_mcnemar = mcnemar_test(mask1, mask2)

        if RUN_YUEN_TEST:
            _, diff_mae = yuen_t_test(np.abs(e1), np.abs(e2), tr=0)
            yuen_mae_p, trim_mae_diff = yuen_t_test(np.abs(e1), np.abs(e2), tr=YUEN_TRIM_PROPORTION)
            yuen_rmse_p, trim_sq_diff = yuen_t_test(e1**2, e2**2, tr=YUEN_TRIM_PROPORTION)
        else:
            diff_mae = np.mean(np.abs(e1)) - np.mean(np.abs(e2))

        row = {
            'Group': group_name, 'Formula A': f1, 'Formula B': f2, 'N': n_global,
            'MPE Diff': np.mean(e1) - np.mean(e2), 'Boot (MPE) p': boot_res['P_MPE'],
            'MAE Diff': diff_mae, 'Boot (MAE) p': boot_res['P_MAE'],
            'MedAE Diff': np.median(np.abs(e1)) - np.median(np.abs(e2)), 'Boot (MedAE) p': boot_res['P_MedAE'],
            'RMSE Diff': np.sqrt(np.mean(e1**2)) - np.sqrt(np.mean(e2**2)), 'Boot (RMSE) p': boot_res['P_RMSE'],
            'SD Diff': np.std(e1, ddof=1) - np.std(e2, ddof=1),
            'Morgan-Pitman (SD) p': p_sd,
            'McNemar Diff': np.mean(mask1) - np.mean(mask2),
            'McNemar p': p_mcnemar
        }
        if ci_level:
            # confidence interval columns straight after each difference
            lbl = f"{ci_level * 100:g}% CI"
            ci_key = {'MPE Diff': 'CI_MPE', 'MAE Diff': 'CI_MAE', 'MedAE Diff': 'CI_MedAE', 'RMSE Diff': 'CI_RMSE',
                      'SD Diff': 'CI_SD', 'McNemar Diff': 'CI_PROP'}
            new_row = {}
            for col, val in row.items():
                new_row[col] = val
                if col in ci_key and ci_key[col] in boot_res:
                    new_row[f'{col} {lbl} low'], new_row[f'{col} {lbl} high'] = boot_res[ci_key[col]]
            row = new_row
        if RUN_YUEN_TEST:
            row['Trimmed MAE Diff'] = trim_mae_diff; row['Yuen (MAE) p'] = yuen_mae_p; row['Yuen (RMSE) p'] = yuen_rmse_p

        return {
            'row': row,
            'rw_mpe_obs': boot_res['RW_MPE_obs'], 'rw_mpe_null': boot_res['RW_MPE_null'],
            'rw_mae_obs': boot_res['RW_MAE_obs'], 'rw_mae_null': boot_res['RW_MAE_null'],
            'rw_rmse_obs': boot_res['RW_RMSE_obs'], 'rw_rmse_null': boot_res['RW_RMSE_null'],
            'rw_medae_obs': boot_res['RW_MedAE_obs'], 'rw_medae_null': boot_res['RW_MedAE_null'],
            'sd_obs': obs_sd, 'sd_null': null_sd
        }

    results = []
    rw_stats = {m: {'obs': [], 'null': []} for m in ['MPE', 'MAE', 'RMSE', 'MedAE', 'SD']}

    with ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(process_pair_task, pairs):
            results.append(res['row'])
            for m in ['MPE', 'MAE', 'RMSE', 'MedAE']:
                rw_stats[m]['obs'].append(res[f'rw_{m.lower()}_obs'])
                rw_stats[m]['null'].append(res[f'rw_{m.lower()}_null'])
            rw_stats['SD']['obs'].append(res['sd_obs'])
            rw_stats['SD']['null'].append(res['sd_null'])

    methodology = {
        'Dependence Strategy': f"Bilateral Clustering (n={bs_iterations} Resamples)" if has_clusters else "Independent Eyes",
        'Precision Test (SD)': "Morgan-Pitman (Wild Cluster Bootstrap)" if needs_wcb else "Morgan-Pitman (HC4 Asymptotic)",
        'Accuracy Test (McNemar)': "Yang's Mod. Obuchowski (Cluster-Robust)" if has_clusters else "Asymptotic McNemar",
        'MPE Test': "Studentized Cluster Bootstrap (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
        'MAE Test': "Studentized Cluster Bootstrap (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
        'MedAE Test': "Cluster Percentile Bootstrap" if has_clusters else "Percentile Bootstrap",
        'RMSE Test': "Studentized Cluster Bootstrap on MSE (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
    }
    if ci_level:
        methodology['Confidence Intervals'] = (f"{ci_level * 100:g}% bootstrap intervals for each difference: bootstrap-t (MPE, MAE), "
                                               "percentile (RMSE, MedAE, SD, proportion); not adjusted for multiple comparisons")

    df_results = pd.DataFrame(results)
    # eyes and patients actually tested in this group (used to label figures)
    df_results.attrs['group_info'] = {'n_eyes': int(n_global), 'n_patients': int(K), 'clustered': bool(has_clusters),
                                      'patient_column': patient_col}
    return df_results, rw_stats, methodology

def adjust_pvalues(df_results, rw_stats=None, method='hommel'):
    """Add adjusted p-value columns.

    Romano-Wolf is used where bootstrap null distributions exist; other columns use
    FALLBACK_TEST_METHOD (Hommel by default).
    """
    if df_results.empty: return df_results

    rw_metric_mapping = {
        'Boot (MPE) p': 'MPE', 'Boot (MAE) p': 'MAE',
        'Boot (MedAE) p': 'MedAE', 'Boot (RMSE) p': 'RMSE',
        'Morgan-Pitman (SD) p': 'SD'
    }

    p_cols = [c for c in df_results.columns if ' p' in c and 'adj' not in c]

    for col in p_cols:
        pvals = df_results[col].values
        pvals_adj = np.full(len(pvals), np.nan)
        mask = ~np.isnan(pvals)

        # a single comparison needs no adjustment
        if np.sum(mask) <= 1:
            pvals_adj[mask] = pvals[mask]
            use_fallback = False
        else:
            use_fallback = True

        # Romano-Wolf
        if method == 'romano_wolf' and col in rw_metric_mapping and rw_stats is not None and use_fallback:
            metric = rw_metric_mapping[col]
            obs = np.array(rw_stats[metric]['obs'])
            null_list = rw_stats[metric]['null']

            if len(obs) > 1 and len(null_list) > 0:
                null_matrix = np.column_stack(null_list)
                if np.sum(np.abs(null_matrix)) > 0:
                    pvals_adj_rw = romano_wolf_correction(obs, null_matrix)
                    pvals_adj[mask] = pvals_adj_rw[mask]
                    use_fallback = False

        # otherwise Hommel, Holm, etc.
        if use_fallback:
            current_method = FALLBACK_TEST_METHOD if method == 'romano_wolf' else method
            if np.sum(mask) > 0:
                _, adj, _, _ = multipletests(pvals[mask], method=current_method)
                pvals_adj[mask] = adj

        # an adjusted p-value is never smaller than the raw one
        if np.sum(mask) > 0:
            pvals_adj[mask] = np.maximum(pvals_adj[mask], pvals[mask])

        try:
            idx = df_results.columns.get_loc(col) + 1
            new_col_name = col.replace(' p', ' p_adj')
            if new_col_name in df_results.columns:
                  df_results[new_col_name] = pvals_adj
            else:
                df_results.insert(idx, new_col_name, pvals_adj)
        except Exception:
            pass

    return df_results

def analyze_prediction_errors(df_input, axl_data=None, settings=None):
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    axl_short = settings.get('axl_short', AXL_SHORT_CUTOFF)
    axl_long = settings.get('axl_long', AXL_LONG_CUTOFF)

    def _process_group(df_sub, name):
        if df_sub.empty: return None
        res_df, rw_stats, methodology = run_biometry_analysis(
            df_sub,
            group_name=name,
            settings=settings
        )
        if res_df.empty: return None
        info = dict(res_df.attrs.get('group_info', {}), methodology=methodology)
        res_df = adjust_pvalues(res_df, rw_stats=rw_stats, method=fwer_method)
        return res_df, info

    groups = [("All", df_input)]

    if axl_data is not None:
        if isinstance(df_input, pd.DataFrame) and hasattr(axl_data, 'index'):
            common_idx = df_input.index.intersection(axl_data.index)
            df_aligned = df_input.loc[common_idx]
            axl_aligned = axl_data.loc[common_idx].iloc[:, 0] if isinstance(axl_data, pd.DataFrame) else axl_data.loc[common_idx]
        else:
            df_aligned, axl_aligned = df_input, axl_data

        axl_vals = np.array(axl_aligned)

        mask_short = axl_vals <= axl_short
        mask_long = axl_vals >= axl_long
        mask_medium = (axl_vals > axl_short) & (axl_vals < axl_long)

        if np.sum(mask_short) > 0:
            groups.append(("Short", df_aligned.iloc[mask_short]))

        if np.sum(mask_medium) > 0:
            groups.append(("Medium", df_aligned.iloc[mask_medium]))

        if np.sum(mask_long) > 0:
            groups.append(("Long", df_aligned.iloc[mask_long]))

    # The groups are independent (each draws its own resamples from BOOTSTRAP_SEED), so they run side by side;
    # results are collected in the order above, identical to running them one after another.
    print(f"--- Running Analysis: {', '.join(name for name, _ in groups)} ---")
    if RUN_GROUPS_IN_PARALLEL and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            outcomes = list(executor.map(lambda g: _process_group(g[1], g[0]), groups))
    else:
        outcomes = [_process_group(df_sub, name) for name, df_sub in groups]

    all_results_frames, group_info = [], {}
    for (name, _), outcome in zip(groups, outcomes):
        if outcome is not None:
            all_results_frames.append(outcome[0])
            group_info[name] = outcome[1]

    if not all_results_frames: return pd.DataFrame()

    final_df = pd.concat(all_results_frames, ignore_index=True)

    # The methods table describes the first group tested (All eyes); where a subgroup used a different method
    # (e.g. it has no bilateral pairs), that is added to the entry.
    base_name = next(iter(group_info))
    global_methodology = dict(group_info[base_name]['methodology'])
    for name, info in group_info.items():
        if name == base_name: continue
        for k, v in info['methodology'].items():
            if k not in global_methodology:
                global_methodology[k] = f"{name} eyes: {v}"
            elif v != group_info[base_name]['methodology'].get(k):
                global_methodology[k] += f"; {name} eyes: {v}"

    # describe the multiplicity correction in the methodology table
    if global_methodology:
        if fwer_method == 'romano_wolf': fwer_name = 'Romano-Wolf'
        elif fwer_method == 'fdr_bh': fwer_name = 'FDR-BH'
        else: fwer_name = fwer_method.capitalize()

        if fwer_method == 'romano_wolf':
            global_methodology['FWER Correction'] = f"{fwer_name} (Bootstrapped Tests) / Hommel (Asymptotic Fallback)"
        else:
            global_methodology['FWER Correction'] = fwer_name

        # put the thresholds into the labels
        mc_t = float(settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD))
        inv_t = float(settings.get('invalid_threshold', INVALID_THRESHOLD))
        labelled = {}
        for k, v in global_methodology.items():
            if k == 'Accuracy Test (McNemar)':
                labelled[f'Accuracy Test (McNemar, |PE| ≤ {mc_t:.2f} D)'] = f"{v} – proportion of eyes with |PE| ≤ {mc_t:.2f} D"
            else:
                labelled[k] = v
        labelled['Invalid PE Exclusion'] = f"Eyes with |PE| > {inv_t:.2f} D excluded from the pairwise dependence tests"
        global_methodology = labelled

        summary_data = [
            {'Test Category': k, 'Methodology Applied': v}
            for k, v in global_methodology.items()
        ]
        # kept as a list of dicts (a DataFrame in attrs breaks pd.concat)
        final_df.attrs['methodology_summary'] = summary_data
    final_df.attrs['mcnemar_threshold'] = float(settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD))
    # per group: eyes and patients tested, whether bilateral clustering was used, and the tests applied
    final_df.attrs['group_info'] = group_info
    final_df.attrs['fwer_method'] = fwer_method

    return final_df

# ------------------------------------------------------------------------------
# Excel output
# ------------------------------------------------------------------------------
def _set_column_widths(writer, sheet_name, widths):
    """Column widths for an openpyxl or an xlsxwriter ExcelWriter sheet."""
    try:
        ws = writer.sheets[sheet_name]
        if hasattr(ws, 'set_column'):                                   # xlsxwriter
            for idx, w in enumerate(widths):
                ws.set_column(idx, idx, w)
        else:                                                           # openpyxl
            from openpyxl.utils import get_column_letter
            for idx, w in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(idx)].width = w
    except (ImportError, AttributeError, KeyError):
        pass

def write_advanced_stats_to_excel(df_results, writer, sheet_name='Advanced_Stats'):
    """Write the results to an Excel sheet and the tests used to a 'Statistical Tests' sheet."""
    if df_results.empty:
        print("Warning: DataFrame is empty. Nothing to write.")
        return

    cols_to_drop = ['Precision Test (SD)', 'Accuracy Test (McNemar)', 'MAE Test', 'Dependence Strategy', 'FWER Correction']
    df_clean = df_results.drop(columns=[c for c in cols_to_drop if c in df_results.columns], errors='ignore')

    # e.g. "McNemar p_adj" -> "McNemar (≤0.50D) p_adj" so the tested threshold travels with the numbers
    mc_t = df_results.attrs.get('mcnemar_threshold') if hasattr(df_results, 'attrs') else None
    if mc_t is not None:
        df_clean = df_clean.rename(columns={c: c.replace('McNemar', f'McNemar (≤{float(mc_t):.2f}D)')
                                            for c in df_clean.columns if 'McNemar' in c})

    df_clean.to_excel(writer, sheet_name=sheet_name, index=False)

    _set_column_widths(writer, sheet_name, [15 if ' p' in col else 22 for col in df_clean.columns])

    if hasattr(df_results, 'attrs') and 'methodology_summary' in df_results.attrs:
        summary_data = df_results.attrs['methodology_summary']
        if isinstance(summary_data, list):
            meth_df = pd.DataFrame(summary_data)
        elif isinstance(summary_data, pd.DataFrame):
            meth_df = summary_data
        else:
            meth_df = pd.DataFrame()

        if not meth_df.empty:
            meth_sheet = 'Statistical Tests'
            meth_df.to_excel(writer, sheet_name=meth_sheet, index=False)

            _set_column_widths(writer, meth_sheet, [35 if idx == 0 else 75 for idx in range(len(meth_df.columns))])
