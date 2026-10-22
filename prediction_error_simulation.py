"""
Monte Carlo simulation of IOL formula prediction errors with one or both eyes per patient.

This is the simulation behind:
    Moussa G, Cannon N, Cooke DL. Keeping both eyes in intraocular lens formula comparisons:
    development and validation of a cluster-aware statistical suite (submitted).

Each simulated study has 50-1000 patients, 30-50% of whom contribute both eyes. Two formulas (A and B)
are compared. In the null phase they are equally accurate, so every significant result is a false
positive. In the power phase formula B has a 7% larger SD, mostly in the tails, plus a cubic error in
short and long eyes. Every study is analysed with both eyes (patient-level tests) and with one eye per
patient (standard tests), and the p-value of every test is written to trials.csv.

Run:  python prediction_error_simulation.py            (full run, 10 000 + 5000 studies)
      python prediction_error_simulation.py --quick    (100 + 50 studies, to check it works)

The full run takes several hours on a desktop PC; it uses all but two CPU cores.
"""
import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import stats

# Number of studies and bootstrap resamples
N_NULL_SIMULATIONS = 10000
N_POWER_SIMULATIONS = 5000
BOOTSTRAP_ITERS = 3999

# Patients
BILATERAL_RATIO_MIN = 0.30      # share of patients with both eyes in a study, drawn from this range
BILATERAL_RATIO_MAX = 0.50
INTER_EYE_CORR = 0.30           # correlation of the prediction error between fellow eyes

# Prediction errors (calibrated to 2550 eyes and seven modern formulas)
BASE_SD = 0.45
INTER_FORMULA_CORR = 0.95
OUTLIER_FRACTION = 0.075        # share of eyes with a larger error
OUTLIER_MULTIPLIER = 2.2

# Power phase: formula B is made less accurate
CORE_DEGRADATION = 1.01         # every error widened by 1%
TOTAL_DEGRADATION = 1.07        # SD 7% larger overall; the rest of the widening goes into the tails
SIMULATE_AL_BIAS = True
AL_MEAN = 23.4                  # axial length (mm), UK National Ophthalmology Database
AL_SD = 1.33
AL_CUBIC_COEFF = 0.005          # error of 0.005 * (AL - 23.4)^3 D added to formula B

MASTER_SEED = 42
MAX_PATIENTS = 1005             # width of the shared resampling matrices


# ----------------------------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------------------------
def generate_study(n_patients=250, bilateral_ratio=0.40, seed=42):
    """One simulated study: a row per eye with the patient ID, axial length and both formulas' errors."""
    rng = np.random.default_rng(seed)
    n_formulas = 2
    base_sds = np.array([BASE_SD, BASE_SD])
    means = np.array([0.00, 0.00])

    cov = np.zeros((n_formulas, n_formulas))
    for i in range(n_formulas):
        for j in range(n_formulas):
            if i == j:
                cov[i, j] = base_sds[i] ** 2
            else:
                cov[i, j] = INTER_FORMULA_CORR * base_sds[i] * base_sds[j]

    eyes_per_patient = rng.choice([1, 2], size=n_patients, p=[1 - bilateral_ratio, bilateral_ratio])
    patient_als = rng.normal(loc=AL_MEAN, scale=AL_SD, size=n_patients)

    rows = []
    for p in range(n_patients):
        patient_id = f"PT_{p + 1:04d}"
        # shared patient component gives fellow eyes a correlation of INTER_EYE_CORR
        patient_effect = rng.multivariate_normal(np.zeros(n_formulas), cov * INTER_EYE_CORR)
        for _ in range(eyes_per_patient[p]):
            scale = OUTLIER_MULTIPLIER if rng.random() < OUTLIER_FRACTION else 1.0
            eye_effect = rng.multivariate_normal(np.zeros(n_formulas), cov * (1.0 - INTER_EYE_CORR))
            errors = means + (patient_effect + eye_effect) * scale
            rows.append({'Patient_ID': patient_id, 'AL': patient_als[p],
                         'Formula_A': errors[0], 'Formula_B': errors[1]})
    return pd.DataFrame(rows)


def widen_tails(e_base, core_deg, total_deg, tail_percentile=85, smoothness=10):
    """Widen the errors by core_deg everywhere, then stretch the tails (smoothly, above the 85th
    percentile of |error|) until the SD is total_deg times the original."""
    base_var = np.var(e_base, ddof=1)
    target_ss = base_var * (total_deg ** 2) * (len(e_base) - 1)

    e_mean = np.mean(e_base)
    e_core = (e_base - e_mean) * core_deg + e_mean
    e_mean_core = np.mean(e_core)
    e_centred = e_core - e_mean_core

    abs_e = np.abs(e_centred)
    threshold = np.percentile(abs_e, tail_percentile)
    w = 1.0 / (1.0 + np.exp(-np.clip(smoothness * (abs_e - threshold), -500, 500)))

    A = e_centred * (1.0 - w)
    B = e_centred * w
    # solve for the tail multiplier c that gives the target sum of squares
    a = np.sum(B ** 2)
    b = 2.0 * np.sum(A * B)
    d = np.sum(A ** 2) - target_ss
    disc = b ** 2 - 4 * a * d
    c = max(1.0, (-b + np.sqrt(disc)) / (2.0 * a)) if (disc > 0 and a > 0) else 1.0
    return A + c * B + e_mean_core


def less_accurate(e2, al):
    """Formula B in the power phase."""
    e2 = widen_tails(e2, CORE_DEGRADATION, TOTAL_DEGRADATION)
    if SIMULATE_AL_BIAS:
        e2 = e2 + AL_CUBIC_COEFF * ((al - AL_MEAN) ** 3)
    return e2


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------
def olshc4(x, y):
    """OLS slope test with HC4 standard errors. Adapted from Wilcox's olshc4 (Rallfun v45)."""
    n = len(y)
    if n < 3:
        return 1.0
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
    omega = (res ** 2) / denom

    sigma = XtX_inv @ ((X.T * omega) @ X) @ XtX_inv
    se_slope = np.sqrt(sigma[1, 1])
    if se_slope == 0:
        return 1.0
    t_stat = beta[1] / se_slope
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))


def comdvar(x, y):
    """Morgan-Pitman test of equal SDs (HC4). Adapted from Wilcox's comdvar (Rallfun v45)."""
    diff, summ = x - y, x + y
    var_d, var_s = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_d == 0 or var_s == 0:
        return 1.0
    z1 = (diff - np.mean(diff)) / np.sqrt(var_d)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_s)
    return olshc4(z1, z2)


def wild_cluster_slope(x, y, cluster_ids, W):
    """Wild cluster bootstrap-t for the slope of y on x (restricted to the null, Rademacher weights
    W of shape (B, patients), CR0 standard error; Cameron, Gelbach & Miller 2008)."""
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = cluster_idx.max() + 1
    B = W.shape[0]
    if K < 3:
        return 1.0, 0.0, np.zeros(B)

    N = len(x)
    x_bar, y_bar = np.mean(x), np.mean(y)
    dx = x - x_bar
    sum_dx2 = np.sum(dx ** 2)
    if sum_dx2 == 0:
        return 1.0, 0.0, np.zeros(B)

    beta1 = np.sum(dx * y) / sum_dx2
    eps_null = y - y_bar

    # patient sums, so every bootstrap sample is a matrix product rather than a loop
    A_g = np.bincount(cluster_idx, weights=dx * eps_null)
    B_g = np.bincount(cluster_idx, weights=dx)
    C_g = np.bincount(cluster_idx, weights=eps_null)
    D_g = np.bincount(cluster_idx, weights=dx ** 2)

    eps_hat = y - (y_bar + beta1 * dx)
    S_g = np.bincount(cluster_idx, weights=dx * eps_hat)
    var_beta1 = np.sum(S_g ** 2) / (sum_dx2 ** 2)
    if var_beta1 == 0:
        return 1.0, 0.0, np.zeros(B)
    t_obs = beta1 / np.sqrt(var_beta1)

    beta1_star = (W @ A_g) / sum_dx2
    y_bar_star = (W @ C_g) / N
    S_g_star = W * A_g - y_bar_star[:, None] * B_g - beta1_star[:, None] * D_g
    var_star = np.sum(S_g_star ** 2, axis=1) / (sum_dx2 ** 2)

    t_star = np.zeros(B)
    ok = var_star > 0
    t_star[ok] = beta1_star[ok] / np.sqrt(var_star[ok])
    p = float(np.mean(np.abs(t_star) >= np.abs(t_obs)))
    return p, np.abs(t_obs), np.abs(t_star)


def comdvar_wild_cluster(x, y, cluster_ids, W):
    """Morgan-Pitman test with a wild cluster bootstrap p-value (both eyes)."""
    diff, summ = x - y, x + y
    var_d, var_s = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_d == 0 or var_s == 0:
        return 1.0, 0.0, np.zeros(W.shape[0])
    z1 = (diff - np.mean(diff)) / np.sqrt(var_d)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_s)
    return wild_cluster_slope(z1, z2, cluster_ids, W)


def comdvar_percentile(x, y, cluster_ids, b_indices):
    """Morgan-Pitman slope with a percentile patient bootstrap (comparison only)."""
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    B = b_indices.shape[0]
    diff, summ = x - y, x + y
    var_d, var_s = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_d == 0 or var_s == 0:
        return 1.0
    z1 = (diff - np.mean(diff)) / np.sqrt(var_d)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_s)

    N_g = np.bincount(cluster_idx)
    Z1_g = np.bincount(cluster_idx, weights=z1)
    Z2_g = np.bincount(cluster_idx, weights=z2)
    Z11_g = np.bincount(cluster_idx, weights=z1 ** 2)
    Z12_g = np.bincount(cluster_idx, weights=z1 * z2)

    b_N = np.sum(N_g[b_indices], axis=1)
    b_Z1 = np.sum(Z1_g[b_indices], axis=1)
    b_Z2 = np.sum(Z2_g[b_indices], axis=1)
    b_Z11 = np.sum(Z11_g[b_indices], axis=1)
    b_Z12 = np.sum(Z12_g[b_indices], axis=1)
    b_z1_bar = b_Z1 / b_N

    num = b_Z12 - b_z1_bar * b_Z2
    den = b_Z11 - b_N * b_z1_bar ** 2
    slopes = np.zeros(B)
    ok = den > 0
    slopes[ok] = num[ok] / den[ok]
    return min(2 * min(np.mean(slopes > 0), np.mean(slopes < 0)), 1.0)


def mcnemar_yates(mask1, mask2):
    """McNemar test with continuity correction."""
    b = np.sum(mask1 & ~mask2)
    c = np.sum(~mask1 & mask2)
    if b == c:
        return 1.0
    return min(1.0, float(stats.chi2.sf(((abs(b - c) - 1.0) ** 2) / (b + c), 1)))


def mcnemar_uncorrected(mask1, mask2):
    """McNemar test without continuity correction (like-for-like comparator for Yang's test)."""
    b = np.sum(mask1 & ~mask2)
    c = np.sum(~mask1 & mask2)
    if b == c:
        return 1.0
    return min(1.0, float(stats.chi2.sf((b - c) ** 2 / (b + c), 1)))


def mcnemar_clustered(mask1, mask2, cluster_ids):
    """Yang, Sun & Hardin (2010) modified Obuchowski test: chi-square, 1 df, no continuity correction."""
    d = mask1.astype(int) - mask2.astype(int)
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = cluster_idx.max() + 1
    if K < 2:
        return 1.0
    D_k = np.bincount(cluster_idx, weights=d)
    n_k = np.bincount(cluster_idx)
    sum_D = np.sum(D_k)
    if sum_D == 0:
        return 1.0
    p_diff = sum_D / np.sum(n_k)
    denom = 0.5 * np.sum((D_k - n_k * p_diff) ** 2 + D_k ** 2)
    if denom == 0:
        return 1.0
    chi2_stat = ((K - 1) / K) * sum_D ** 2 / denom
    return min(1.0, float(stats.chi2.sf(chi2_stat, 1)))


def bootstrap_mae_rmse(e1, e2, cluster_ids, b_indices):
    """Patient bootstrap for MAE and RMSE. Returns the bootstrap-t (CR1 standard error) and percentile
    p-values: (MAE t, MAE percentile, RMSE t, RMSE percentile). RMSE is tested through the MSE."""
    abs_e1, abs_e2 = np.abs(e1), np.abs(e2)
    sq_e1, sq_e2 = e1 ** 2, e2 ** 2
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = cluster_idx.max() + 1
    B = b_indices.shape[0]

    S_mae = np.bincount(cluster_idx, weights=abs_e1 - abs_e2)
    S_mse = np.bincount(cluster_idx, weights=sq_e1 - sq_e2)
    S_sq1 = np.bincount(cluster_idx, weights=sq_e1)
    S_sq2 = np.bincount(cluster_idx, weights=sq_e2)
    N_g = np.bincount(cluster_idx)
    N = np.sum(N_g)

    obs_mae = np.sum(S_mae) / N
    obs_mse = np.sum(S_mse) / N

    def cr1_se(S, mean):
        res = S - mean * N_g
        var = (K / (K - 1)) * np.sum(res ** 2) / (N ** 2) if K > 1 else 0.0
        return np.sqrt(max(0.0, var))

    se_mae, se_mse = cr1_se(S_mae, obs_mae), cr1_se(S_mse, obs_mse)
    t_mae = obs_mae / se_mae if se_mae > 0 else 0.0
    t_mse = obs_mse / se_mse if se_mse > 0 else 0.0

    b_S_mae_mat, b_S_mse_mat, b_N_mat = S_mae[b_indices], S_mse[b_indices], N_g[b_indices]
    b_N = np.sum(b_N_mat, axis=1)
    boot_mae = np.sum(b_S_mae_mat, axis=1) / b_N
    boot_mse = np.sum(b_S_mse_mat, axis=1) / b_N
    with np.errstate(invalid='ignore'):
        boot_rmse = (np.sqrt(np.maximum(0.0, np.sum(S_sq1[b_indices], axis=1) / b_N))
                     - np.sqrt(np.maximum(0.0, np.sum(S_sq2[b_indices], axis=1) / b_N)))

    def boot_se(b_S_mat, boot_mean):
        res = b_S_mat - boot_mean[:, None] * b_N_mat
        var = (K / (K - 1)) * np.sum(res ** 2, axis=1) / (b_N ** 2) if K > 1 else np.zeros(B)
        return np.sqrt(np.maximum(0.0, var))

    se_b_mae, se_b_mse = boot_se(b_S_mae_mat, boot_mae), boot_se(b_S_mse_mat, boot_mse)

    p_mae_perc = 2 * min(np.mean(boot_mae > 0), np.mean(boot_mae < 0))
    p_rmse_perc = 2 * min(np.mean(boot_rmse > 0), np.mean(boot_rmse < 0))
    t_star_mae = np.where(se_b_mae > 0, (boot_mae - obs_mae) / se_b_mae, 0.0)
    t_star_mse = np.where(se_b_mse > 0, (boot_mse - obs_mse) / se_b_mse, 0.0)
    p_mae_t = np.mean(np.abs(t_star_mae) >= np.abs(t_mae))
    p_rmse_t = np.mean(np.abs(t_star_mse) >= np.abs(t_mse))
    return p_mae_t, p_mae_perc, p_rmse_t, p_rmse_perc


def bootstrap_medae(e1, e2, cluster_ids, b_indices):
    """Percentile patient bootstrap for the difference in median absolute error."""
    abs_e1, abs_e2 = np.abs(e1), np.abs(e2)
    _, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = cluster_idx.max() + 1
    B = b_indices.shape[0]

    if K == len(e1):
        boot = np.median(abs_e1[b_indices], axis=1) - np.median(abs_e2[b_indices], axis=1)
    else:
        # at most two eyes per patient: store as a (patients, 2) array padded with NaN
        e1_pad = np.full((K, 2), np.nan)
        e2_pad = np.full((K, 2), np.nan)
        counts = np.zeros(K, dtype=int)
        for i, c in enumerate(cluster_idx):
            e1_pad[c, counts[c]] = abs_e1[i]
            e2_pad[c, counts[c]] = abs_e2[i]
            counts[c] += 1
        with np.errstate(invalid='ignore'):
            boot = (np.nanmedian(e1_pad[b_indices].reshape(B, -1), axis=1)
                    - np.nanmedian(e2_pad[b_indices].reshape(B, -1), axis=1))

    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return 1.0
    return min(2 * min(np.mean(boot > 0), np.mean(boot < 0)), 1.0)


# ----------------------------------------------------------------------------------------------
# One study
# ----------------------------------------------------------------------------------------------
def resampling_matrices():
    """Rademacher weights and bootstrap draws shared by every study (drawn first from MASTER_SEED)."""
    rng = np.random.default_rng(MASTER_SEED)
    W = rng.choice([-1, 1], size=(BOOTSTRAP_ITERS, MAX_PATIENTS))
    R = rng.random(size=(BOOTSTRAP_ITERS, MAX_PATIENTS))
    return rng, W, R


def run_tests(e1, e2, cluster_ids, W, R, prefix, both_eyes):
    K = len(np.unique(cluster_ids))
    W_k = W[:, :K]
    b_idx = (R[:, :K] * K).astype(int)
    within1, within2 = np.abs(e1) <= 0.50, np.abs(e2) <= 0.50

    p = {'WCB': comdvar_wild_cluster(e1, e2, cluster_ids, W_k)[0],
         'HC4': comdvar(e1, e2)}
    p['MAE_stud'], p['MAE_perc'], p['RMSE_stud'], p['RMSE_perc'] = bootstrap_mae_rmse(e1, e2, cluster_ids, b_idx)
    p['SD_perc'] = comdvar_percentile(e1, e2, cluster_ids, b_idx)
    p['MEDAE'] = bootstrap_medae(e1, e2, cluster_ids, b_idx)
    try:
        p['WSR'] = stats.wilcoxon(np.abs(e1), np.abs(e2))[1]
    except ValueError:
        p['WSR'] = 1.0
    p['MCN_yates'] = mcnemar_yates(within1, within2)
    p['MCN_uncorr'] = mcnemar_uncorrected(within1, within2)
    if both_eyes:
        p['YANG'] = mcnemar_clustered(within1, within2, cluster_ids)
    return {f'{prefix}_{k}': v for k, v in p.items()}


_SHARED = {}


def _init_worker():
    _, _SHARED['W'], _SHARED['R'] = resampling_matrices()


def run_study(job):
    phase, i, n, ratio, seed = job
    df = generate_study(n_patients=n, bilateral_ratio=ratio, seed=seed)
    e1, e2, ids = df['Formula_A'].values, df['Formula_B'].values, df['Patient_ID'].values
    if phase == 'power':
        e2 = less_accurate(e2, df['AL'].values)
    row = dict(phase=phase, trial=i, N=n, ratio=ratio, eyes=len(e1), K=len(np.unique(ids)))
    row.update(run_tests(e1, e2, ids, _SHARED['W'], _SHARED['R'], 'C', True))      # both eyes
    _, first = np.unique(ids, return_index=True)                                    # one eye per patient
    row.update(run_tests(e1[first], e2[first], ids[first], _SHARED['W'], _SHARED['R'], 'I', False))
    return row


def study_list(n_null, n_power):
    """Study size, bilateral share and seed for every study, in the same random stream as the paper."""
    rng, _, _ = resampling_matrices()
    jobs = []
    for phase, n_studies in (('null', n_null), ('power', n_power)):
        for i in range(n_studies):
            n = int(rng.integers(50, 1001))
            ratio = float(rng.uniform(BILATERAL_RATIO_MIN, BILATERAL_RATIO_MAX))
            jobs.append((phase, i, n, ratio, rng.integers(0, 100000)))
    return jobs


def summarise(trials):
    tests = [c[2:] for c in trials.columns if c.startswith('C_')]
    lines = [f"{'Test':<12}{'Type I, both':>14}{'Type I, one':>13}{'Power, both':>13}{'Power, one':>12}"]
    for t in tests:
        vals = []
        for phase in ('null', 'power'):
            sub = trials[trials.phase == phase]
            for side in ('C', 'I'):
                col = f'{side}_{t}'
                vals.append(f"{100 * (sub[col] < 0.05).mean():.2f}%" if col in sub and len(sub) else '-')
        lines.append(f"{t:<12}{vals[0]:>14}{vals[1]:>13}{vals[2]:>13}{vals[3]:>12}")
    return '\n'.join(lines)


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    n_null, n_power = (100, 50) if quick else (N_NULL_SIMULATIONS, N_POWER_SIMULATIONS)
    jobs = study_list(N_NULL_SIMULATIONS, N_POWER_SIMULATIONS)
    if quick:
        jobs = [j for j in jobs if j[1] < (n_null if j[0] == 'null' else n_power)]

    workers = max(1, (os.cpu_count() or 4) - 2)
    rows = []
    with Pool(workers, initializer=_init_worker) as pool:
        for k, row in enumerate(pool.imap(run_study, jobs, chunksize=10), 1):
            rows.append(row)
            if k % 250 == 0 or k == len(jobs):
                print(f'{k} / {len(jobs)} studies', flush=True)

    trials = pd.DataFrame(rows)
    trials.to_csv('trials.csv', index=False)
    print('\nPercentage of studies with p < 0.05 (C = both eyes, I = one eye per patient)\n')
    print(summarise(trials))
