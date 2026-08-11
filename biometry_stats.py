import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BOOTSTRAP_ITERATIONS = 2000

# Seed Configuration
BOOTSTRAP_SEED = 42

if BOOTSTRAP_SEED is None:
    BOOTSTRAP_SEED = int(np.random.default_rng().integers(0, 2**32))
    print(f"--- [Biometry Stats] Random Bootstrap Seed Generated: {BOOTSTRAP_SEED} ---")
else:
    print(f"--- [Biometry Stats] Using Fixed Bootstrap Seed: {BOOTSTRAP_SEED} ---")

# Default Global Fallbacks (Only used if settings dict is missing values)
YUEN_TRIM_PROPORTION = 0.2
MCNEMAR_THRESHOLD = 0.50
MULTIPLE_TEST_METHOD = 'romano_wolf'
FALLBACK_TEST_METHOD = 'hommel'
INVALID_THRESHOLD = 4.0
AXL_SHORT_CUTOFF = 22.0
AXL_LONG_CUTOFF = 26.0

# Statistical Methods
USE_STUDENTIZED_BS = True       
ENFORCE_INDEPENDENCE = False
RUN_YUEN_TEST = False           
RUN_SD_TEST = True              

# ==============================================================================
# 1. CORE STATISTICAL FUNCTIONS
# ==============================================================================

def _calculate_percentile_bootstrap_p(boot_diffs):
    """Fallback percentile evaluation for MedAE or non-studentized runs."""
    valid_diffs = boot_diffs[~np.isnan(boot_diffs)]
    if len(valid_diffs) == 0: 
        return 1.0
    p = 2 * min(np.mean(valid_diffs > 0), np.mean(valid_diffs < 0))
    return min(p, 1.0)

def mcnemar_test(mask1, mask2):
    """Computes the asymptotic McNemar test with continuity correction."""
    b = np.sum(mask1 & ~mask2)
    c = np.sum(~mask1 & mask2)
    n = b + c
    if b == c: return 1.0
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / n
    p_val = stats.chi2.sf(chi2_stat, 1)
    return min(1.0, float(p_val))

def clustered_mcnemar_test(mask1, mask2, cluster_ids):
    """Yang's Modified Obuchowski Test (2010) - Cluster Robust"""
    d = mask1.astype(int) - mask2.astype(int)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    
    D_i = np.bincount(cluster_idx, weights=d)
    sum_D = np.sum(D_i)
    var_D = np.sum(D_i**2)
    
    if var_D == 0: return 1.0
        
    adj_var_D = var_D * (K / (K - 1)) if K > 1 else var_D
    num = max(0.0, abs(sum_D) - 1.0)
    t_stat = num / np.sqrt(adj_var_D)
    
    p_val = 2 * stats.t.sf(t_stat, K - 1) if K > 1 else 1.0
    return min(1.0, float(p_val))

def _wild_cluster_bootstrap_slope(x, y, cluster_idx, K, W_g):
    """
    ULTRA-FAST SPARSE VECTORIZED WCB-t for bivariate OLS slope.
    Bypasses the 2000-iteration loop via sparse matrix multiplication.
    """
    B = W_g.shape[0]
    N = len(x)
    if K < 3: return 1.0, 0.0, np.zeros(B) 
    
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    dx = x - x_bar
    sum_dx2 = np.sum(dx**2)
    if sum_dx2 == 0: return 1.0, 0.0, np.zeros(B)
    
    # Original sample stat
    beta1_hat = np.sum(dx * y) / sum_dx2
    eps_hat = y - (y_bar + beta1_hat * dx)
    
    # CR0 SE
    S_g = np.bincount(cluster_idx, weights=dx * eps_hat)
    var_beta1 = np.sum(S_g**2) / (sum_dx2**2)
    if var_beta1 == 0: return 1.0, 0.0, np.zeros(B)
    t_obs = beta1_hat / np.sqrt(var_beta1)
    
    # Sparse Indicator Matrix for ultra-fast cluster grouping without loop
    I_sparse = csr_matrix((np.ones(N), (np.arange(N), cluster_idx)), shape=(N, K))
    
    eps_tilde = y - y_bar 
    t_stars = np.zeros(B)
    
    CHUNK_SIZE = 500
    for i in range(0, B, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, B)
        W_chunk = W_g[i:end, cluster_idx] # Shape: (chunk, N)
        
        y_star = y_bar + W_chunk * eps_tilde # (chunk, N)
        y_star_bar = np.mean(y_star, axis=1, keepdims=True) # (chunk, 1)
        
        beta1_star = np.sum(dx * y_star, axis=1, keepdims=True) / sum_dx2 # (chunk, 1)
        eps_star = y_star - (y_star_bar + beta1_star * dx) # (chunk, N)
        
        # (chunk, N) @ (N, K) -> (chunk, K). Replaces inner bincount loop entirely!
        S_g_star = (dx * eps_star) @ I_sparse
        var_beta1_star = np.sum(S_g_star**2, axis=1) / (sum_dx2**2) # (chunk,)
        
        valid = var_beta1_star > 0
        t_stars[i:end][valid] = beta1_star.flatten()[valid] / np.sqrt(var_beta1_star[valid])
            
    p_val = np.mean(np.abs(t_stars) >= np.abs(t_obs))
    return float(p_val), np.abs(t_obs), np.abs(t_stars)

def comdvar_wild_cluster(x, y, cluster_idx, K, W_g):
    """Test hypothesis that two dependent variables have equal variances using Morgan-Pitman + Wild Cluster Bootstrap."""
    diff = x - y
    summ = x + y
    
    var_d = np.var(diff, ddof=1)
    var_s = np.var(summ, ddof=1)
    B = W_g.shape[0]
    if var_d == 0 or var_s == 0: 
        return 1.0, 0.0, np.zeros(B)
        
    z1 = (diff - np.mean(diff)) / np.sqrt(var_d)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_s)
    
    return _wild_cluster_bootstrap_slope(z1, z2, cluster_idx, K, W_g)

def olshc4(x, y):
    """Independent HC4 Robust Regression Estimator for Morgan-Pitman."""
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
    """Independent HC4 Morgan-Pitman test for variance equality."""
    diff = x - y
    summ = x + y
    var_x, var_y = np.var(diff, ddof=1), np.var(summ, ddof=1)
    if var_x == 0 or var_y == 0: return 1.0
    z1 = (diff - np.mean(diff)) / np.sqrt(var_x)
    z2 = (summ - np.mean(summ)) / np.sqrt(var_y)
    return olshc4(z1, z2)

def perform_advanced_bootstrap(e1, e2, cluster_idx, K, b_indices, W_g, use_studentized=True):
    """
    Highly optimized cluster bootstrap engine.
    - Smooth metrics (MAE, RMSE, MPE) use CR1 Studentized Bootstrap.
    - MedAE uses NaN-Padded Matrix Vectorization to bypass list comprehensions entirely.
    """
    B = b_indices.shape[0]
    
    abs_e1, abs_e2 = np.abs(e1), np.abs(e2)
    sq_e1, sq_e2 = e1**2, e2**2
    
    d_mpe = e1 - e2
    d_mae = abs_e1 - abs_e2
    d_mse = sq_e1 - sq_e2
    
    # 1. PRIMARY INFERENCE FOR MAE/RMSE/MPE
    S_mpe = np.bincount(cluster_idx, weights=d_mpe)
    S_mae = np.bincount(cluster_idx, weights=d_mae)
    S_mse = np.bincount(cluster_idx, weights=d_mse)
    S_sq1 = np.bincount(cluster_idx, weights=sq_e1)
    S_sq2 = np.bincount(cluster_idx, weights=sq_e2)
    N_g = np.bincount(cluster_idx)
    N_total = np.sum(N_g)
    
    obs_mpe = np.sum(S_mpe) / N_total
    obs_mae = np.sum(S_mae) / N_total
    obs_mse = np.sum(S_mse) / N_total
    obs_medae = np.median(abs_e1) - np.median(abs_e2)
    obs_rmse_diff = np.sqrt(np.mean(sq_e1)) - np.sqrt(np.mean(sq_e2))
    
    def calc_cr1_se(S_vec, N_vec, mean_val, K_val, N_tot):
        res = S_vec - mean_val * N_vec
        var = (K_val / (K_val - 1)) * np.sum(res**2) / (N_tot**2) if K_val > 1 else 0.0
        return np.sqrt(max(0.0, var))
        
    obs_se_mpe = calc_cr1_se(S_mpe, N_g, obs_mpe, K, N_total)
    obs_se_mae = calc_cr1_se(S_mae, N_g, obs_mae, K, N_total)
    obs_se_mse = calc_cr1_se(S_mse, N_g, obs_mse, K, N_total)
    
    t_obs_mpe = obs_mpe / obs_se_mpe if obs_se_mpe > 0 else 0.0
    t_obs_mae = obs_mae / obs_se_mae if obs_se_mae > 0 else 0.0
    t_obs_mse = obs_mse / obs_se_mse if obs_se_mse > 0 else 0.0
    
    b_S_mpe = np.zeros(B); b_S_mae = np.zeros(B); b_S_mse = np.zeros(B)
    b_S_sq1 = np.zeros(B); b_S_sq2 = np.zeros(B); b_N = np.zeros(B)
    var_mpe = np.zeros(B); var_mae = np.zeros(B); var_mse = np.zeros(B)

    CHUNK_SIZE = 500
    for i in range(0, B, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, B)
        b_idx_chunk = b_indices[i:end]

        b_S_mpe_mat = S_mpe[b_idx_chunk]
        b_S_mae_mat = S_mae[b_idx_chunk]
        b_S_mse_mat = S_mse[b_idx_chunk]
        b_N_mat = N_g[b_idx_chunk]

        b_S_mpe_c = np.sum(b_S_mpe_mat, axis=1)
        b_S_mae_c = np.sum(b_S_mae_mat, axis=1)
        b_S_mse_c = np.sum(b_S_mse_mat, axis=1)
        b_N_c = np.sum(b_N_mat, axis=1)

        b_S_mpe[i:end] = b_S_mpe_c
        b_S_mae[i:end] = b_S_mae_c
        b_S_mse[i:end] = b_S_mse_c
        b_S_sq1[i:end] = np.sum(S_sq1[b_idx_chunk], axis=1)
        b_S_sq2[i:end] = np.sum(S_sq2[b_idx_chunk], axis=1)
        b_N[i:end] = b_N_c

        if use_studentized:
            boot_mpe_c = b_S_mpe_c / b_N_c
            boot_mae_c = b_S_mae_c / b_N_c
            boot_mse_c = b_S_mse_c / b_N_c

            res_mpe = b_S_mpe_mat - boot_mpe_c[:, None] * b_N_mat
            var_mpe[i:end] = (K / (K - 1)) * np.sum(res_mpe**2, axis=1) / (b_N_c**2) if K > 1 else 0.0

            res_mae = b_S_mae_mat - boot_mae_c[:, None] * b_N_mat
            var_mae[i:end] = (K / (K - 1)) * np.sum(res_mae**2, axis=1) / (b_N_c**2) if K > 1 else 0.0

            res_mse = b_S_mse_mat - boot_mse_c[:, None] * b_N_mat
            var_mse[i:end] = (K / (K - 1)) * np.sum(res_mse**2, axis=1) / (b_N_c**2) if K > 1 else 0.0
            
    boot_mpe = b_S_mpe / b_N
    boot_mae = b_S_mae / b_N
    boot_mse = b_S_mse / b_N
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        boot_rmse_diffs = np.sqrt(np.maximum(0.0, b_S_sq1 / b_N)) - np.sqrt(np.maximum(0.0, b_S_sq2 / b_N))
    
    # =========================================================================
    # 2. PRIMARY INFERENCE FOR MedAE: NaN-Padded Matrix Vectorization
    # =========================================================================
    sort_idx = np.argsort(cluster_idx)
    e1_sorted = abs_e1[sort_idx]
    e2_sorted = abs_e2[sort_idx]
    
    max_c_size = np.max(N_g)
    
    if max_c_size <= 10:
        # SUPER FAST: Pad jagged clusters into a uniform NaN-matrix and broadcast
        e1_padded = np.full((K, max_c_size), np.nan)
        e2_padded = np.full((K, max_c_size), np.nan)
        
        split_points = np.cumsum(N_g)[:-1]
        e1_clusters = np.split(e1_sorted, split_points)
        e2_clusters = np.split(e2_sorted, split_points)
        
        for k_idx in range(K):
            size = N_g[k_idx]
            e1_padded[k_idx, :size] = e1_clusters[k_idx]
            e2_padded[k_idx, :size] = e2_clusters[k_idx]
            
        boot_medae = np.zeros(B)
        for i in range(0, B, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, B)
            b_idx_chunk = b_indices[i:end] # (chunk, K)
            
            chunk_e1 = e1_padded[b_idx_chunk].reshape(end - i, -1) # (chunk, K * max_c_size)
            chunk_e2 = e2_padded[b_idx_chunk].reshape(end - i, -1)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                boot_medae[i:end] = np.nanmedian(chunk_e1, axis=1) - np.nanmedian(chunk_e2, axis=1)
    else:
        # Fallback ONLY if dataset has massive illogical clusters (e.g. 10+ eyes per patient)
        split_points = np.cumsum(N_g)[:-1]
        e1_clusters = np.split(e1_sorted, split_points)
        e2_clusters = np.split(e2_sorted, split_points)
        
        boot_medae = np.zeros(B)
        for i in range(B):
            c_idxs = b_indices[i]
            b_samp1 = np.concatenate([e1_clusters[idx] for idx in c_idxs])
            b_samp2 = np.concatenate([e2_clusters[idx] for idx in c_idxs])
            boot_medae[i] = np.median(b_samp1) - np.median(b_samp2)
            
    p_medae = _calculate_percentile_bootstrap_p(boot_medae)

    # =========================================================================
    # RESOLVE T-STATS FOR ALL METRICS
    # =========================================================================
    if use_studentized:
        boot_mpe_se = np.sqrt(np.maximum(0.0, var_mpe))
        boot_mae_se = np.sqrt(np.maximum(0.0, var_mae))
        boot_mse_se = np.sqrt(np.maximum(0.0, var_mse))
        
        # Suppress divide by zero warnings since we explicitly catch zeros via np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stars_mpe = np.where(boot_mpe_se > 0, (boot_mpe - obs_mpe) / boot_mpe_se, 0.0)
            t_stars_mae = np.where(boot_mae_se > 0, (boot_mae - obs_mae) / boot_mae_se, 0.0)
            t_stars_mse = np.where(boot_mse_se > 0, (boot_mse - obs_mse) / boot_mse_se, 0.0)
            
        p_mpe = np.mean(np.abs(t_stars_mpe) >= np.abs(t_obs_mpe))
        p_mae = np.mean(np.abs(t_stars_mae) >= np.abs(t_obs_mae))
        p_rmse = np.mean(np.abs(t_stars_mse) >= np.abs(t_obs_mse))
        
        rw_mpe_obs = np.abs(t_obs_mpe)
        rw_mpe_null = np.abs(t_stars_mpe)
        
        rw_mae_obs = np.abs(t_obs_mae)
        rw_mae_null = np.abs(t_stars_mae)
        
        rw_rmse_obs = np.abs(t_obs_mse)
        rw_rmse_null = np.abs(t_stars_mse)
        
        boot_se_medae = np.std(boot_medae, ddof=1)
        boot_se_medae = max(boot_se_medae, 1e-6)  # Safeguard against zero-variance collapse
        
        rw_medae_obs = np.abs(obs_medae) / boot_se_medae
        rw_medae_null = np.abs(boot_medae - obs_medae) / boot_se_medae
            
    else:
        p_mpe = _calculate_percentile_bootstrap_p(boot_mpe)
        p_mae = _calculate_percentile_bootstrap_p(boot_mae)
        p_rmse = _calculate_percentile_bootstrap_p(boot_rmse_diffs)
        
        rw_mpe_obs = np.abs(obs_mpe); rw_mpe_null = np.abs(boot_mpe - obs_mpe)
        rw_mae_obs = np.abs(obs_mae); rw_mae_null = np.abs(boot_mae - obs_mae)
        rw_rmse_obs = np.abs(obs_rmse_diff); rw_rmse_null = np.abs(boot_rmse_diffs - obs_rmse_diff)
        rw_medae_obs = np.abs(obs_medae); rw_medae_null = np.abs(boot_medae - obs_medae)

    return {
        'P_MPE': p_mpe, 'P_MAE': p_mae, 'P_RMSE': p_rmse, 'P_MedAE': p_medae,
        'RW_MPE_obs': rw_mpe_obs, 'RW_MPE_null': rw_mpe_null,
        'RW_MAE_obs': rw_mae_obs, 'RW_MAE_null': rw_mae_null,
        'RW_RMSE_obs': rw_rmse_obs, 'RW_RMSE_null': rw_rmse_null,
        'RW_MedAE_obs': rw_medae_obs, 'RW_MedAE_null': rw_medae_null,
    }


def yuen_t_test(x, y, tr=0):
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
    M = len(obs_stats)
    if M == 0: return np.array([])
    
    adj_p = np.zeros(M)
    order = np.argsort(obs_stats)[::-1]
    sorted_nulls = boot_null_stats[:, order]
    
    max_nulls = np.maximum.accumulate(sorted_nulls[:, ::-1], axis=1)[:, ::-1]
    
    for i in range(M):
        p_val = np.mean(max_nulls[:, i] >= obs_stats[order[i]])
        adj_p[order[i]] = p_val
        
    for i in range(1, M):
        adj_p[order[i]] = max(adj_p[order[i]], adj_p[order[i-1]])
        
    return adj_p

# ==============================================================================
# 2. MAIN ANALYSIS LOGIC WITH TRUE MULTITHREADING
# ==============================================================================

def run_biometry_analysis(df_input, group_name="All", settings=None):
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    bs_iterations = settings.get('bootstrap_iterations', BOOTSTRAP_ITERATIONS)
    invalid_threshold = settings.get('invalid_threshold', INVALID_THRESHOLD)
    mcnemar_threshold = settings.get('mcnemar_threshold', MCNEMAR_THRESHOLD)

    numeric_df = df_input.select_dtypes(include=[np.number])
    formulas = []
    
    patient_id_variants = ['patient_id', 'id_patient', 'patientid', 'pt_id', 'id', 'hosp_id', 'hospid', 'ptid']
    
    # Standard biometry metadata that should NEVER be treated as a formula PE
    ignore_variants = ['axl', 'al', 'acd', 'acdepth', 'k1', 'k2', 'meank', 'age', 'gender', 'gender_num', 
                       'eye', 'side', 'lt', 'lensthickness', 'cct', 'wtw', 'hcd', 'iol_implanted', 
                       'postoperative_refraction', 'corneal_refractive_index']
                       
    patient_col = None
    for c in df_input.columns:
        c_lower = str(c).lower()
        if c_lower in patient_id_variants:
            patient_col = c
            continue
            
        # Ignore known metadata, prediction columns, or optimized constants
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
    
    if n_global < 2: return pd.DataFrame(), {}, {}
        
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    
    # Cast to string to prevent mixed-type sorting errors in NumPy unique function
    cluster_ids = df_input.loc[valid_mask, patient_col].astype(str).values if patient_col else np.arange(n_global)
    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    K = len(unique_clusters)
    has_clusters = K < n_global
    
    global_b_indices = rng.choice(K, size=(bs_iterations, K), replace=True)
    global_W_g = rng.choice([-1, 1], size=(bs_iterations, K))
    
    needs_bootstrap_nulls = fwer_method == 'romano_wolf'
    needs_wcb = RUN_SD_TEST and (has_clusters or needs_bootstrap_nulls)
    
    pairs = list(combinations(formulas, 2))
    
    bs_type = "Studentized (CR1)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized" if USE_STUDENTIZED_BS else "Percentile"
    print(f"[{group_name}] Processing {len(pairs)} comparisons ({bs_type}, N={n_global}, Iterations={bs_iterations})...")
    
    # ---------------------------------------------------------
    # MULTITHREADED PAIR PROCESSING
    # ---------------------------------------------------------
    def process_pair_task(pair):
        f1, f2 = pair
        e1, e2 = valid_df[f1].values, valid_df[f2].values
        
        boot_res = perform_advanced_bootstrap(e1, e2, cluster_idx, K, global_b_indices, global_W_g, use_studentized=USE_STUDENTIZED_BS)
        
        if RUN_SD_TEST:
            if needs_wcb: p_sd, obs_sd, null_sd = comdvar_wild_cluster(e1, e2, cluster_idx, K, global_W_g)
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
        'MAE Test': "Absolute Error Trimming (Studentized CR1 Bootstrap)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
        'MedAE Test': "Cluster Percentile Bootstrap (Matrix Vectorized)" if has_clusters else "Percentile Bootstrap",
        'RMSE Test': "Mean Squared Error Proxy (Studentized CR1 Bootstrap)" if (USE_STUDENTIZED_BS and has_clusters) else "Studentized Bootstrap" if USE_STUDENTIZED_BS else "Percentile Bootstrap",
    }
        
    return pd.DataFrame(results), rw_stats, methodology

def adjust_pvalues(df_results, rw_stats=None, method='hommel'):
    """
    Robust Multiple Testing Adjuster
    Applies Romano-Wolf if requested AND a valid null distribution exists.
    Otherwise, gracefully falls back to FALLBACK_TEST_METHOD (e.g., Hommel) for that specific column.
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
        
        # 0) If only 1 comparison (e.g., 2 formulas), FWER is mathematically meaningless. Skip it.
        if np.sum(mask) <= 1:
            pvals_adj[mask] = pvals[mask]
            use_fallback = False
        else:
            use_fallback = True
        
        # 1) Try Romano-Wolf Correction
        if method == 'romano_wolf' and col in rw_metric_mapping and rw_stats is not None and use_fallback:
            metric = rw_metric_mapping[col]
            obs = np.array(rw_stats[metric]['obs'])
            null_list = rw_stats[metric]['null']
            
            # Only run Romano-Wolf if we have multiple comparisons (M > 1)
            if len(obs) > 1 and len(null_list) > 0:
                null_matrix = np.column_stack(null_list)
                if np.sum(np.abs(null_matrix)) > 0:
                    pvals_adj_rw = romano_wolf_correction(obs, null_matrix)
                    pvals_adj[mask] = pvals_adj_rw[mask]
                    use_fallback = False
        
        # 2) Fallback to Standard Corrections
        if use_fallback:
            current_method = FALLBACK_TEST_METHOD if method == 'romano_wolf' else method
            if np.sum(mask) > 0:
                _, adj, _, _ = multipletests(pvals[mask], method=current_method)
                pvals_adj[mask] = adj
                
        # Enforce mathematical bounds of FWER: Adjusted p-value MUST be >= Raw p-value
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
    # Retrieve Thread-Safe Settings
    settings = settings or {}
    fwer_method = settings.get('fwer_method', MULTIPLE_TEST_METHOD)
    axl_short = settings.get('axl_short', AXL_SHORT_CUTOFF)
    axl_long = settings.get('axl_long', AXL_LONG_CUTOFF)

    all_results_frames = []
    global_methodology = {}
    
    def _process_group(df_sub, name):
        if df_sub.empty: return
        res_df, rw_stats, methodology = run_biometry_analysis(
            df_sub, 
            group_name=name, 
            settings=settings
        )
        if not res_df.empty:
            res_df = adjust_pvalues(res_df, rw_stats=rw_stats, method=fwer_method)
            all_results_frames.append(res_df)
            global_methodology.update(methodology)

    print("--- Running Analysis: All ---")
    _process_group(df_input, "All")
        
    if axl_data is not None:
        print("--- Running Analysis: Subgroups (AL) ---")
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
            _process_group(df_aligned.iloc[mask_short], "Short")
            
        if np.sum(mask_medium) > 0: 
            _process_group(df_aligned.iloc[mask_medium], "Medium")
            
        if np.sum(mask_long) > 0: 
            _process_group(df_aligned.iloc[mask_long], "Long")
                
    if not all_results_frames: return pd.DataFrame()
        
    final_df = pd.concat(all_results_frames, ignore_index=True)
    
    # Properly assign the correct fallback string for the methodology table
    if global_methodology:
        if fwer_method == 'romano_wolf': fwer_name = 'Romano-Wolf'
        elif fwer_method == 'fdr_bh': fwer_name = 'FDR-BH'
        else: fwer_name = fwer_method.capitalize()
        
        if fwer_method == 'romano_wolf':
            global_methodology['FWER Correction'] = f"{fwer_name} (Bootstrapped Tests) / Hommel (Asymptotic Fallback)"
        else:
            global_methodology['FWER Correction'] = fwer_name
            
        summary_data = [
            {'Test Category': k, 'Methodology Applied': v}
            for k, v in global_methodology.items()
        ]
        # Persist as a list of dicts to avoid pandas concat comparison errors
        final_df.attrs['methodology_summary'] = summary_data
        
    return final_df

# ==============================================================================
# 3. EXPORT HELPERS
# ==============================================================================
def write_advanced_stats_to_excel(df_results, writer, sheet_name='Advanced_Stats'):
    """
    Saves the results DataFrame to Excel and safely applies formatting if openpyxl is available.
    Also extracts and writes the methodology summary to a separate 'Methodology' sheet.
    """
    if df_results.empty:
        print("Warning: DataFrame is empty. Nothing to write.")
        return
        
    cols_to_drop = ['Precision Test (SD)', 'Accuracy Test (McNemar)', 'MAE Test', 'Dependence Strategy', 'FWER Correction']
    df_clean = df_results.drop(columns=[c for c in cols_to_drop if c in df_results.columns], errors='ignore')
    
    df_clean.to_excel(writer, sheet_name=sheet_name, index=False)
    
    try:
        import openpyxl
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df_clean.columns, 1):
            col_letter = openpyxl.utils.get_column_letter(idx)
            worksheet.column_dimensions[col_letter].width = 15 if ' p' in col else 22
    except (ImportError, AttributeError, KeyError):
        pass 
        
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
            
            try:
                import openpyxl
                meth_ws = writer.sheets[meth_sheet]
                for idx, col in enumerate(meth_df.columns, 1):
                    col_letter = openpyxl.utils.get_column_letter(idx)
                    meth_ws.column_dimensions[col_letter].width = 35 if idx == 1 else 75
            except (ImportError, AttributeError, KeyError):
                pass
