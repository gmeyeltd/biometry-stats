import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import warnings

# Configuration
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

if BOOTSTRAP_SEED is None:
    BOOTSTRAP_SEED = int(np.random.default_rng().integers(0, 2**32))
    print(f"Bootstrap seed: {BOOTSTRAP_SEED} (random)")
else:
    print(f"Bootstrap seed: {BOOTSTRAP_SEED} (fixed)")

YUEN_TRIM_PROPORTION = 0.2
MCNEMAR_THRESHOLD = 0.50

USE_STUDENTIZED_BS = True 
MULTIPLE_TEST_METHOD = 'hommel' # 'romano_wolf', 'hommel', 'holm', 'bonferroni', 'fdr_bh'
RUN_YUEN_TEST = False 

AXL_SHORT_CUTOFF = 22.0
AXL_LONG_CUTOFF = 26.0
INVALID_THRESHOLD = 4.0


def _calculate_percentile_bootstrap_p(boot_diffs):
    valid_diffs = boot_diffs[~np.isnan(boot_diffs)]
    if len(valid_diffs) == 0: 
        return 1.0
    p = 2 * min(np.mean(valid_diffs > 0), np.mean(valid_diffs < 0))
    return min(p, 1.0)

def _calculate_studentized_bootstrap_p(data_vector, indices):
    n = len(data_vector)
    obs_mean = np.mean(data_vector)
    obs_se = np.std(data_vector, ddof=1) / np.sqrt(n)
    
    if obs_se == 0:
        return 1.0, 0.0, np.zeros(indices.shape[0])
        
    t_obs = obs_mean / obs_se
    shifted_data = data_vector - obs_mean
    
    # Pad for cluster-bootstrap ragged arrays (index N returns NaN)
    shifted_data_pad = np.append(shifted_data, np.nan)
    boot_samples = shifted_data_pad[indices]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b_means = np.nanmean(boot_samples, axis=1)
        b_stds = np.nanstd(boot_samples, axis=1, ddof=1)
        b_n = np.sum(~np.isnan(boot_samples), axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            b_ses = b_stds / np.sqrt(b_n)
            t_stars = b_means / b_ses
        
    t_stars = np.nan_to_num(t_stars, nan=0.0)
    p_value = np.mean(np.abs(t_stars) >= np.abs(t_obs))
    
    return p_value, np.abs(t_obs), np.abs(t_stars)

def _calculate_bca_bootstrap(data1, data2, boot_stats, stat_func, cluster_ids):
    valid_boot = boot_stats[~np.isnan(boot_stats)]
    B = len(valid_boot)
    if B == 0: 
        return np.nan, np.nan
    
    theta_hat = stat_func(data1, data2)
    
    # 1. Acceleration factor (a) using jackknife per-cluster
    unique_clusters = np.unique(cluster_ids)
    jk_stats = np.zeros(len(unique_clusters))
    for i, cid in enumerate(unique_clusters):
        mask = cluster_ids != cid
        jk_stats[i] = stat_func(data1[mask], data2[mask])
            
    mean_jk = np.mean(jk_stats)
    diffs = mean_jk - jk_stats
    sum_sq = np.sum(diffs**2)
    a = np.sum(diffs**3) / (6 * (sum_sq**1.5)) if sum_sq > 0 else 0.0
    
    # 2. Bias correction factor (z0)
    p_theta = (np.sum(valid_boot < theta_hat) + 0.5 * np.sum(valid_boot == theta_hat)) / B
    p_theta = np.clip(p_theta, 1/(2*B), 1 - 1/(2*B))
    z0 = stats.norm.ppf(p_theta)
    
    # 3. BCa CI bounds
    z_025 = stats.norm.ppf(0.025)
    z_975 = stats.norm.ppf(0.975)
    
    den_1 = 1 - a * (z0 + z_025)
    den_2 = 1 - a * (z0 + z_975)
    
    alpha_1_z = z0 + (z0 + z_025) / den_1 if den_1 != 0 else z0 + z0 + z_025
    alpha_2_z = z0 + (z0 + z_975) / den_2 if den_2 != 0 else z0 + z0 + z_975
    
    q1 = np.clip(stats.norm.cdf(alpha_1_z), 0, 1)
    q2 = np.clip(stats.norm.cdf(alpha_2_z), 0, 1)
    
    ci_lower = np.percentile(valid_boot, q1 * 100)
    ci_upper = np.percentile(valid_boot, q2 * 100)
    
    return ci_lower, ci_upper

def perform_advanced_bootstrap(errors_1: np.array, errors_2: np.array, indices: np.array, cluster_ids: np.array) -> dict:
    # Pad arrays with NaN to support varying cluster sample sizes
    e1_pad = np.append(errors_1, np.nan)
    e2_pad = np.append(errors_2, np.nan)
    
    samp1 = e1_pad[indices]
    samp2 = e2_pad[indices]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        n_valid = np.sum(~np.isnan(samp1), axis=1)
        
        boot_mpe_diffs = np.nanmean(samp1, axis=1) - np.nanmean(samp2, axis=1)
        boot_mae_diffs = np.nanmean(np.abs(samp1), axis=1) - np.nanmean(np.abs(samp2), axis=1)
        boot_medae_diffs = np.nanmedian(np.abs(samp1), axis=1) - np.nanmedian(np.abs(samp2), axis=1)
        boot_rmse_diffs = np.sqrt(np.nanmean(samp1**2, axis=1)) - np.sqrt(np.nanmean(samp2**2, axis=1))
        boot_sd_diffs = np.nanstd(samp1, axis=1, ddof=1) - np.nanstd(samp2, axis=1, ddof=1)
        
        mask1 = np.abs(samp1) <= MCNEMAR_THRESHOLD
        mask2 = np.abs(samp2) <= MCNEMAR_THRESHOLD
        boot_mcnemar_diffs = np.nansum(mask1, axis=1) / n_valid - np.nansum(mask2, axis=1) / n_valid

    # Romano-Wolf raw differences for step-down Max-T
    rw_mpe_obs = np.abs(np.mean(errors_1) - np.mean(errors_2))
    rw_mpe_null = np.abs(boot_mpe_diffs - (np.mean(errors_1) - np.mean(errors_2)))
    
    rw_mae_obs = np.abs(np.mean(np.abs(errors_1)) - np.mean(np.abs(errors_2)))
    rw_mae_null = np.abs(boot_mae_diffs - (np.mean(np.abs(errors_1)) - np.mean(np.abs(errors_2))))
    
    rw_rmse_obs = np.abs(np.sqrt(np.mean(errors_1**2)) - np.sqrt(np.mean(errors_2**2)))
    rw_rmse_null = np.abs(boot_rmse_diffs - (np.sqrt(np.mean(errors_1**2)) - np.sqrt(np.mean(errors_2**2))))

    if USE_STUDENTIZED_BS:
        p_mpe, _, _ = _calculate_studentized_bootstrap_p(errors_1 - errors_2, indices)
        p_mae, _, _ = _calculate_studentized_bootstrap_p(np.abs(errors_1) - np.abs(errors_2), indices)
        p_rmse, _, _ = _calculate_studentized_bootstrap_p((errors_1**2) - (errors_2**2), indices)
    else:
        p_mpe = _calculate_percentile_bootstrap_p(boot_mpe_diffs)
        p_mae = _calculate_percentile_bootstrap_p(boot_mae_diffs)
        p_rmse = _calculate_percentile_bootstrap_p(boot_rmse_diffs)

    stat_func_medae = lambda d1, d2: np.median(np.abs(d1)) - np.median(np.abs(d2))
    p_medae = _calculate_percentile_bootstrap_p(boot_medae_diffs)
    rw_medae_obs = np.abs(stat_func_medae(errors_1, errors_2))
    rw_medae_null = np.abs(boot_medae_diffs - stat_func_medae(errors_1, errors_2))
    
    stat_func_sd = lambda d1, d2: np.std(d1, ddof=1) - np.std(d2, ddof=1)
    p_sd = _calculate_percentile_bootstrap_p(boot_sd_diffs)
    rw_sd_obs = np.abs(stat_func_sd(errors_1, errors_2))
    rw_sd_null = np.abs(boot_sd_diffs - stat_func_sd(errors_1, errors_2))
    
    stat_func_mcnemar = lambda d1, d2: np.mean(np.abs(d1) <= MCNEMAR_THRESHOLD) - np.mean(np.abs(d2) <= MCNEMAR_THRESHOLD)
    p_mcnemar = _calculate_percentile_bootstrap_p(boot_mcnemar_diffs)
    rw_mcnemar_obs = np.abs(stat_func_mcnemar(errors_1, errors_2))
    rw_mcnemar_null = np.abs(boot_mcnemar_diffs - stat_func_mcnemar(errors_1, errors_2))

    return {
        'P_MPE': p_mpe,
        'P_MAE': p_mae,
        'P_MedAE': p_medae,
        'P_RMSE': p_rmse,
        'P_SD': p_sd,
        'P_McNemar': p_mcnemar,
        
        'RW_MPE_obs': rw_mpe_obs, 'RW_MPE_null': rw_mpe_null,
        'RW_MAE_obs': rw_mae_obs, 'RW_MAE_null': rw_mae_null,
        'RW_RMSE_obs': rw_rmse_obs, 'RW_RMSE_null': rw_rmse_null,
        'RW_MedAE_obs': rw_medae_obs, 'RW_MedAE_null': rw_medae_null,
        'RW_SD_obs': rw_sd_obs, 'RW_SD_null': rw_sd_null,
        'RW_McNemar_obs': rw_mcnemar_obs, 'RW_McNemar_null': rw_mcnemar_null,
    }

def yuen_t_test(x, y, tr=0):
    x = np.array(x)
    y = np.array(y)
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
    
    if g > 0:
        trimmed_mean_diff = np.mean(d_sort[g:-g])
    else:
        trimmed_mean_diff = np.mean(d)
        
    t_stat = trimmed_mean_diff / se_tr
    df = h - 1
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    return p_val, trimmed_mean_diff

def romano_wolf_correction(obs_stats, boot_null_stats):
    M = len(obs_stats)
    if M == 0: 
        return np.array([])
    
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

def run_biometry_analysis(df_input, group_name="All"):
    numeric_df = df_input.select_dtypes(include=[np.number])
    formulas = []
    
    patient_col = None
    for c in df_input.columns:
        c_lower = str(c).lower()
        if c_lower in ['patient_id', 'id_patient', 'patientid']:
            patient_col = c
            continue
        if c_lower == 'id' or c_lower.startswith('id_'): 
            continue
        if c in numeric_df.columns:
            formulas.append(c)
        
    if len(formulas) < 2:
        return pd.DataFrame(), {}
        
    # Listwise Deletion
    valid_mask = pd.Series(True, index=numeric_df.index)
    for f in formulas:
        valid_mask &= ~np.isnan(numeric_df[f])
        valid_mask &= np.abs(numeric_df[f]) <= INVALID_THRESHOLD
        
    if patient_col:
        valid_mask &= df_input[patient_col].notna()
        print(f"[{group_name}] Resampling by cluster: '{patient_col}'")
    else:
        print(f"[{group_name}] No patient ID found. Using standard independent resampling.")
        
    valid_df = numeric_df[valid_mask]
    n_global = len(valid_df)
    
    if n_global < 2:
        print(f"[{group_name}] Insufficient data after filtering. Skipping.")
        return pd.DataFrame(), {}
        
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    
    if patient_col:
        cluster_ids = df_input.loc[valid_mask, patient_col].values
    else:
        cluster_ids = np.arange(n_global)
        
    unique_clusters = np.unique(cluster_ids)
    K = len(unique_clusters)
    cluster_to_idx = {cid: np.where(cluster_ids == cid)[0] for cid in unique_clusters}
    
    boot_indices_list = []
    max_len = 0
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_clusters = rng.choice(unique_clusters, size=K, replace=True)
        idx = np.concatenate([cluster_to_idx[cid] for cid in sampled_clusters])
        boot_indices_list.append(idx)
        max_len = max(max_len, len(idx))
        
    # Pad shorter samples so they map to NaN
    global_indices = np.full((BOOTSTRAP_ITERATIONS, max_len), n_global, dtype=int)
    for i in range(BOOTSTRAP_ITERATIONS):
        idx = boot_indices_list[i]
        global_indices[i, :len(idx)] = idx
    
    pairs = list(combinations(formulas, 2))
    results = []
    
    rw_stats = {
        'MPE': {'obs': [], 'null': []},
        'MAE': {'obs': [], 'null': []},
        'RMSE': {'obs': [], 'null': []},
        'MedAE': {'obs': [], 'null': []},
        'SD': {'obs': [], 'null': []},
        'McNemar': {'obs': [], 'null': []}
    }
    
    bs_type = "Studentized" if USE_STUDENTIZED_BS else "Percentile"
    print(f"[{group_name}] Processing {len(pairs)} comparisons ({bs_type}, N={n_global})...")
    
    for f1, f2 in pairs:
        e1 = valid_df[f1].values
        e2 = valid_df[f2].values
        
        boot_res = perform_advanced_bootstrap(e1, e2, global_indices, cluster_ids)
        
        rw_stats['MPE']['obs'].append(boot_res['RW_MPE_obs'])
        rw_stats['MPE']['null'].append(boot_res['RW_MPE_null'])
        rw_stats['MAE']['obs'].append(boot_res['RW_MAE_obs'])
        rw_stats['MAE']['null'].append(boot_res['RW_MAE_null'])
        rw_stats['RMSE']['obs'].append(boot_res['RW_RMSE_obs'])
        rw_stats['RMSE']['null'].append(boot_res['RW_RMSE_null'])
        rw_stats['MedAE']['obs'].append(boot_res['RW_MedAE_obs'])
        rw_stats['MedAE']['null'].append(boot_res['RW_MedAE_null'])
        rw_stats['SD']['obs'].append(boot_res['RW_SD_obs'])
        rw_stats['SD']['null'].append(boot_res['RW_SD_null'])
        rw_stats['McNemar']['obs'].append(boot_res['RW_McNemar_obs'])
        rw_stats['McNemar']['null'].append(boot_res['RW_McNemar_null'])
        
        if RUN_YUEN_TEST:
            _, diff_mae = yuen_t_test(np.abs(e1), np.abs(e2), tr=0)
            yuen_mae_p, trim_mae_diff = yuen_t_test(np.abs(e1), np.abs(e2), tr=YUEN_TRIM_PROPORTION)
            yuen_rmse_p, trim_sq_diff = yuen_t_test(e1**2, e2**2, tr=YUEN_TRIM_PROPORTION)
        else:
            diff_mae = np.mean(np.abs(e1)) - np.mean(np.abs(e2))
            
        diff_mpe = np.mean(e1) - np.mean(e2)
        diff_medae = np.median(np.abs(e1)) - np.median(np.abs(e2))
        diff_rmse = np.sqrt(np.mean(e1**2)) - np.sqrt(np.mean(e2**2))
        diff_sd = np.std(e1, ddof=1) - np.std(e2, ddof=1)
        
        prop1 = np.mean(np.abs(e1) <= MCNEMAR_THRESHOLD)
        prop2 = np.mean(np.abs(e2) <= MCNEMAR_THRESHOLD)
        diff_mcnemar = prop1 - prop2
        
        row = {
            'Group': group_name, 'Formula A': f1, 'Formula B': f2, 'N': n_global,
            'MPE Diff': diff_mpe, 'Boot (MPE) p': boot_res['P_MPE'],
            'MAE Diff': diff_mae, 'Boot (MAE) p': boot_res['P_MAE'],
            'MedAE Diff': diff_medae, 'Boot (MedAE) p': boot_res['P_MedAE'],
            'RMSE Diff': diff_rmse, 'Boot (RMSE) p': boot_res['P_RMSE'],
            'SD Diff': diff_sd, 'Boot (SD) p': boot_res['P_SD'],
            'McNemar Diff': diff_mcnemar, 'Boot (McNemar) p': boot_res['P_McNemar']
        }
        
        if RUN_YUEN_TEST:
            row['Trimmed MAE Diff'] = trim_mae_diff
            row['Yuen (MAE) p'] = yuen_mae_p
            row['Yuen (RMSE) p'] = yuen_rmse_p
            
        results.append(row)
        
    return pd.DataFrame(results), rw_stats

def adjust_pvalues(df_results, rw_stats=None, method='hommel'):
    if df_results.empty: 
        return df_results
        
    rw_metric_mapping = {
        'Boot (MPE) p': 'MPE',
        'Boot (MAE) p': 'MAE',
        'Boot (MedAE) p': 'MedAE',
        'Boot (RMSE) p': 'RMSE',
        'Boot (SD) p': 'SD',
        'Boot (McNemar) p': 'McNemar'
    }
        
    p_cols = [c for c in df_results.columns if ' p' in c and 'adj' not in c]
    
    for col in p_cols:
        if method == 'romano_wolf' and col in rw_metric_mapping and rw_stats is not None:
            metric = rw_metric_mapping[col]
            obs = np.array(rw_stats[metric]['obs'])
            null_matrix = np.column_stack(rw_stats[metric]['null'])
            pvals_adj = romano_wolf_correction(obs, null_matrix)
        else:
            fallback_method = 'hommel' if method == 'romano_wolf' else method
            pvals = df_results[col].values
            pvals_adj = np.full(len(pvals), np.nan)
            
            mask = ~np.isnan(pvals)
            if np.sum(mask) > 0:
                reject, adj, _, _ = multipletests(pvals[mask], method=fallback_method)
                pvals_adj[mask] = adj
                
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

def analyze_prediction_errors(df_input, axl_data=None):
    all_results_frames = []
    
    def _process_group(df_sub, name):
        if df_sub.empty: 
            return
        res_df, rw_stats = run_biometry_analysis(df_sub, group_name=name)
        if not res_df.empty:
            res_df = adjust_pvalues(res_df, rw_stats=rw_stats, method=MULTIPLE_TEST_METHOD)
            all_results_frames.append(res_df)

    _process_group(df_input, "All")
        
    if axl_data is not None:
        if isinstance(df_input, pd.DataFrame) and hasattr(axl_data, 'index'):
            common_idx = df_input.index.intersection(axl_data.index)
            df_aligned = df_input.loc[common_idx]
            if isinstance(axl_data, pd.DataFrame):
                axl_aligned = axl_data.loc[common_idx].iloc[:, 0]
            else:
                axl_aligned = axl_data.loc[common_idx]
        else:
            df_aligned = df_input
            axl_aligned = axl_data
            
        axl_vals = np.array(axl_aligned)
        
        mask_short = axl_vals <= AXL_SHORT_CUTOFF
        mask_long = axl_vals >= AXL_LONG_CUTOFF
        mask_medium = (axl_vals > AXL_SHORT_CUTOFF) & (axl_vals < AXL_LONG_CUTOFF)
        
        if np.sum(mask_short) > 0: 
            if BOOTSTRAP_SEED is not None: np.random.seed(BOOTSTRAP_SEED)
            _process_group(df_aligned.iloc[mask_short], "Short")
            
        if np.sum(mask_medium) > 0: 
            if BOOTSTRAP_SEED is not None: np.random.seed(BOOTSTRAP_SEED)
            _process_group(df_aligned.iloc[mask_medium], "Medium")
            
        if np.sum(mask_long) > 0: 
            if BOOTSTRAP_SEED is not None: np.random.seed(BOOTSTRAP_SEED)
            _process_group(df_aligned.iloc[mask_long], "Long")
                
    if not all_results_frames:
        return pd.DataFrame()
        
    final_df = pd.concat(all_results_frames, ignore_index=True)
    return final_df

def process_excel_stats(file_path):
    try:
        df = pd.read_excel(file_path)
        results_df = analyze_prediction_errors(df, axl_data=None)
        return results_df
    except Exception as e:
        return f"Error: {str(e)}"