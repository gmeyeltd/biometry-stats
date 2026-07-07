Moussa Research Suite: Advanced Biometry Statistics

A comprehensive Python toolkit for rigorous, heteroscedastic statistical analysis of Intraocular Lens (IOL) power calculation formulas.

This engine evaluates IOL prediction errors by computing appropriate summary metrics (MAE, RMSE, MedAE, MPE, SD, and proportions within ≤0.50D) and generating non-parametric inferences using high-iteration bootstrap resampling. It is mathematically equivalent to the Wilcox-Holladay-Wang-Koch (WHWK) R framework for scale comparisons, but implements advanced vectorized operations, true multithreading, and additional FWER (Family-Wise Error Rate) controls.

Graphical User Interface (GUI) available at https://gmeye.co.uk/#research

Key Features

Robust Resampling Pipeline: Implements both Studentised (Bootstrap-t) and Percentile bootstrapping, handling leptokurtic distributions and heavy tails common in prediction error data without parametric assumptions.

Cluster Bootstrapping & Independence Enforcement: Automatically detects Patient_ID columns and resamples at the patient level rather than the eye level, handling bilateral dependencies (intra-patient correlation). Alternatively, force strict independence by randomly dropping to one eye per patient.

Advanced Variance & Threshold Testing: Includes the Morgan-Pitman test (utilizing HC4 robust regression or Wild Cluster Bootstrapping) for Standard Deviation comparisons, and Yang's Modified Obuchowski Test for cluster-robust McNemar threshold evaluation.

Multiple Comparison Correction: Supports standard p-value adjustment methods (Hommel, Holm, Bonferroni, FDR) alongside advanced Romano-Wolf Step-Down Max-T corrections for tight FWER control.

High-Performance Multithreading: Built on numpy, pandas, and standard library ThreadPoolExecutor to execute pairwise comparisons across all available CPU cores, supporting large resampling iterations (e.g., >4,000) in seconds.

Subgroup Analysis: Allows automatic partitioning of results based on axial length (Short, Medium, Long) cutoffs.

Installation

Ensure you have Python 3.8+ installed. The script relies on standard data science libraries:

pip install numpy pandas scipy statsmodels openpyxl


Basic Usage

The simplest way to use the suite is via the GUI at https://gmeye.co.uk/#research.
Otherwise, you can load your data into a pandas DataFrame containing your spherical equivalent prediction errors (SEQ-PE) and pass it to the analysis function. Each column should represent a different IOL formula.

import pandas as pd
from biometry_stats import analyze_prediction_errors, write_advanced_stats_to_excel

# Load your biometry data (from CSV or Excel)
df = pd.read_excel("my_biometry_data.xlsx")

# Define custom settings (overrides defaults)
settings = {
    'bootstrap_iterations': 2000,
    'fwer_method': 'romano_wolf',
    'axl_short': 22.0,
    'axl_long': 26.0
}

# Optional: Provide a Series of Axial Lengths to automatically generate subgroup analyses
axl_series = df['AXL'] if 'AXL' in df.columns else None

# Run the analysis
results_df = analyze_prediction_errors(df, axl_data=axl_series, settings=settings)

# Save results to Excel using the built-in formatter to generate methodology summaries
with pd.ExcelWriter("analysis_output.xlsx") as writer:
    write_advanced_stats_to_excel(results_df, writer)


Data Formatting Requirements

Prediction Errors: Columns containing numeric prediction error data will be automatically identified as formulas to be compared. Ensure your data has been zeroised (mean error optimised to zero) prior to RMSE analysis for exact precision proxying.

Patient IDs (Optional): If your dataset contains a column named patient_id (or similar variants like id_patient, ptid), the script will automatically activate Cluster Bootstrapping.

Outliers: Data exceeding the validation threshold (default $\pm4.0$D) is filtered via listwise deletion prior to matrix generation.

Configuration & Statistical Methodology

While the suite defaults to Hommel corrections to match the WHWK methodology for MAE and RMSE, setting fwer_method = 'romano_wolf' in your settings dictionary activates a Step-Down Max-T algorithm. This evaluates the joint distribution of the test statistics, providing a powerful, less conservative alternative to standard step-up procedures.

For threshold evaluations (e.g., percentage of eyes within ≤0.50D), this suite utilises the Asymptotic McNemar test with continuity correction. If bilateral patient data is detected, it automatically upgrades to Yang's Modified Obuchowski Test (2010) to control for intra-cluster correlation.

Citation

If you utilise this framework in your research, please refer to the core validation manuscript:

The Moussa Research Suite: validation and demonstration of statistical equivalence to the WHWK Framework for Mean Absolute Error and Root Mean Squared Error (Under Review).
