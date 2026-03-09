# Moussa Research Suite: Advanced Biometry Statistics

A comprehensive Python toolkit for rigorous, heteroscedastic statistical analysis of Intraocular Lens (IOL) power calculation formulas.

This engine evaluates IOL prediction errors by computing appropriate summary metrics (MAE, RMSE, MedAE, MPE, SD, and proportions within ≤0.50D) and generating non-parametric inferences using high-iteration bootstrap resampling. It is mathematically equivalent to the Wilcox-Holladay-Wang-Koch (WHWK) R framework for scale comparisons, but implements advanced vectorised operations and additional FWER (Family-Wise Error Rate) controls.

Graphical User Inteferace (GUI) available at https://gmeye.co.uk/#research
## Key Features

* **Robust Resampling Pipeline:** Implements both Studentised (Bootstrap-t) and Percentile bootstrapping, handling leptokurtic distributions and heavy tails common in prediction error data without parametric assumptions.

* **Multiple Comparison Correction:** Supports standard p-value adjustment methods (Hommel, Holm, Bonferroni, FDR) alongside advanced **Romano-Wolf Step-Down Max-T** corrections for tight FWER control.

* **Cluster Bootstrapping:** Automatically detects `Patient_ID` columns and resamples at the patient level rather than the eye level, elegantly handling bilateral dependencies (intra-patient correlation).

* **Subgroup Analysis:** Allows automatic partitioning of results based on axial length (Short, Medium, Long) cutoffs.

* **Fully Vectorised:** Built on `numpy` and `pandas` for rapid execution, easily supporting $\ge$ 4,000 bootstrap iterations in practical timeframes.

## Installation

Ensure you have Python 3.8+ installed. The script relies on standard data science libraries:

```bash
pip install numpy pandas scipy statsmodels
```
## Basic Usage
The simplest way to use the suite is via the GUI at https://gmeye.co.uk/#research
Otherwise you can load your data into a `pandas` DataFrame containing your spherical equivalent prediction errors (SEQ-PE) and pass it to the analysis function. Each column should represent a different IOL formula.
```bash
import pandas as pd
from biometry_stats import analyze_prediction_errors

# Load your biometry data (from CSV or Excel)
df = pd.read_csv("my_biometry_data.csv")

# Optional: Provide a Series of Axial Lengths to automatically generate subgroup analyses
axl_series = df['AXL'] 
results_df = analyze_prediction_errors(df, axl_data=axl_series)

# Save your results
results_df.to_csv("analysis_output.csv", index=False)
```

## Data Formatting Requirements

* **Prediction Errors:** Columns containing numeric prediction error data will be automatically identified as formulas to be compared. Ensure your data has been zeroised (mean error optimised to zero) prior to RMSE analysis for exact precision proxying.

* **Patient IDs (Optional):** If your dataset contains a column named patient_id (case-insensitive), the script will automatically activate Cluster Bootstrapping.

* **Outliers:** Data exceeding INVALID_THRESHOLD (default $\pm4.0$D) is filtered via listwise deletion prior to matrix generation.

## Configuration Options

You can adjust the behaviour of the statistical engine by modifying the configuration variables at the top of `biometry_stats.py`:
```bash
BOOTSTRAP_ITERATIONS = 4000      # Default iterations for Monte Carlo sampling
BOOTSTRAP_SEED = 42              # Fix seed for reproducible results

USE_STUDENTIZED_BS = True        # Uses Bootstrap-t for MAE/RMSE/MPE. False falls back to percentile.
MULTIPLE_TEST_METHOD = 'hommel'  # 'hommel', 'romano_wolf', 'holm', 'bonferroni', 'fdr_bh'

# Subgroup definition (mm)
AXL_SHORT_CUTOFF = 22.0
AXL_LONG_CUTOFF = 26.0
```
**Note on Romano-Wolf and McNemar deviations**
While the suite defaults to Hommel corrections to match the WHWK methodology perfectly for MAE and RMSE, setting `MULTIPLE_TEST_METHOD = 'romano_wolf'` activates a Step-Down Max-T algorithm. 
This evaluates the joint distribution of the test statistics, providing a powerful, less conservative alternative to standard step-up procedures.

Additionally, for threshold evaluations (e.g., percentage of eyes within ≤0.50D), this suite substitutes the traditional Exact McNemar test with a fully nonparametric bootstrapped difference in proportions. This integrates fully with the vectorised matrix and provides superior statistical power by leveraging the entire dataset rather than discarding concordant pairs.

Citation
If you utilise this framework in your research, please refer to the core validation manuscript:

The Moussa Research Suite: validation and demonstration of statistical equivalence to the WHWK Framework for Mean Absolute Error and Root Mean Squared Error (Under Review).

