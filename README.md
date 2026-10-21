# biometry-stats

Python code for comparing the accuracy of intraocular lens (IOL) power formulas. It was written for
formula comparison studies and handles datasets that include one or both eyes of each patient.

If you would rather not use Python, the same tests are available through a web interface:
[gmeye.co.uk/#research](https://gmeye.co.uk/#research).

## What it does

For every pair of formulas, `biometry_stats.py` compares the prediction errors (PE) on:

- mean PE (MPE)
- mean absolute error (MAE)
- root-mean-square error (RMSE)
- median absolute error (MedAE)
- standard deviation (SD)
- the proportion of eyes within ±0.50 D

The tests depend on whether any patient contributes both eyes:

| Metric | One eye per patient | Both eyes (patient as the unit) |
|---|---|---|
| MPE, MAE, RMSE | bootstrap-t | patient bootstrap-t with a CR1 standard error |
| MedAE | percentile bootstrap | patient percentile bootstrap |
| SD | Morgan–Pitman test, HC4 standard errors | Morgan–Pitman test, wild cluster bootstrap |
| Within ±0.50 D | McNemar test (continuity corrected) | Yang's modified Obuchowski test |

Each difference between two formulas can also be reported with a bootstrap confidence interval (bootstrap-t for MPE
and MAE, percentile for RMSE, MedAE, SD and the proportion within the threshold), using the same resamples as the tests.
These intervals are not adjusted for multiple comparisons, so an interval that excludes zero can sit next to a
non-significant adjusted p-value. They are therefore off by default; switch them on with `"report_ci": True` and set the
level with `"ci_level"`.

RMSE is tested through the mean squared error. When patients contribute both eyes, all resampling is
done by patient, so fellow eyes stay together. Bootstrap p-values are calculated as (1 + b)/(B + 1).

Multiple comparisons are handled with the Romano–Wolf step-down procedure by default, or with Hommel,
Holm, Bonferroni or Benjamini–Hochberg (any method accepted by `statsmodels`' `multipletests`).
Results can also be split into short, medium and long eyes.

## Installation

Python 3.8 or later. Download `biometry_stats.py` into your working folder and install:

```bash
pip install numpy pandas scipy statsmodels openpyxl
```

## Usage

```python
import pandas as pd
from biometry_stats import analyze_prediction_errors, write_advanced_stats_to_excel

df = pd.read_excel("my_biometry_data.xlsx")

settings = {
    "bootstrap_iterations": 4999,
    "fwer_method": "romano_wolf",
    "axl_short": 22.0,
    "axl_long": 26.0,
    "report_ci": False,         # unadjusted confidence interval for every difference (off by default)
    "ci_level": 0.95,
}

# optional: axial length for the short / medium / long subgroups
axl = df["AXL"] if "AXL" in df.columns else None

results = analyze_prediction_errors(df, axl_data=axl, settings=settings)

with pd.ExcelWriter("analysis_output.xlsx") as writer:
    write_advanced_stats_to_excel(results, writer)
```

The Excel file has one sheet of results (differences, p-values and adjusted p-values for every pair
of formulas) and a second sheet listing the tests that were used.

## Preparing your data

- One row per eye and one column per formula, holding the prediction error in dioptres. Every numeric
  column is treated as a formula unless it is a patient ID, a recognised biometry column (AXL, AL,
  ACD, K1, K2, age, LT, CCT, WTW and similar), a column starting `predict_` or a column ending
  `_const`.
- Optimise the lens constants first, so that each formula's mean PE is zero.
- To analyse both eyes, include a patient ID column named `patient_id` (or `id`, `pt_id`, `ptid`,
  `patientid`, `id_patient`, `hosp_id` or `hospid`). Eyes sharing an ID are resampled together.
  Without one, every row is treated as a separate patient.
- Eyes with a missing PE, or a PE beyond ±4.00 D for any formula, are excluded from all comparisons.

## Settings

Passed in the `settings` dictionary:

| Key | Default | |
|---|---|---|
| `bootstrap_iterations` | 2000 | number of bootstrap resamples |
| `fwer_method` | `"romano_wolf"` | or `"hommel"`, `"holm"`, `"bonferroni"`, `"fdr_bh"` |
| `invalid_threshold` | 4.0 | eyes with a larger \|PE\| are excluded (D) |
| `mcnemar_threshold` | 0.50 | threshold for the proportion test (D) |
| `axl_short`, `axl_long` | 22.0, 26.0 | subgroup cut-offs (mm) |

Set at the top of the file:

| Name | Default | |
|---|---|---|
| `BOOTSTRAP_SEED` | 42 | `None` gives a different random seed each run |
| `USE_STUDENTIZED_BS` | `True` | `False` uses percentile bootstraps throughout |
| `ENFORCE_INDEPENDENCE` | `False` | `True` keeps one random eye per patient |
| `RUN_SD_TEST` | `True` | Morgan–Pitman test of SDs |
| `RUN_YUEN_TEST` | `False` | adds Yuen's trimmed-mean test |

With Romano–Wolf correction, the SD test uses the wild bootstrap even when each patient has one eye,
because the correction needs its bootstrap distribution.

## Simulation

`prediction_error_simulation.py` is the Monte Carlo simulation used to check these tests: whether
they keep the false-positive rate at 5% when both eyes are analysed, and how much power they gain
over a one-eye analysis. It simulates 10 000 studies in which two formulas are equally accurate and
5000 in which one is less accurate, analyses each study with both eyes and with one eye per patient,
and writes every p-value to `trials.csv`. The parameters are at the top of the file.

```bash
python prediction_error_simulation.py --quick   # 150 studies, a few minutes
python prediction_error_simulation.py           # full run; several hours, all but two CPU cores
```

The simulation keeps its own copies of the tests, as they were run for the paper. These calculate
bootstrap p-values as b/B rather than (1 + b)/(B + 1); with 3999 resamples the difference is negligible.

## Validation against the WHWK functions

The `validation/` folder holds the scripts used to check that this code reproduces the WHWK R functions
(Rallfun-v45, https://osf.io/xhe8u) exactly, by giving both programs the same bootstrap resamples. No data
are included. To repeat the check on your own data, put `Rallfun-v45.txt` in the repository root and an Excel
workbook with one sheet per dataset (columns HQ, H1, Haigis, SRK/T, BU2, K6) at `validation/data/datasets.xlsx`,
edit the sheet names at the top of `00_prepare_data.py`, and run the steps in order:

```bash
python validation/00_prepare_data.py          # one CSV per dataset
Rscript validation/01_whwk_reference.R        # WHWK functions in R; saves the resamples they use
python validation/02_moussa_and_compare.py    # this code on the same resamples; compares every result
```

`03_benchmark.R` / `03_benchmark.py` time both programs, `04_deviations.py` repeats the comparison with each
program's own resamples and with the improved settings, and `07_seed_variability.py` repeats the bootstrap
comparisons over 100 seeds. On Windows, `run_all.bat` runs steps 0 to 2.

## Citation

If you use this code, please cite:

Moussa G, Cannon N, Cooke DL. Keeping both eyes in intraocular lens formula comparisons: development
and validation of a cluster-aware statistical suite. Submitted.

For the check against the WHWK functions:

Moussa G, Cannon N, Cooke DL. The Moussa Research Suite: a free web tool that exactly reproduces the
Wilcox–Holladay–Wang–Koch statistics for intraocular lens formula comparison. Submitted.

The one-eye tests follow the recommendations of:

Holladay JT, Wilcox RR, Koch DD, Wang L. Review and recommendations for univariate statistical analysis
of spherical equivalent prediction error for IOL power calculations. J Cataract Refract Surg.
2021;47(1):65–77.

## Acknowledgements

Thanks to Professor Rand Wilcox: `olshc4` and `comdvar` were adapted into Python, with his permission,
from his R functions (Rallfun v45, [osf.io/xhe8u](https://osf.io/xhe8u)).

Other methods: Cameron, Gelbach and Miller (2008) for the wild cluster bootstrap; Yang, Sun and
Hardin (2010) for the clustered McNemar test; Romano and Wolf (2005) for the step-down correction.

## Licence

MIT. See `LICENSE`.
