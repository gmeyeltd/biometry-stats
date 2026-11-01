# biometry-stats

Python code for comparing the accuracy of intraocular lens (IOL) power formulas. It was written for formula
comparison studies and handles datasets that include one or both eyes of each patient, with the patient as the unit
of analysis.

If you would rather not use Python, the same tests are available through a web interface:
[gmeye.co.uk/#research](https://gmeye.co.uk/#research).

## What it does

For every pair of formulas it compares the prediction errors (PE) on:

- mean PE (MPE)
- mean absolute error (MAE)
- root-mean-square error (RMSE)
- median absolute error (MedAE)
- standard deviation (SD)
- the proportion of eyes within ±0.50 D

It also gives each formula's MPE, MAE, MedAE, RMSE, SD and percentage within ±0.50 D with a 95% confidence
interval (CI), and repeats everything for short, medium and long eyes and for any other subgroups you choose.

The tests depend on whether any patient contributes both eyes:

| Metric | One eye per patient | Both eyes (patient as the unit) |
|---|---|---|
| MPE, MAE, RMSE | bootstrap-t | patient bootstrap-t with a CR1 standard error |
| MedAE | percentile bootstrap | patient percentile bootstrap |
| SD | Morgan–Pitman test, HC4 standard errors | Morgan–Pitman test, wild cluster bootstrap |
| Within ±0.50 D | McNemar test (continuity corrected) | Yang's modified Obuchowski test |

RMSE is tested through the mean squared error. When patients contribute both eyes, all resampling is done by
patient, so fellow eyes stay together. Bootstrap p-values are calculated as (1 + b)/(B + 1). The CIs for differences
in MPE and MAE are symmetric bootstrap-t intervals from the same resamples as the p-values, so an interval excludes 0
when p < 0.05. Each formula's own CI is a percentile bootstrap of patients.

## Multiple comparisons

P-values are adjusted for the number of formula pairs, with the Romano–Wolf step-down procedure by default (Hommel
for the proportion within ±0.50 D), or with Hommel, Holm, Bonferroni or Benjamini–Hochberg throughout.

Each difference between two formulas has an unadjusted 95% CI. An adjusted CI is given only where it comes from the
same adjustment as the p-value:

| `fwer_method` | Adjusted CI |
|---|---|
| `romano_wolf` (default) | simultaneous intervals for MPE, MAE and MedAE (single-step max-\|t\|, the interval counterpart of Romano–Wolf) |
| `bonferroni` | Bonferroni intervals for every metric |
| `fdr_bh` | Benjamini–Yekutieli false-coverage-rate intervals for every metric |
| `holm`, `hommel` | none: these stepwise methods have no corresponding intervals |

Romano–Wolf intervals are single-step, so one that excludes 0 always has an adjusted p-value below 0.05, although a
step-down rejection can occasionally have an interval that includes 0. The Statistical Tests sheet names the method
used for each metric. With Romano–Wolf, the SD test uses the wild
bootstrap even when each patient has one eye, because the correction needs its bootstrap distribution.

## Installation

Python 3.8 or later. Download `biometry_stats_advanced.py` and `ophthalmology_stats.py` into the same folder and
install:

```bash
pip install numpy pandas scipy statsmodels openpyxl
```

`ophthalmology_stats.py` holds the statistical tests. It is the same file as in
[ophthalmology-stats](https://github.com/gmeyeltd/ophthalmology-stats), which applies these tests to any ophthalmic
prediction model. `biometry_stats_advanced.py` adds the IOL settings.

## Usage

From the command line:

```bash
python biometry_stats_advanced.py my_data.xlsx --out results.xlsx
python biometry_stats_advanced.py my_data.xlsx --subgroup eye --subgroup K1:bins=3 --out results.xlsx
python biometry_stats_advanced.py my_data.xlsx --cross AXL K1:bins=3 --out results.xlsx
```

`python biometry_stats_advanced.py --help` lists every option.

In Python:

```python
import pandas as pd
import biometry_stats_advanced as bsa

df = pd.read_excel("my_data.xlsx")
results = bsa.analyze_prediction_errors(df, subgroup=["eye", "K1:bins=3"])
bsa.write_results_to_excel(results, "results.xlsx")
```

## Preparing your data

- One row per eye and one column per formula, holding the prediction error in dioptres. Every numeric column is
  treated as a formula unless it is a patient ID, a recognised biometry column (AXL, AL, ACD, K1, K2, age, LT, CCT,
  WTW and similar), a subgroup column, a column starting `predict_` or a column ending `_const`. To choose the
  formulas yourself, use `--formulas`.
- Optimise the lens constants first, so that each formula's mean PE is zero.
- To analyse both eyes, include a patient ID column named `patient_id` (or `id`, `pt_id`, `ptid`, `patientid`,
  `id_patient`, `hosp_id` or `hospid`). Eyes sharing an ID are resampled together. Without one, every row is treated
  as a separate patient.
- Eyes with a missing PE, or a PE beyond ±4.00 D for any formula, are excluded from all comparisons
  (`--exclude-above` to change the limit).

## Subgroups

When the data have an axial length column (AXL, AL or axial_length), results are also given for short (AL ≤22 mm),
medium (AL >22 to <26 mm) and long eyes (AL ≥26 mm). `--no-al-groups` leaves them out.

`--subgroup` adds any other column, and can be repeated:

| `--subgroup` | Groups |
|---|---|
| `eye` | one per value (R, L) |
| `K1:bins=3` | three equal-count groups (tertiles) |
| `AXL:cuts=22,24.5,26` | ≤22, >22 to <24.5, 24.5 to <26, ≥26: the first group includes the first cut point and the last group the last |

`--cross ROWS COLUMNS` crosses two subgroups (for example `--cross AXL K1:bins=3`) and adds a Grid sheet for heat
maps. Cells with fewer than 20 patients get point estimates only. Each subgroup is adjusted for its own formula
pairs, so subgroup results are exploratory.

## Output

An Excel workbook. Results has one row per pair of formulas, with the differences, CIs, p-values and adjusted
p-values, and Statistical Tests lists the tests and adjustments used. Formula Performance gives each formula's MPE,
MAE, MedAE, RMSE, SD and percentage within ±0.50 D with 95% CIs. Estimates has one row per pair of formulas and
metric, with the method of each test and adjustment, ρΔ (the correlation between fellow eyes of the per-eye
difference between the two formulas) and the one-eye factor (how many times as many patients a study keeping one
eye per patient would need for the same precision). Sample gives the numbers of eyes, patients and patients with
both eyes in each group, and Grid, with `--cross` only, the values for heat maps.

## Settings

Passed as `settings={...}` to `analyze_prediction_errors`; the command-line equivalents are in `--help`.

| Key | Default | |
|---|---|---|
| `bootstrap_iterations` | 4999 | number of bootstrap resamples |
| `seed` | 42 | random seed |
| `fwer_method` | `"romano_wolf"` | or `"hommel"`, `"holm"`, `"bonferroni"`, `"fdr_bh"`, `"none"` |
| `invalid_threshold` | 4.0 | eyes with a larger \|PE\| are excluded (D) |
| `mcnemar_threshold` | 0.50 | threshold for the proportion within (D) |
| `axl_short`, `axl_long` | 22.0, 26.0 | short eyes ≤ `axl_short`, long eyes ≥ `axl_long` (mm) |
| `ci_level` | 0.95 | level of the CIs |
| `one_eye` | `False` | keep one randomly chosen eye per patient |

## Simulation

`prediction_error_simulation.py` is the Monte Carlo simulation used to check these tests: whether they keep the
false-positive rate at 5% when both eyes are analysed, and how much power they gain over a one-eye analysis. It
simulates 10 000 studies in which two formulas are equally accurate and 5000 in which one is less accurate, analyses
each study with both eyes and with one eye per patient, and writes every p-value to `trials.csv`. The parameters
are at the top of the file.

```bash
python prediction_error_simulation.py --quick   # 150 studies, a few minutes
python prediction_error_simulation.py           # full run; several hours, all but two CPU cores
```

The simulation keeps its own copies of the tests, as they were run for the paper. These calculate bootstrap p-values
as b/B rather than (1 + b)/(B + 1); with 3999 resamples the difference is negligible.

## Citation

If you use this code, please cite:

Moussa G, Cannon N, Cooke DL. Keeping both eyes in intraocular lens formula comparisons: development and validation
of a cluster-aware statistical suite. Submitted.

For the check against the Wilcox–Holladay–Wang–Koch (WHWK) R functions:

Moussa G, Cannon N, Cooke DL. The Moussa Research Suite: a free web tool that exactly reproduces the
Wilcox–Holladay–Wang–Koch statistics for intraocular lens formula comparison. Submitted.

The one-eye tests follow the recommendations of:

Holladay JT, Wilcox RR, Koch DD, Wang L. Review and recommendations for univariate statistical analysis of
spherical equivalent prediction error for IOL power calculations. J Cataract Refract Surg. 2021;47(1):65–77.

## Acknowledgements

Thanks to Professor Rand Wilcox: `olshc4` and `comdvar` were adapted into Python, with his permission, from his R
functions (Rallfun v45, [osf.io/xhe8u](https://osf.io/xhe8u)).

Other methods: Cameron, Gelbach and Miller (2008) for the wild cluster bootstrap; Yang, Sun and Hardin (2010) for the
clustered McNemar test; Romano and Wolf (2005) for the step-down correction.

## Licence

MIT. See `LICENSE`.
