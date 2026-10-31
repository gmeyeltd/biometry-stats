"""
biometry-stats: compare intraocular lens (IOL) power formulas when some patients contribute both eyes, with the
patient as the unit of analysis.

The tests are those of ophthalmology_stats.py (the engine, kept in the same folder). This file adds the IOL
settings: prediction errors (PE) in dioptres, eyes with |PE| above 4 D excluded, the proportion of eyes within
±0.50 D, and short, medium and long eyes (axial length ≤22, >22 to <26 and ≥26 mm).

  python biometry_stats_advanced.py my_data.xlsx --out results.xlsx

or, in Python:

  import pandas as pd
  import biometry_stats_advanced as bsa
  res = bsa.analyze_prediction_errors(pd.read_excel("my_data.xlsx"))
  bsa.write_results_to_excel(res, "results.xlsx")
"""
import re

import numpy as np
import pandas as pd

import ophthalmology_stats as osx

__version__ = "1.0"

PATIENT_ID_VARIANTS = ['patient_id', 'id_patient', 'patientid', 'pt_id', 'id', 'hosp_id', 'hospid', 'ptid']
IGNORE_VARIANTS = ['axl', 'al', 'acd', 'acdepth', 'k1', 'k2', 'meank', 'age', 'gender', 'gender_num', 'eye', 'side',
                   'lt', 'lensthickness', 'cct', 'wtw', 'hcd', 'iol_implanted', 'postoperative_refraction',
                   'corneal_refractive_index']
AL_VARIANTS = ['axl', 'al', 'axial_length', 'axiallength']

DEFAULT_SETTINGS = {
    "bootstrap_iterations": 4999,
    "seed": 42,
    "fwer_method": "romano_wolf",     # or 'hommel', 'holm', 'bonferroni', 'fdr_bh', 'none'
    "invalid_threshold": 4.0,         # eyes with |PE| above this for any formula are excluded (D)
    "mcnemar_threshold": 0.50,        # proportion of eyes with |PE| <= this (D)
    "axl_short": 22.0,                # short eyes: AL <= axl_short (mm)
    "axl_long": 26.0,                 # long eyes: AL >= axl_long (mm)
    "report_ci": True,                # 95% CIs for every formula and every difference
    "ci_level": 0.95,
    "one_eye": False,                 # keep one randomly chosen eye per patient
    "min_group_size": 5,              # groups with fewer valid eyes are not tested
    "cross_min_patients": 20,         # crossed subgroups: cells with fewer patients get point estimates only
    "verbose": True,
}

METRICS = [("bias", "MPE"), ("mae", "MAE"), ("medae", "MedAE"), ("rmse", "RMSE"), ("sd", "SD"), ("within", "Within")]
_BLOCK = [(r"^MPE |^Boot \(MPE\)", 0), (r"^MAE |^Boot \(MAE\)", 1), (r"^MedAE |^Boot \(MedAE\)", 2),
          (r"^RMSE |^Boot \(RMSE\)", 3), (r"^SD |^Morgan-Pitman", 4), (r"^McNemar", 5)]


def _settings(settings):
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    return s


def detect_columns(df, extra_ignore=()):
    """Formula columns, the patient ID column and the axial length column, by name. Every numeric column is a
    formula unless it is a patient ID, a biometry column (AXL, K1, ACD, ...), a column starting 'predict_' or
    ending '_const', or a subgroup column."""
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    patient, al, formulas = None, None, []
    skip = {str(c).lower() for c in extra_ignore}
    for c in df.columns:
        low = str(c).lower()
        if patient is None and low in PATIENT_ID_VARIANTS:
            patient = c
            continue
        if al is None and low in AL_VARIANTS and c in numeric:
            al = c
        if (low in IGNORE_VARIANTS or low in AL_VARIANTS or low.startswith('predict_') or low.endswith('_const')
                or low in skip):
            continue
        if c in numeric:
            formulas.append(c)
    return formulas, patient, al


def _cut_labels(column, cuts, unit=""):
    """Labels for cut points c1 < ... < cm: the first group includes c1 (≤c1), the last includes cm (≥cm), and in
    between each group includes its lower cut point. One cut point gives ≤c1 and >c1."""
    n = osx._num
    c = sorted({float(x) for x in np.atleast_1d(cuts)})
    if len(c) == 1:
        return c, [f"{column} ≤{n(c[0])}{unit}", f"{column} >{n(c[0])}{unit}"]
    labels = [f"{column} ≤{n(c[0])}{unit}"]
    for i in range(len(c) - 1):
        lo = f">{n(c[0])}" if i == 0 else n(c[i])
        labels.append(f"{column} {lo} to <{n(c[i + 1])}{unit}")
    labels.append(f"{column} ≥{n(c[-1])}{unit}")
    return c, labels


def _cut_groups(x, cuts):
    """Group index of each value for _cut_labels (NaN stays NaN)."""
    x = np.asarray(x, float)
    g = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    if len(cuts) == 1:
        g[ok] = (x[ok] > cuts[0]).astype(float)
        return g
    idx = np.searchsorted(cuts, x[ok], side="right").astype(float)     # interior groups include their lower cut
    idx[x[ok] <= cuts[0]] = 0                                           # first group includes c1
    idx[x[ok] >= cuts[-1]] = len(cuts)                                  # last group includes cm
    g[ok] = idx
    return g


def _derive(data, spec, al_col, s, name, relabel):
    """One subgroup definition in the form the engine takes, with its description. Cut points (and the default
    axial length groups) become an ordered categorical column with the labels above; other definitions pass
    through."""
    d = osx.parse_subgroup_spec(spec) if isinstance(spec, str) else dict(spec)
    col = d["column"]
    if col not in data.columns:
        raise KeyError(f"subgroup column {col!r} is not in the data")
    default_al = col == al_col and d["cuts"] is None and d["bins"] is None
    if not default_al and d["cuts"] is None:
        return spec, osx.define_subgroups(data, col, None, d["bins"])[1]["summary"]
    if default_al:
        cuts = [float(s["axl_short"]), float(s["axl_long"])]
        _, generic = _cut_labels(col, cuts, " mm")
        labels = [f"Short ({generic[0]})", f"Medium ({generic[1]})", f"Long ({generic[2]})"]
        summary = (f"{col}: short ≤{osx._num(cuts[0])} mm, medium >{osx._num(cuts[0])} to <{osx._num(cuts[1])} mm, "
                   f"long ≥{osx._num(cuts[1])} mm")
    else:
        cuts, labels = _cut_labels(col, d["cuts"])
        summary = (f"{col}, cut points {', '.join(osx._num(c) for c in cuts)}: "
                   f"{', '.join(lb[len(col) + 1:] for lb in labels)} (the first group includes the first cut point "
                   "and the last group the last)")
    g = _cut_groups(pd.to_numeric(data[col], errors="coerce"), cuts)
    if np.isnan(g).any():
        summary += f"; {int(np.isnan(g).sum())} eye(s) with no {col} value are in All only"
    data[name] = pd.Categorical.from_codes(np.where(np.isnan(g), -1, g).astype(int), categories=labels, ordered=True)
    relabel += [(f"{name} = {lb}", lb) for lb in labels] + [(name, col)]
    return name, summary


def _relabel(obj, pairs):
    """Replace the engine's names for derived subgroups (in strings and dictionary keys, recursively)."""
    if isinstance(obj, str):
        for old, new in pairs:
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, dict):
        return {_relabel(k, pairs): _relabel(v, pairs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relabel(v, pairs) for v in obj]
    return obj


def _to_formulas(text):
    return text.replace("models", "formulas").replace("model", "formula").replace("Models", "Formulas")


def _iol_methodology(rows, s, sub_desc, cross_desc):
    out = [{"Test Category": "Excluded eyes",
            "Methodology Applied": f"Eyes with a missing PE, or |PE| > {s['invalid_threshold']:.2f} D for any formula, "
                                   "are excluded from all comparisons"}]
    for r in rows:
        cat, txt = r["Test Category"], r["Methodology Applied"]
        if cat == "Weighting of Eyes (mean-loss metrics)":
            continue
        if cat == "Tolerance (proportion within)":
            txt = f"|PE| <= {s['mcnemar_threshold']:.2f} D"
        if cat == "Subgroups":
            txt = "; ".join(sub_desc) + ". Each subgroup is analyzed separately with the same tests"
        if cat == "Crossed subgroups":
            txt = (f"{cross_desc[0]}, crossed with {cross_desc[1]}. Cells with fewer than {s['cross_min_patients']} "
                   "patients get point estimates only, without intervals or tests")
        out.append({"Test Category": cat, "Methodology Applied": _to_formulas(txt)})
    out.append({"Test Category": "Formula performance",
                "Methodology Applied": "Each formula's MPE, MAE, MedAE, RMSE, SD and percentage of eyes within "
                                       f"±{s['mcnemar_threshold']:.2f} D, with a {s['ci_level'] * 100:g}% CI from the "
                                       "percentile patient bootstrap"})
    return out


def analyze_prediction_errors(df, axl_data=None, settings=None, subgroup=None, cross=None, formulas=None,
                              patient=None, al_groups=True):
    """Compare every pair of formulas, for all eyes and for each subgroup.

    df        one row per eye: a patient ID column and one column of PE (D) per formula
    axl_data  optional axial lengths (a Series aligned with df), if df has no AXL column
    subgroup  a column or a list of them, each analyzed separately: 'eye', 'K1:bins=3' (equal-count groups) or
              'AXL:cuts=22,24.5,26' (cut points: ≤22, >22 to <24.5, 24.5 to <26, ≥26)
    cross     optional [rows, columns] subgroups crossed with each other, for heat maps (Grid sheet)
    al_groups short, medium and long eyes are added when there is an axial length column (False: not)
    Returns the comparisons, with each formula's performance, the methods and the sample in .attrs."""
    s = _settings(settings)
    data = df.copy()
    if axl_data is not None:
        data["AXL"] = pd.Series(np.asarray(axl_data, float), index=data.index) if not hasattr(axl_data, "index") \
            else pd.to_numeric(axl_data, errors="coerce").reindex(data.index)
    specs = [] if subgroup is None else ([subgroup] if osx._is_single_spec(subgroup) else list(subgroup))
    cross_specs = None if cross is None else list(cross)
    sub_cols = [osx.parse_subgroup_spec(x)["column"] if isinstance(x, str) else x["column"]
                for x in specs + (cross_specs or [])]
    found, pid, al_col = detect_columns(data, extra_ignore=sub_cols)
    if axl_data is not None:
        al_col = "AXL"
    formulas = list(formulas) if formulas is not None else found
    patient = patient if patient is not None else pid
    if len(formulas) < 2:
        raise ValueError(f"need at least two formula columns of prediction errors; found {formulas}")
    spec_cols = sub_cols[:len(specs)]
    if al_groups and al_col is not None and al_col not in spec_cols:
        specs.insert(0, al_col)
    relabel, sub_desc, cross_desc, used = [], [], [], set(data.columns)

    def name_for(x):                                     # a new column for each derived grouping, e.g. 'AXL groups'
        col = osx.parse_subgroup_spec(x)["column"] if isinstance(x, str) else x["column"]
        name, k = f"{col} groups", 2
        while name in used:
            name, k = f"{col} groups ({k})", k + 1
        used.add(name)
        return name

    for k, x in enumerate(specs):
        specs[k], desc = _derive(data, x, al_col, s, name_for(x), relabel)
        sub_desc.append(desc)
    for k, x in enumerate(cross_specs or []):
        cross_specs[k], desc = _derive(data, x, al_col, s, name_for(x), relabel)
        cross_desc.append(desc)
    eng = {"bootstrap_iterations": s["bootstrap_iterations"], "seed": s["seed"], "fwer_method": s["fwer_method"],
           "report_ci": s["report_ci"], "ci_level": s["ci_level"], "within": s["mcnemar_threshold"],
           "invalid_threshold": s["invalid_threshold"], "one_eye": s["one_eye"],
           "min_group_size": s["min_group_size"], "cross_min_patients": s["cross_min_patients"],
           "verbose": s["verbose"]}
    res = osx.analyze(data, task="continuous", models=formulas, patient=patient, subgroup=specs or None,
                      cross=cross_specs, settings=eng)
    if res.empty:
        return res
    pairs = sorted(relabel, key=lambda p: -len(p[0]))
    attrs = {k: _relabel(v, pairs) for k, v in res.attrs.items()}
    res = res.rename(columns={"Model A": "Formula A", "Model B": "Formula B"})
    res["Group"] = res["Group"].map(lambda g: _relabel(g, pairs))
    rank = {c: next((r for pat, r in _BLOCK if re.search(pat, c)), -1) for c in res.columns}
    res = res[sorted(res.columns, key=lambda c: rank[c])]          # MPE, MAE, MedAE, RMSE, SD, within
    res.attrs = attrs
    res.attrs["methodology_summary"] = _iol_methodology(attrs.get("methodology_summary", []), s, sub_desc, cross_desc)
    res.attrs["settings"] = s
    return res


def formula_performance(results):
    """Each formula's MPE, MAE, MedAE, RMSE, SD and percentage within the tolerance, with CIs: one row per
    group and formula."""
    mp = pd.DataFrame(results.attrs.get("model_performance", []))
    if mp.empty:
        return mp
    s = results.attrs.get("settings", DEFAULT_SETTINGS)
    lbl = f"{s['ci_level'] * 100:g}% CI"
    within = f"% within ±{s['mcnemar_threshold']:.2f} D"
    rows = []
    for (g, f), d in mp.groupby(["group", "model"], sort=False):
        r = {"Group": g, "Formula": f, "Eyes": int(d.n_eyes.iloc[0]), "Patients": int(d.n_patients.iloc[0])}
        for key, name in METRICS:
            x = d[d.metric == key]
            if x.empty:
                continue
            k = 100.0 if key == "within" else 1.0
            name = within if key == "within" else name
            r[name] = k * x.estimate.iloc[0]
            r[f"{name} {lbl} low"] = k * x.ci_low.iloc[0]
            r[f"{name} {lbl} high"] = k * x.ci_high.iloc[0]
        rows.append(r)
    return pd.DataFrame(rows)


def estimates(results):
    """Every comparison, one row per group, pair of formulas and metric: both formulas' estimates, the difference
    with its CIs, p-values and adjustments (each with its method), ρΔ (the fellow-eye correlation of the per-eye
    difference between the two formulas) and the one-eye factor (how many times as many patients a study keeping
    one eye per patient needs for the same precision)."""
    est = pd.DataFrame(results.attrs.get("long", []))
    if est.empty:
        return est
    s = results.attrs.get("settings", DEFAULT_SETTINGS)
    names = dict(METRICS, within=f"Within ±{s['mcnemar_threshold']:.2f} D")
    est["metric"] = est["metric"].map(lambda m: names.get(m, m))
    return est.rename(columns={"model_a": "formula_a", "model_b": "formula_b"})


def _sample(results):
    gi = pd.DataFrame(results.attrs.get("group_info", {})).T.reset_index().rename(columns={"index": "Group"})
    gi = gi.drop(columns=[c for c in ("task", "truth", "weighting", "unit") if c in gi.columns])
    return gi.rename(columns={"models": "formulas"})


def write_advanced_stats_to_excel(results, writer):
    """Write Results, Statistical Tests, Formula Performance, Estimates and Sample into an open pandas
    ExcelWriter."""
    osx._write_results_sheet(results, writer)
    fp = formula_performance(results)
    if not fp.empty:
        fp.to_excel(writer, sheet_name="Formula Performance", index=False)
        osx._set_column_widths(writer, "Formula Performance", [30, 14, 8, 9] + [12] * (len(fp.columns) - 4))
    est = estimates(results)
    if not est.empty:
        est.to_excel(writer, sheet_name="Estimates", index=False)
        osx._set_column_widths(writer, "Estimates", [36 if c in ("test", "p_adj_method", "ci_adj_method") else 16
                                                     for c in est.columns])
    gi = _sample(results)
    if not gi.empty:
        gi.to_excel(writer, sheet_name="Sample", index=False)
        osx._set_column_widths(writer, "Sample", [30, 14, 60] + [16] * (len(gi.columns) - 3))


def write_results_to_excel(results, path):
    """Write the workbook: Results, Statistical Tests, Formula Performance, Estimates and Sample, plus Grid with
    crossed subgroups."""
    with pd.ExcelWriter(path) as writer:
        write_advanced_stats_to_excel(results, writer)
    if results.attrs.get("grid"):
        osx._write_grid_sheet(path, results)
        from openpyxl import load_workbook
        wb = load_workbook(path)
        ws = wb["Grid"]
        ws["A1"] = _to_formulas(str(ws["A1"].value))
        wb.save(path)


def _command_line(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Compare IOL power formulas with the patient as the unit of analysis.")
    ap.add_argument("data", help="Excel or CSV file: one row per eye, a patient ID column and one column of "
                                 "prediction errors (D) per formula")
    ap.add_argument("--sheet", default=0, help="Excel sheet (default: the first)")
    ap.add_argument("--formulas", nargs="+", default=None, help="formula columns (default: found by name)")
    ap.add_argument("--patient", default=None, help="patient ID column (default: patient_id, id, hosp_id, ...)")
    ap.add_argument("--subgroup", action="append", default=None, metavar="SPEC",
                    help="also give results for each subgroup of a column; repeat for several, e.g. --subgroup eye "
                         "--subgroup K1:bins=3 --subgroup AXL:cuts=22,24.5,26")
    ap.add_argument("--cross", nargs=2, default=None, metavar=("ROWS", "COLUMNS"),
                    help="cross two subgroups for heat maps (Grid sheet), e.g. --cross AXL K1:bins=3")
    ap.add_argument("--no-al-groups", action="store_true", help="leave out the short, medium and long eyes")
    ap.add_argument("--exclude-above", type=float, default=4.0, metavar="D",
                    help="exclude eyes with |PE| above D for any formula (default 4)")
    ap.add_argument("--within", type=float, default=0.50, metavar="D", help="tolerance for the proportion within (D)")
    ap.add_argument("--fwer", default="romano_wolf", choices=["romano_wolf", "hommel", "holm", "bonferroni",
                                                              "fdr_bh", "none"])
    ap.add_argument("--iterations", type=int, default=4999, help="bootstrap resamples (default 4999)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--one-eye", action="store_true", help="keep one randomly chosen eye per patient")
    ap.add_argument("--out", default="biometry_stats_results.xlsx")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.data) if str(a.data).lower().endswith(".csv") else pd.read_excel(a.data, sheet_name=a.sheet)
    res = analyze_prediction_errors(df, subgroup=a.subgroup, cross=a.cross, formulas=a.formulas, patient=a.patient,
                                    al_groups=not a.no_al_groups,
                                    settings={"invalid_threshold": a.exclude_above, "mcnemar_threshold": a.within,
                                              "fwer_method": a.fwer, "bootstrap_iterations": a.iterations,
                                              "seed": a.seed, "one_eye": a.one_eye})
    write_results_to_excel(res, a.out)
    print(f"Results written to {a.out}")
    return res


if __name__ == "__main__":
    _command_line()
