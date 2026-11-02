"""
Step 0: export the eight Cannon et al. datasets to plain CSV so R and Python read identical numbers.

Reads  an Excel workbook with one sheet per dataset (path as the first argument; default data/datasets.xlsx)
Writes prepared/<id>.csv (complete cases only, six formula columns) and prepared/datasets.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "data" / "datasets.xlsx"
OUT = BASE / "prepared"
OUT.mkdir(exist_ok=True)

# sheet name in the authors' workbook -> short id (order = order reported in the paper); edit for your own workbook
DATASETS = {
    "Oklahoma 114 ZCB00 Control eyes": "OK_ZCB00",
    "Cyprus 135 Auto-ref WF eyes": "Cyprus",
    "Italy 216 Aladdin": "Italy",
    "Japan 569 WF": "Japan",
    "Oklahoma 713 WF eyes": "OK_WF",
    "Portugal 825 eyes": "Portugal",
    "France Dataset": "France",
    "Korean Dataset": "Korea",
}
FORMULAS = ["HQ", "H1", "Haigis", "SRK/T", "BU2", "K6"]   # column order = WHWK variable order 1..6
SAFE = {f: f.replace("/", "") for f in FORMULAS}          # SRK/T -> SRKT for CSV headers

# Eye counts reported by Cannon et al. (Am J Ophthalmol 2025;273:33-42, Table 1); used as a check
EXPECTED = {"OK_ZCB00": 114, "Cyprus": 135, "Italy": 216, "Japan": 569, "OK_WF": 713, "Portugal": 825, "France": 2631, "Korea": 2636}
# The Canadian ('France') sheet was not yet zeroed: its 'Dataset Descriptions' note says
# "Had to add -0.07297 to each eye to zero the mean". Applied here so every dataset has mean prediction error 0.
SHIFT = {"France": -0.07297}

rows = []
for sheet, did in DATASETS.items():
    raw = pd.read_excel(XLSX, sheet_name=sheet)
    # eyes only: stop at the first fully empty row (the Italian sheet has summary rows MAD, SD and
    # Geary ratio below a blank row, which must not be analysed as eyes)
    blank = raw[FORMULAS].isna().all(axis=1).to_numpy()
    if blank.any():
        raw = raw.iloc[: int(blank.argmax())]
    x = raw[FORMULAS].apply(pd.to_numeric, errors="coerce") + SHIFT.get(did, 0.0)
    complete = x.dropna()
    assert len(complete) == EXPECTED[did], f"{did}: {len(complete)} eyes, expected {EXPECTED[did]}"
    complete.rename(columns=SAFE).to_csv(OUT / f"{did}.csv", index=False, float_format="%.17g")
    rows.append({
        "id": did, "sheet": sheet, "rows_in_sheet": len(x), "n_analysed": len(complete),
        "n_incomplete_dropped": len(x) - len(complete),
        "max_abs_PE": float(np.abs(complete.values).max()),
        **{f"mean_PE_{SAFE[f]}": complete[f].mean() for f in FORMULAS},
        **{f"SD_PE_{SAFE[f]}": complete[f].std(ddof=1) for f in FORMULAS},
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "datasets.csv", index=False)
print(summary[["id", "rows_in_sheet", "n_analysed", "n_incomplete_dropped", "max_abs_PE"]].to_string(index=False))
print(f"Total eyes analysed: {summary.n_analysed.sum()}")
