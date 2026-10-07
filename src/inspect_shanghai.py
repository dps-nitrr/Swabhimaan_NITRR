"""Print the structure of the Shanghai T2DM files so the converter can be written correctly.

Usage (from the project root):
    python -m src.inspect_shanghai data/raw/shanghai

Writes everything to inspect_output.txt (UTF-8). It only READS the dataset.
v2: prints the reason when a file cannot be read, plus food / insulin patterns and glucose statistics.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/shanghai")
OUT = Path("inspect_output.txt")
sys.stdout = open(OUT, "w", encoding="utf-8")   # avoids Windows console encoding errors
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 90)


def read_any(path, **kw):
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    return pd.read_csv(path, **kw)


files = sorted(p for p in ROOT.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv"))
summary = [p for p in files if "summary" in p.name.lower()]
records = [p for p in files if "summary" not in p.name.lower()]

print("=" * 70)
print("FOLDER:", ROOT.resolve())
print("record files:", len(records), "| summary files:", [p.name for p in summary])
print("file extensions:", Counter(p.suffix.lower() for p in records))
print("file size KB (min/median/max):", *(int(np.percentile([p.stat().st_size / 1024 for p in records], q)) for q in (0, 50, 100)))

# ---------------- overview over ALL record files ----------------
rows, failed = [], []
colsets, food, ins_pat, glucose_all = {}, Counter(), Counter(), []
n_diet_events = n_ins_events = n_diet_on_cgm = 0
span_days, step_counts = [], Counter()

for p in records:
    try:
        d = read_any(p)
    except Exception as e:                      # <-- the reason why a file cannot be read
        failed.append((p.name, f"{type(e).__name__}: {str(e)[:200]}"))
        rows.append((p.name, -1, -1, -1))
        continue
    cgm_col = next((c for c in d.columns if "cgm" in str(c).lower()), None)
    n_cgm = int(d[cgm_col].notna().sum()) if cgm_col is not None else -1
    rows.append((p.name, len(d), n_cgm, len(d.columns)))
    colsets.setdefault(tuple(map(str, d.columns)), []).append(p.name)
    if cgm_col is None:
        continue
    g = pd.to_numeric(d[cgm_col], errors="coerce")
    glucose_all.append(g.dropna().to_numpy())
    try:
        t = pd.to_datetime(d[d.columns[0]])
        span_days.append((t.max() - t.min()).total_seconds() / 86400)
        step_counts[str(t.diff().mode().iloc[0])] += 1
    except Exception:
        pass
    diet_cols = [c for c in d.columns if "diet" in str(c).lower() or "饮食" in str(c) or "进食" in str(c)]
    dc = next((c for c in diet_cols if d[c].notna().any()), None)
    if dc is not None:
        vals = d[dc].dropna().astype(str)
        n_diet_events += len(vals)
        n_diet_on_cgm += int(g[vals.index].notna().sum())
        for v in vals:
            for line in re.split(r"[\n;,，；]+", v):
                name = re.sub(r"[\d.]+\s*(g|ml|克)?", "", line, flags=re.I).strip(" :-()")
                if name:
                    food[name.lower()] += 1
    for c in d.columns:
        if "insulin dose - s.c" in str(c).lower():
            vals = d[c].dropna().astype(str)
            n_ins_events += len(vals)
            for v in vals:
                ins_pat[re.sub(r"\d+(\.\d+)?", "N", v)] += 1

ov = pd.DataFrame(rows, columns=["file", "rows", "cgm_values", "n_cols"])
print("\n" + "=" * 70)
print("OVERVIEW (-1 means the file could NOT be read)")
print(ov.describe().round(1).to_string())
print(f"\nFILES THAT FAILED TO READ: {len(failed)} of {len(records)}")
print("error types:", Counter(m.split(':')[0] for _, m in failed))
for name, msg in failed[:12]:
    print("  ", name, "->", msg)

print("\nnumber of distinct column layouts among readable files:", len(colsets))
for cols, names in colsets.items():
    print(f"  {len(names)} files, e.g. {names[0]}: {list(cols)}")

print("\n" + "=" * 70)
print("RECORD LENGTH (days) over readable files:", pd.Series(span_days).describe().round(1).to_dict())
print("time step per file (most common diff):", dict(step_counts))
if glucose_all:
    g = np.concatenate(glucose_all)
    print(f"\nGLUCOSE over {len(g)} CGM values: mean {g.mean():.0f}, min {g.min():.0f}, max {g.max():.0f} mg/dL")
    print(f"  below 70: {(g < 70).mean():.1%} | 70-180: {((g >= 70) & (g <= 180)).mean():.1%} | above 180: {(g > 180).mean():.1%}")
print(f"\nDIET events: {n_diet_events} (of which on a row that has a CGM value: {n_diet_on_cgm})")
print(f"INSULIN s.c. events: {n_ins_events}")
print("\nMOST COMMON FOODS (name only, weights removed) - top 80:")
for k, v in food.most_common(80):
    print(f"  {v:5d}  {k}")
print("\nINSULIN STRING PATTERNS (numbers replaced by N) - top 25:")
for k, v in ins_pat.most_common(25):
    print(f"  {v:5d}  {k!r}")

# ---------------- summary file ----------------
for p in summary:
    print("\n" + "=" * 70)
    print("SUMMARY FILE:", p.name)
    try:
        s = read_any(p)
        print("shape:", s.shape)
        print("first column sample:", list(s.iloc[:5, 0]))
        print("\nHypoglycemic Agents - most common values:")
        for c in s.columns:
            if "hypoglycemic agents" in str(c).lower():
                print(s[c].astype(str).value_counts().head(15).to_string())
        print("\nnumeric columns summary:\n", s.describe().T[["count", "mean", "min", "max"]].round(2).to_string())
    except Exception as e:
        print("could not read summary:", type(e).__name__, e)

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Done. Now open", OUT.resolve(), "and paste its content in the chat (or attach the file).")
