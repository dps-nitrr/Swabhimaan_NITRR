"""Convert the Shanghai T2DM dataset into GlucoTwin's format.

Dataset: Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023). License: CC BY 4.0.

    python -m src.convert_shanghai --raw data/raw/shanghai --out data/shanghai

Input : <raw>/Shanghai_T2DM/*.xls|*.xlsx   (one file per CGM record, 15-min CGM)
        <raw>/Shanghai_T2DM_Summary.xlsx   (one row per record: demographics + labs)
Output: <out>/cgm.csv, ehr.csv, schema.json, convert_report.txt

What is derived (and therefore approximate):
  * carbs         : estimated from the free-text diet log with data/food_carbs.csv
                    ("data not available" meals get the median meal of that time of day)
  * insulin_fast  : s.c. rapid / short / premixed insulin, units (+ pump bolus if present)
  * insulin_basal : s.c. long-acting insulin (degludec / detemir / glargine), units
  * HbA1c         : converted from mmol/mol to % (NGSP = 0.09148 x IFCC + 2.152)
No steps / activity exist in this dataset.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASAL_WORDS = ("degludec", "detemir", "glargine", "glarigine")
INSULIN_WORDS = ("insulin", "novolin", "humulin", "gansulin", "scilin", "lantus", "levemir", "tresiba")
WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g|ml)(?![a-z])", re.I)

TS_COLS = ["glucose", "carbs", "insulin_fast", "insulin_basal", "hour_sin", "hour_cos"]
SUM_COLS = ["carbs", "insulin_fast", "insulin_basal"]
EHR_COLS = ["age", "female", "bmi", "hba1c", "diabetes_years", "on_insulin", "on_metformin",
            "fpg", "ppg_2h", "cpep_fasting", "cpep_2h", "egfr"]


# ----------------------------------------------------------------------------- helpers
def read_any(path):
    return pd.read_excel(path)


def find_col(df, *keys, any_of=()):
    """First column whose lower-cased name contains all `keys` (and, if given, one of `any_of`)."""
    for c in df.columns:
        name = str(c).lower().replace("\xa0", " ")
        if all(k in name for k in keys) and (not any_of or any(a in name for a in any_of)):
            return c
    return None


def load_food_table(path):
    t = pd.read_csv(path, comment="#")
    t["keyword"] = t["keyword"].astype(str).str.lower().str.strip()
    t = t.sort_values("keyword", key=lambda s: s.str.len(), ascending=False)
    return list(zip(t["keyword"], t["carbs_g_per_100g"].astype(float)))


def parse_meal(text, table):
    """Diet cell -> (carbs in g or None if unknown, unmatched [(name, grams)], n_items, total grams)."""
    t = str(text).replace("\xa0", " ").strip()
    if not t or "not available" in t.lower():
        return None, [], 0, 0.0
    total, unmatched, n_items, grams_sum = 0.0, [], 0, 0.0
    for line in re.split(r"[\n;；,，]+", t):
        line = line.strip()
        if not line:
            continue
        m = WEIGHT_RE.search(line)
        if m:
            grams, name = float(m.group(1)), line[:m.start()] + " " + line[m.end():]
        else:
            m2 = re.search(r"(\d+(?:\.\d+)?)\s*$", line)
            grams = float(m2.group(1)) if m2 else 100.0
            name = line[:m2.start()] if m2 else line
        name = re.sub(r"\s+", " ", name).strip(" :-()").lower()
        if not name:
            continue
        n_items += 1
        grams_sum += grams
        per100 = next((c for k, c in table if k in name), None)
        if per100 is None:
            unmatched.append((name, grams))
        else:
            total += grams * per100 / 100.0
    return min(total, 250.0), unmatched, n_items, grams_sum


def parse_insulin(text):
    """'insulin glargine, 12 IU, Humulin 70/30, 8 IU' -> (fast_units, basal_units, parsed_ok)."""
    t = str(text).replace("\xa0", " ")
    fast = basal = 0.0
    ok = False
    for seg in re.split(r"IU", t, flags=re.I)[:-1]:           # every segment ends right before 'IU'
        s = seg.rstrip(" ,;")
        m = re.search(r"(\d+(?:\.\d+)?)$", s)
        if not m:
            continue
        ok = True
        dose, name = float(m.group(1)), s[:m.start()].lower()
        if any(w in name for w in BASAL_WORDS):
            basal += dose
        else:
            fast += dose
    return min(fast, 100.0), min(basal, 100.0), ok


def hour_bucket(ts):
    h = ts.hour
    return "breakfast" if 5 <= h < 10 else "lunch" if 10 <= h < 15 else "snack" if 15 <= h < 18 else "dinner"


# ----------------------------------------------------------------------------- one record
def convert_record(path, table, st):
    df = read_any(path)
    dcol, gcol = df.columns[0], find_col(df, "cgm")
    if gcol is None:
        st["no_cgm_col"].append(path.name)
        return None
    df = df.copy()
    df["_ts"] = pd.to_datetime(df[dcol], errors="coerce")
    df["_g"] = pd.to_numeric(df[gcol], errors="coerce")
    df = df[df["_ts"].notna()].sort_values("_ts").drop_duplicates("_ts").reset_index(drop=True)

    n = len(df)
    carbs, unknown = np.zeros(n), np.zeros(n, bool)
    fast, basal = np.zeros(n), np.zeros(n)

    diet = find_col(df, "dietary")
    if diet is None or not df[diet].notna().any():
        cn = [c for c in df.columns if "饮食" in str(c) or "进食" in str(c)]
        if cn and df[cn[0]].notna().any():
            st["diet_chinese_only"].append(path.name)
        diet = None
    if diet is not None:
        for i in df.index[df[diet].notna()]:
            c, unm, n_items, gsum = parse_meal(df.at[i, diet], table)
            st["diet_events"] += 1
            if c is None:
                unknown[i] = True
                st["diet_unknown"] += 1
            else:
                carbs[i] = c
                st["meal_carbs"].append(c)
                for name, grams in unm:
                    st["unmatched"][name] += grams
                st["food_grams"] += sum(g for _, g in unm)
                st["all_grams"] += gsum
                st["items"] += n_items

    sc = find_col(df, "s.c")
    if sc is not None:
        for i in df.index[df[sc].notna()]:
            f, b, ok = parse_insulin(df.at[i, sc])
            if not ok:
                st["insulin_unparsed"].append(str(df.at[i, sc])[:60])
                continue
            fast[i] += f
            basal[i] += b
            st["ins_fast_events"] += f > 0
            st["ins_basal_events"] += b > 0

    bolus = find_col(df, "csii", "bolus")
    if bolus is not None:
        v = pd.to_numeric(df[bolus], errors="coerce").fillna(0).to_numpy()
        fast += v
        if (v > 0).any():
            st["pump_records"] += 1
    basal_rate = next((c for c in df.columns if "csii" in str(c).lower() and "bolus" not in str(c).lower()
                       or "基础" in str(c)), None)
    if basal_rate is not None and pd.to_numeric(df[basal_rate], errors="coerce").notna().any():
        st["pump_basal_ignored"] += 1
    iv = find_col(df, "i.v")
    if iv is not None and df[iv].notna().any():
        st["iv_records"] += 1

    out = pd.DataFrame({"timestamp": df["_ts"], "glucose": df["_g"], "carbs": carbs,
                        "insulin_fast": fast, "insulin_basal": basal, "unknown_meal": unknown})
    return out


# ----------------------------------------------------------------------------- summary -> EHR
def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def build_ehr(summary_path, stems):
    s = pd.read_excel(summary_path)
    key = s.columns[0]
    s[key] = s[key].astype(str).str.strip()
    s = s.drop_duplicates(key).set_index(key)

    def col(*keys):
        c = find_col(s, *keys)
        return s[c] if c is not None else pd.Series(np.nan, index=s.index)

    agents = col("hypoglycemic agents").astype(str).str.lower()
    e = pd.DataFrame(index=s.index)
    e["age"] = to_num(col("age (years)"))
    e["female"] = (to_num(col("gender")) == 1).astype(float)
    e["bmi"] = to_num(col("bmi"))
    e["hba1c"] = to_num(col("hba1c")) * 0.09148 + 2.152
    e["diabetes_years"] = to_num(col("duration"))
    e["on_insulin"] = agents.apply(lambda a: float(any(w in a for w in INSULIN_WORDS)))
    e["on_metformin"] = agents.apply(lambda a: float("metformin" in a))
    e["fpg"] = to_num(col("fasting plasma glucose"))
    e["ppg_2h"] = to_num(col("2-hour postprandial plasma glucose"))
    e["cpep_fasting"] = to_num(col("fasting c-peptide"))
    e["cpep_2h"] = to_num(col("2-hour postprandial c-peptide"))
    e["egfr"] = to_num(col("glomerular"))
    # NOTE: 'Hypoglycemia (yes/no)' is deliberately NOT used: it may describe the very CGM period we
    # predict, which would leak the outcome.
    return e.reindex(stems)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/shanghai")
    ap.add_argument("--out", default="data/shanghai")
    ap.add_argument("--food", default="data/food_carbs.csv")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = load_food_table(args.food)
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in (".xls", ".xlsx")
                   and "summary" not in p.name.lower() and "t1dm" not in str(p).lower()
                   and "__macosx" not in str(p).lower() and not p.name.startswith("._"))
    summary = next(p for p in raw.rglob("*") if "summary" in p.name.lower() and "t2dm" in p.name.lower()
                   and not p.name.startswith("._") and "__macosx" not in str(p).lower())

    st = {"diet_events": 0, "diet_unknown": 0, "meal_carbs": [], "unmatched": Counter(), "food_grams": 0.0, "all_grams": 0.0,
          "items": 0, "ins_fast_events": 0, "ins_basal_events": 0, "insulin_unparsed": [], "pump_records": 0,
          "pump_basal_ignored": 0, "iv_records": 0, "no_cgm_col": [], "diet_chinese_only": []}
    recs, failed = [], []
    for p in files:
        try:
            r = convert_record(p, table, st)
        except Exception as ex:                     # keep going, but report it
            failed.append((p.name, f"{type(ex).__name__}: {str(ex)[:120]}"))
            continue
        if r is not None and len(r):
            recs.append((p.stem, r))

    # unknown meals ("data not available") -> median meal of the same time of day
    known = pd.concat([r.loc[r.carbs > 0, ["timestamp", "carbs"]] for _, r in recs])
    med = known.groupby(known["timestamp"].apply(hour_bucket))["carbs"].median().to_dict()
    overall = float(known["carbs"].median())
    for _, r in recs:
        idx = r.index[r["unknown_meal"]]
        if len(idx) == 0:
            continue
        r.loc[idx, "carbs"] = [med.get(hour_bucket(t), overall) for t in r.loc[idx, "timestamp"]]

    stems = [s for s, _ in recs]
    ehr = build_ehr(summary, stems)
    no_summary = [s for s in stems if ehr.loc[s].isna().all()]

    cgm = pd.concat([r.drop(columns="unknown_meal").assign(patient_id=i) for i, (_, r) in enumerate(recs)],
                    ignore_index=True)
    cgm = cgm[["patient_id", "timestamp", "glucose", "carbs", "insulin_fast", "insulin_basal"]]
    cgm["glucose"] = cgm["glucose"].round(1)
    cgm["carbs"] = cgm["carbs"].round(1)
    ehr_out = ehr.reset_index(drop=True)
    ehr_out.insert(0, "patient", [int(s.split("_")[0]) for s in stems])
    ehr_out.insert(0, "record", stems)
    ehr_out.insert(0, "patient_id", range(len(stems)))

    cgm.to_csv(out / "cgm.csv", index=False)
    ehr_out.to_csv(out / "ehr.csv", index=False)
    (out / "schema.json").write_text(json.dumps({
        "name": "shanghai_t2dm", "ts_cols": TS_COLS, "sum_cols": SUM_COLS, "ehr_cols": EHR_COLS,
        "split": "by_patient"}, indent=2))

    # ---- report ----
    g = cgm["glucose"].dropna()
    mc = np.array(st["meal_carbs"]) if st["meal_carbs"] else np.array([0.0])
    unm = st["unmatched"].most_common(30)
    lines = [
        f"records converted: {len(recs)} of {len(files)} | failed: {len(failed)} {failed[:5]}",
        f"records without CGM column: {st['no_cgm_col']}",
        f"patients: {ehr_out['patient'].nunique()} | rows: {len(cgm)} | summary rows missing: {no_summary}",
        f"glucose: mean {g.mean():.0f}, <70: {(g < 70).mean():.1%}, 70-180: {((g >= 70) & (g <= 180)).mean():.1%}, >180: {(g > 180).mean():.1%}",
        f"diet events: {st['diet_events']} | 'data not available': {st['diet_unknown']} "
        f"({st['diet_unknown'] / max(st['diet_events'], 1):.0%}) -> filled with median meal by time of day {({k: round(v) for k, v in med.items()})}",
        f"records where diet text exists only in the Chinese column (skipped): {st['diet_chinese_only']}",
        f"estimated carbs per known meal: median {np.median(mc):.0f} g, 10-90%: {np.percentile(mc, 10):.0f}-{np.percentile(mc, 90):.0f} g, max {mc.max():.0f} g",
        f"food items: {st['items']} | grams of food with NO carb-table match: {st['food_grams']:.0f} of {st['all_grams']:.0f} g "
        f"({st['food_grams'] / max(st['all_grams'], 1e-9):.1%}); unmatched foods count as 0 g carbs",
        "top unmatched foods (name, total grams):",
        *[f"    {n}: {int(w)}" for n, w in unm],
        f"insulin events: fast/mixed {int(st['ins_fast_events'])}, basal (long-acting) {int(st['ins_basal_events'])}",
        f"insulin strings that could not be parsed ({len(st['insulin_unparsed'])}): {Counter(st['insulin_unparsed']).most_common(8)}",
        f"records with pump bolus: {st['pump_records']} | pump basal rate present but ignored: {st['pump_basal_ignored']} | i.v. insulin ignored: {st['iv_records']}",
        "EHR missing values per feature (filled with the TRAIN mean later):",
        *[f"    {c}: {int(ehr[c].isna().sum())}" for c in EHR_COLS],
        f"EHR sample (first 3 records):\n{ehr.head(3).round(2).T.to_string()}",
    ]
    report = "\n".join(lines)
    (out / "convert_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out/'cgm.csv'}, {out/'ehr.csv'}, {out/'schema.json'}, {out/'convert_report.txt'}")


if __name__ == "__main__":
    main()
