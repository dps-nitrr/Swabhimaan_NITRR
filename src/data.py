"""Data loading, windowing, train/val/test split and scaling.

A dataset folder contains
  cgm.csv     : patient_id, timestamp, <time-series columns...>        (one row per reading / event)
  ehr.csv     : patient_id, [record, patient,] <static EHR columns...> (one row per patient_id)
  schema.json : (optional) which columns to use and how to split. Without it the SYNTHETIC defaults apply.

patient_id identifies one continuous recording ("series"). In real data one person can have several
recordings; the optional `patient` column in ehr.csv groups them so that the patient-level split never
puts the same person in train AND test.

Split modes
  within_patient : every series is cut by time (first 70% train, next 10% val, last 20% test).
  by_patient     : whole PEOPLE go to train / val / test (70/10/20 of people). Used for short real records.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DEFAULT_SCHEMA = {
    "name": "synthetic",
    "ts_cols": ["glucose", "carbs", "steps", "insulin", "hour_sin", "hour_cos"],
    "sum_cols": ["carbs", "steps", "insulin"],          # summed when resampling (glucose is averaged)
    "ehr_cols": ["age", "sex", "bmi", "hba1c", "diabetes_years", "on_insulin", "on_metformin", "sbp"],
    "split": "within_patient",
}
TS_COLS = DEFAULT_SCHEMA["ts_cols"]                     # kept for backwards compatibility
EHR_COLS = DEFAULT_SCHEMA["ehr_cols"]
STEP = "15min"
L_DEFAULT = 24        # input window: 24 x 15 min = 6 hours
H_DEFAULT = 8         # forecast horizon: 8 x 15 min = 2 hours
HYPO, HYPER = 70.0, 180.0


def load_schema(data_dir):
    p = Path(data_dir) / "schema.json"
    return load_json(p) if p.exists() else dict(DEFAULT_SCHEMA)


def load_raw(data_dir):
    d = Path(data_dir)
    cgm = pd.read_csv(d / "cgm.csv")
    # real files can mix timestamp formats (with / without seconds); parse each value on its own
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], format="mixed", errors="coerce")
    cgm = cgm[cgm["timestamp"].notna()].reset_index(drop=True)
    ehr = pd.read_csv(d / "ehr.csv")
    return cgm, ehr


def fill_small_gaps(x, max_gap):
    """Linearly interpolate NaN runs of length <= max_gap; leave longer gaps as NaN."""
    x = x.copy()
    isn = np.isnan(x)
    n, i = len(x), 0
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1], x[j]])
            i = j
        else:
            i += 1
    return x


def patient_frames(cgm, schema=None, max_gap=2):
    """patient_id -> DataFrame on a regular 15-min grid (columns = schema ts_cols).

    Glucose is averaged per bin, event columns are summed. Glucose gaps <= 30 min are interpolated."""
    schema = schema or DEFAULT_SCHEMA
    base = [c for c in schema["ts_cols"] if c not in ("hour_sin", "hour_cos")]
    agg = {c: ("sum" if c in schema["sum_cols"] else "mean") for c in base}
    out = {}
    for pid, df in cgm.groupby("patient_id"):
        r = df.set_index("timestamp").sort_index()[base].resample(STEP).agg(agg)
        r["glucose"] = fill_small_gaps(r["glucose"].to_numpy(float), max_gap)
        hour = r.index.hour + r.index.minute / 60
        r["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        r["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out[int(pid)] = r[schema["ts_cols"]]
    return out


def patient_arrays(cgm, schema=None, max_gap=2):
    """patient_id -> (n, F) numpy array on the regular grid."""
    return {pid: f.to_numpy(float) for pid, f in patient_frames(cgm, schema, max_gap).items()}


def make_windows(arr, L, H):
    """Sliding windows. X: (N, L, F) past; Y: (N, H) future glucose; t: index of last input step."""
    n = len(arr)
    if n < L + H:
        return (np.empty((0, L, arr.shape[1])), np.empty((0, H)), np.empty(0, dtype=int))
    wins = sliding_window_view(arr, L, axis=0).transpose(0, 2, 1)     # (n-L+1, L, F)
    m = n - H - L + 1
    X = wins[:m]
    Y = sliding_window_view(arr[:, 0], H)[L:L + m]
    t = np.arange(L - 1, L - 1 + m)
    ok = ~np.isnan(X[:, :, 0]).any(1) & ~np.isnan(Y).any(1)
    return X[ok], Y[ok], t[ok]


def events(Y):
    """Adverse-event labels from the next-2h glucose: [hypo, hyper] as float (N, 2)."""
    return np.stack([(Y < HYPO).any(1), (Y > HYPER).any(1)], axis=1).astype(np.float32)


def assign_groups(groups, fracs, seed):
    """Randomly assign each distinct group (person) to train / val / test."""
    ids = np.array(sorted({int(g) for g in groups}))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr, n_va = int(round(len(ids) * fracs[0])), int(round(len(ids) * fracs[1]))
    parts = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}
    return {int(g): name for name, arr in parts.items() for g in arr}


def prepare(data_dir, L=L_DEFAULT, H=H_DEFAULT, fracs=(0.7, 0.1, 0.2), seed=0):
    """Build train / val / test windows (no leakage) and the scaler (fitted on TRAIN only)."""
    schema = load_schema(data_dir)
    cgm, ehr_raw = load_raw(data_dir)
    arrays = patient_arrays(cgm, schema)
    ehr_raw = ehr_raw.set_index("patient_id")
    group_of = ehr_raw["patient"] if "patient" in ehr_raw.columns else pd.Series(ehr_raw.index, index=ehr_raw.index)
    feat = ehr_raw[schema["ehr_cols"]].astype(float)

    by_patient = schema.get("split") == "by_patient"
    if by_patient:
        person = assign_groups([group_of[p] for p in arrays], fracs, seed)
        assignment = {pid: person[int(group_of[pid])] for pid in arrays}
    else:
        assignment = {pid: "all" for pid in arrays}
    train_ids = [p for p, s in assignment.items() if s in ("train", "all")]
    feat = feat.fillna(feat.loc[train_ids].mean()).fillna(0.0)          # missing labs -> TRAIN mean

    keys = ("X", "Y", "E", "pid", "t")
    splits = {k: {x: [] for x in keys} for k in ("train", "val", "test")}

    def add(name, pid, arr, offset):
        X, Y, t = make_windows(arr, L, H)
        if len(X) == 0:
            return
        s = splits[name]
        s["X"].append(X)
        s["Y"].append(Y)
        s["t"].append(t + offset)
        s["pid"].append(np.full(len(X), pid))
        s["E"].append(np.repeat(feat.loc[[pid]].to_numpy(), len(X), axis=0))

    for pid, arr in arrays.items():
        if by_patient:
            add(assignment[pid], pid, arr, 0)
        else:
            n = len(arr)
            b1, b2 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
            add("train", pid, arr[:b1], 0)
            add("val", pid, arr[b1:b2], b1)
            add("test", pid, arr[b2:], b2)
    for name in splits:
        splits[name] = {k: np.concatenate(v) for k, v in splits[name].items()}

    flat = splits["train"]["X"].reshape(-1, len(schema["ts_cols"]))
    tr_feat = feat.loc[train_ids]
    scaler = {
        "ts_mean": flat.mean(0).tolist(),
        "ts_std": np.maximum(flat.std(0), 1e-6).tolist(),
        "ehr_mean": tr_feat.mean().to_numpy().tolist(),
        "ehr_std": np.maximum(tr_feat.std().fillna(0).to_numpy(), 1e-6).tolist(),
    }
    return {"splits": splits, "scaler": scaler, "arrays": arrays, "ehr": feat, "L": L, "H": H,
            "schema": schema, "assignment": assignment, "group_of": group_of}


def normalize(split, scaler):
    """Raw split -> float32 arrays ready for the network (X, E, Y scaled; EV = event labels)."""
    tm, ts = np.array(scaler["ts_mean"]), np.array(scaler["ts_std"])
    em, es = np.array(scaler["ehr_mean"]), np.array(scaler["ehr_std"])
    X = ((split["X"] - tm) / ts).astype(np.float32)
    E = ((split["E"] - em) / es).astype(np.float32)
    Y = ((split["Y"] - tm[0]) / ts[0]).astype(np.float32)
    return X, E, Y, events(split["Y"])


def select(split, pid):
    m = split["pid"] == pid
    return {k: v[m] for k, v in split.items()}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())
