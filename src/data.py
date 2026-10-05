"""Data loading, windowing, train/val/test split and scaling.

Expected input schema (this is the contract for REAL data too):
  cgm.csv : patient_id, timestamp, glucose, carbs, steps, insulin
  ehr.csv : patient_id, age, sex, bmi, hba1c, diabetes_years, on_insulin, on_metformin, sbp

Anything with a different format (e.g. the Shanghai T2DM dataset) just needs a
small converter script that writes these two CSVs. The rest of the project
does not change.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TS_COLS = ["glucose", "carbs", "steps", "insulin", "hour_sin", "hour_cos"]
EHR_COLS = ["age", "sex", "bmi", "hba1c", "diabetes_years",
            "on_insulin", "on_metformin", "sbp"]
STEP = "15min"
L_DEFAULT = 24        # input window: 24 x 15 min = 6 hours
H_DEFAULT = 8         # forecast horizon: 8 x 15 min = 2 hours
HYPO, HYPER = 70.0, 180.0


def load_raw(data_dir):
    d = Path(data_dir)
    cgm = pd.read_csv(d / "cgm.csv", parse_dates=["timestamp"])
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


def patient_arrays(cgm, max_gap=2):
    """pid -> (n, 6) array on a regular 15-min grid. Gaps <= 30 min are interpolated."""
    out = {}
    for pid, df in cgm.groupby("patient_id"):
        df = df.set_index("timestamp").sort_index()
        agg = df.resample(STEP).agg({"glucose": "mean", "carbs": "sum",
                                     "steps": "sum", "insulin": "sum"})
        agg["glucose"] = fill_small_gaps(agg["glucose"].to_numpy(float), max_gap)
        hour = agg.index.hour + agg.index.minute / 60
        agg["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        agg["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out[int(pid)] = agg[TS_COLS].to_numpy(float)
    return out


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


def prepare(data_dir, L=L_DEFAULT, H=H_DEFAULT, fracs=(0.7, 0.1, 0.2)):
    """Time-based split inside every patient (no leakage): first 70% train, next 10% val, last 20% test."""
    cgm, ehr = load_raw(data_dir)
    arrays = patient_arrays(cgm)
    ehr = ehr.set_index("patient_id")[EHR_COLS].astype(float)

    keys = ("X", "Y", "E", "pid", "t")
    splits = {k: {x: [] for x in keys} for k in ("train", "val", "test")}
    for pid, arr in arrays.items():
        n = len(arr)
        b1, b2 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
        for name, (a, b) in {"train": (0, b1), "val": (b1, b2), "test": (b2, n)}.items():
            X, Y, t = make_windows(arr[a:b], L, H)
            if len(X) == 0:
                continue
            s = splits[name]
            s["X"].append(X)
            s["Y"].append(Y)
            s["t"].append(t + a)
            s["pid"].append(np.full(len(X), pid))
            s["E"].append(np.repeat(ehr.loc[[pid]].to_numpy(), len(X), axis=0))
    for name in splits:
        splits[name] = {k: np.concatenate(v) for k, v in splits[name].items()}

    flat = splits["train"]["X"].reshape(-1, len(TS_COLS))
    scaler = {
        "ts_mean": flat.mean(0).tolist(),
        "ts_std": np.maximum(flat.std(0), 1e-6).tolist(),
        "ehr_mean": ehr.mean().to_numpy().tolist(),
        "ehr_std": np.maximum(ehr.std().to_numpy(), 1e-6).tolist(),
    }
    return {"splits": splits, "scaler": scaler, "arrays": arrays, "ehr": ehr, "L": L, "H": H}


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
