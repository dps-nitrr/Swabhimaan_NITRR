"""Personalization: turn the population model into a per-patient 'twin'.

For every patient we copy the population model, freeze the convolutional feature
extractor and fine-tune the rest on THAT patient's own early data only.
Then we compare population vs personalized error on that patient's later data.

  python -m src.personalize                                       (synthetic)
  python -m src.personalize --data data/shanghai --models models/shanghai --results results/shanghai

within_patient datasets: uses the patient's own train/val/test periods from the split.
by_patient datasets (short real records): only records of people the population model NEVER saw
(test people) are used; the first 50% of each record is the 'adaptation' period (75% fit / 25% val),
the last 50% is the evaluation period. No window overlaps between adaptation and evaluation.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .data import make_windows, normalize, prepare, select
from .engine import fit, get_device, make_consistency, predict
from .model import load_artifacts

MIN_FIT, MIN_VAL, MIN_EVAL = 20, 5, 10


def rmse(Y_true, Y_pred, h):
    return float(np.sqrt(np.mean((Y_true[:, h - 1] - Y_pred[:, h - 1]) ** 2)))


def windows_dict(arr, e_vec, pid, L, H):
    X, Y, t = make_windows(arr, L, H)
    return {"X": X, "Y": Y, "t": t, "pid": np.full(len(X), pid),
            "E": np.repeat(e_vec[None], len(X), axis=0)}


def patient_parts(data, sp, pid):
    """Return (fit, val, eval) raw window dicts for one record, or None if too short."""
    L, H = data["L"], data["H"]
    if data["schema"].get("split") != "by_patient":
        return tuple(select(sp[k], pid) for k in ("train", "val", "test"))
    arr = data["arrays"][pid]
    e_vec = data["ehr"].loc[pid].to_numpy(float)
    n = len(arr)
    a = int(n * 0.5)
    b = int(a * 0.75)
    parts = (windows_dict(arr[:b], e_vec, pid, L, H),
             windows_dict(arr[b:a], e_vec, pid, L, H),
             windows_dict(arr[a:], e_vec, pid, L, H))
    if len(parts[0]["X"]) < MIN_FIT or len(parts[1]["X"]) < MIN_VAL or len(parts[2]["X"]) < MIN_EVAL:
        return None
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = get_device()
    pop_model, meta = load_artifacts(args.models, device=device)
    data = prepare(args.data, seed=meta.get("seed", 0))
    sc, sp = data["scaler"], data["splits"]
    cols = meta["ts_cols"]
    by_patient = data["schema"].get("split") == "by_patient"

    def pack(split):
        X, E, Y, EV = normalize(split, sc)
        return np.ascontiguousarray(X[:, :, cols]), E, Y, EV

    if by_patient:
        pids = sorted(p for p, s in data["assignment"].items() if s == "test")
    else:
        pids = sorted(set(sp["train"]["pid"]))

    cons = make_consistency(sc, meta["ts_names"], cols) if meta.get("sign_prior") else None
    rows, skipped = [], 0
    for pid in pids:
        parts = patient_parts(data, sp, pid)
        if parts is None:
            skipped += 1
            continue
        tr, va, te = parts
        Xt, Et, _, _ = pack(te)
        pop_pred, _ = predict(pop_model, Xt, Et, sc, device)

        twin = copy.deepcopy(pop_model)
        fit(twin, pack(tr), pack(va), epochs=args.epochs, lr=args.lr, patience=4,
            device=device, freeze_conv=True, verbose=False, consistency=cons)
        twin_pred, _ = predict(twin, Xt, Et, sc, device)

        rows.append({"patient_id": int(pid), "n_eval_windows": len(te["X"]),
                     "RMSE60_population": rmse(te["Y"], pop_pred, 4),
                     "RMSE60_personal": rmse(te["Y"], twin_pred, 4),
                     "RMSE120_population": rmse(te["Y"], pop_pred, 8),
                     "RMSE120_personal": rmse(te["Y"], twin_pred, 8)})
        print(f"record {pid:>4}  RMSE60 population {rows[-1]['RMSE60_population']:.1f} "
              f"-> personal {rows[-1]['RMSE60_personal']:.1f}")

    if not rows:
        print("No record is long enough for personalization (need >= "
              f"{MIN_FIT + MIN_VAL + MIN_EVAL} windows).")
        return
    df = pd.DataFrame(rows)
    Path(args.results).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.results) / "personalization.csv", index=False)
    better = (df.RMSE60_personal < df.RMSE60_population).mean()
    print(f"\nRecords used: {len(df)}  (skipped as too short: {skipped})")
    print(f"Mean RMSE60: population {df.RMSE60_population.mean():.2f}  ->  personalized "
          f"{df.RMSE60_personal.mean():.2f} mg/dL")
    print(f"Mean RMSE120: population {df.RMSE120_population.mean():.2f}  ->  personalized "
          f"{df.RMSE120_personal.mean():.2f} mg/dL")
    if len(df) >= 6:
        p = wilcoxon(df.RMSE60_population, df.RMSE60_personal).pvalue
        print(f"Personalization helped {better:.0%} of records (Wilcoxon p = {p:.4f})")
    else:
        print(f"Personalization helped {better:.0%} of records (too few records for a significance test)")


if __name__ == "__main__":
    main()
