"""Personalization: turn the population model into a per-patient 'twin'.

For every patient we copy the population model, freeze the convolutional feature
extractor and fine-tune the rest on THAT patient's own training windows only.
Then we compare population vs personalized error on that patient's test period.

  python -m src.personalize
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .data import normalize, prepare, select
from .engine import fit, get_device, predict
from .model import load_artifacts


def rmse(Y_true, Y_pred, h):
    return float(np.sqrt(np.mean((Y_true[:, h - 1] - Y_pred[:, h - 1]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = get_device()
    data = prepare(args.data)
    sc, sp = data["scaler"], data["splits"]
    pop_model, meta = load_artifacts(args.models, device=device)
    cols = meta["ts_cols"]

    def pack(split):
        X, E, Y, EV = normalize(split, sc)
        return np.ascontiguousarray(X[:, :, cols]), E, Y, EV

    rows = []
    for pid in sorted(set(sp["train"]["pid"])):
        tr, va, te = (select(sp[k], pid) for k in ("train", "val", "test"))
        Xt, Et, _, _ = pack(te)
        pop_pred, _ = predict(pop_model, Xt, Et, sc, device)

        twin = copy.deepcopy(pop_model)
        fit(twin, pack(tr), pack(va), epochs=args.epochs, lr=args.lr, patience=4,
            device=device, freeze_conv=True, verbose=False)
        twin_pred, _ = predict(twin, Xt, Et, sc, device)

        rows.append({"patient_id": int(pid),
                     "RMSE60_population": rmse(te["Y"], pop_pred, 4),
                     "RMSE60_personal": rmse(te["Y"], twin_pred, 4),
                     "RMSE120_population": rmse(te["Y"], pop_pred, 8),
                     "RMSE120_personal": rmse(te["Y"], twin_pred, 8)})
        print(f"patient {pid:02d}  RMSE60 population {rows[-1]['RMSE60_population']:.1f} "
              f"-> personal {rows[-1]['RMSE60_personal']:.1f}")

    df = pd.DataFrame(rows)
    Path(args.results).mkdir(exist_ok=True)
    df.to_csv(Path(args.results) / "personalization.csv", index=False)
    better = (df.RMSE60_personal < df.RMSE60_population).mean()
    p = wilcoxon(df.RMSE60_population, df.RMSE60_personal).pvalue
    print(f"\nMean RMSE60: population {df.RMSE60_population.mean():.2f}  ->  personalized "
          f"{df.RMSE60_personal.mean():.2f} mg/dL")
    print(f"Mean RMSE120: population {df.RMSE120_population.mean():.2f}  ->  personalized "
          f"{df.RMSE120_personal.mean():.2f} mg/dL")
    print(f"Personalization helped {better:.0%} of patients (Wilcoxon p = {p:.4f})")


if __name__ == "__main__":
    main()
