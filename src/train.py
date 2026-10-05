"""Train and compare: persistence, ridge regression and four FusionNet ablations.

  python -m src.train --epochs 40

Ablation (the 2x2 table you show to the judges):
                         without EHR        with EHR
  CGM only               fusion_cgm         fusion_cgm_ehr
  CGM + logs/time        fusion_logs        fusion_full   <- saved for the dashboard
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .data import EHR_COLS, TS_COLS, normalize, prepare
from .engine import fit, get_device, predict
from .metrics import event_metrics, regression_metrics, scores_from_forecast
from .model import FusionNet, save_artifacts

CONFIGS = {
    "fusion_cgm":     {"ts_cols": [0],                "use_ehr": False},
    "fusion_logs":    {"ts_cols": [0, 1, 2, 3, 4, 5], "use_ehr": False},
    "fusion_cgm_ehr": {"ts_cols": [0],                "use_ehr": True},
    "fusion_full":    {"ts_cols": [0, 1, 2, 3, 4, 5], "use_ehr": True},
}


def pack(split, scaler, cfg):
    X, E, Y, EV = normalize(split, scaler)
    X = np.ascontiguousarray(X[:, :, cfg["ts_cols"]])
    if not cfg["use_ehr"]:
        E = np.zeros_like(E)
    return X, E, Y, EV


def report(name, split, Y_pred, scores):
    row = {"model": name}
    row.update(regression_metrics(split["Y"], Y_pred))
    row.update(event_metrics(split["Y"], scores, split["X"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = get_device()
    print("device:", device)

    data = prepare(args.data)
    sc, sp = data["scaler"], data["splits"]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print({k: len(v["X"]) for k, v in sp.items()}, "windows (train/val/test)")
    rows = []

    # --- baseline 1: persistence (future glucose = current glucose) ---
    pers = np.repeat(te["X"][:, -1:, 0], data["H"], axis=1)
    rows.append(report("persistence", te, pers, scores_from_forecast(pers)))

    # --- baseline 2: ridge regression on flattened window + EHR ---
    Xtr, Etr, Ytr, _ = normalize(tr, sc)
    Xte, Ete, _, _ = normalize(te, sc)
    ridge = Ridge(alpha=1.0).fit(np.hstack([Xtr.reshape(len(Xtr), -1), Etr]), Ytr)
    r_pred = ridge.predict(np.hstack([Xte.reshape(len(Xte), -1), Ete])) * sc["ts_std"][0] + sc["ts_mean"][0]
    rows.append(report("ridge", te, r_pred, scores_from_forecast(r_pred)))

    # --- FusionNet ablations ---
    for name, cfg in CONFIGS.items():
        print(f"\n== training {name} ==")
        model = FusionNet(n_ts_features=len(cfg["ts_cols"]), n_ehr_features=len(EHR_COLS),
                          n_horizons=data["H"])
        fit(model, pack(tr, sc, cfg), pack(va, sc, cfg), epochs=args.epochs, device=device)
        Xt, Et, _, _ = pack(te, sc, cfg)
        Yp, P = predict(model, Xt, Et, sc, device)
        rows.append(report(name, te, Yp, P))
        if name == "fusion_full":
            meta = {"scaler": sc, "ts_cols": cfg["ts_cols"], "ts_names": TS_COLS,
                    "ehr_names": EHR_COLS, "L": data["L"], "H": data["H"]}
            save_artifacts(model, meta, args.models, name)
            print("saved ->", Path(args.models) / f"{name}.pt")

    df = pd.DataFrame(rows).set_index("model")
    Path(args.results).mkdir(exist_ok=True)
    df.to_csv(Path(args.results) / "metrics.csv")
    pd.set_option("display.width", 200)
    print("\n=== TEST RESULTS (mg/dL) ===")
    print(df[["RMSE_30", "RMSE_60", "RMSE_120", "MAE_60"]].round(2))
    print("\n=== EARLY WARNING (glucose currently in range) ===")
    print(df[["AUROC_hypo", "Sens@90Spec_hypo", "AUROC_hyper", "Sens@90Spec_hyper"]].round(3))


if __name__ == "__main__":
    main()
