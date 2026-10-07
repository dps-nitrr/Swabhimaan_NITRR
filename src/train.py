"""Train and compare: persistence, ridge regression and four FusionNet ablations.

  python -m src.train --epochs 40                                   (synthetic data)
  python -m src.train --data data/shanghai --models models/shanghai --results results/shanghai

Ablation (the 2x2 table you show to the judges):
                         without EHR        with EHR
  CGM only               fusion_cgm         fusion_cgm_ehr
  CGM + logs/time        fusion_logs        fusion_full   <- saved for the dashboard
The columns used are read from <data>/schema.json, so the same code runs on any dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from .data import normalize, prepare
from .engine import fit, get_device, predict
from .metrics import bootstrap_rmse_ci, event_metrics, regression_metrics, scores_from_forecast
from .model import FusionNet, save_artifacts


def make_configs(n_ts):
    allc = list(range(n_ts))
    return {
        "fusion_cgm":     {"ts_cols": [0],  "use_ehr": False},
        "fusion_logs":    {"ts_cols": allc, "use_ehr": False},
        "fusion_cgm_ehr": {"ts_cols": [0],  "use_ehr": True},
        "fusion_full":    {"ts_cols": allc, "use_ehr": True},
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
    lo, hi = bootstrap_rmse_ci(split["Y"], Y_pred, split["pid"], h=4)
    row["RMSE_60_lo"], row["RMSE_60_hi"] = lo, hi
    row.update(event_metrics(split["Y"], scores, split["X"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", default="models")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print("device:", device)

    data = prepare(args.data, seed=args.seed)
    schema = data["schema"]
    sc, sp = data["scaler"], data["splits"]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    n_ts, n_ehr = len(schema["ts_cols"]), len(schema["ehr_cols"])
    ehr_noise = 0.1 if schema.get("split") == "by_patient" else 0.0
    print(f"dataset: {schema['name']}   split: {schema.get('split')}   ehr_noise: {ehr_noise}")
    print({k: len(v["X"]) for k, v in sp.items()}, "windows (train/val/test)")
    print({k: len(set(v["pid"])) for k, v in sp.items()}, "records (train/val/test)")
    if schema.get("split") == "by_patient":
        people = {k: len({int(data["group_of"][p]) for p in set(v["pid"])}) for k, v in sp.items()}
        print(people, "distinct people (train/val/test)")
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
    for name, cfg in make_configs(n_ts).items():
        print(f"\n== training {name} ==")
        model = FusionNet(n_ts_features=len(cfg["ts_cols"]), n_ehr_features=n_ehr,
                          n_horizons=data["H"], ehr_noise=ehr_noise if cfg["use_ehr"] else 0.0)
        fit(model, pack(tr, sc, cfg), pack(va, sc, cfg), epochs=args.epochs, device=device)
        Xt, Et, _, _ = pack(te, sc, cfg)
        Yp, P = predict(model, Xt, Et, sc, device)
        rows.append(report(name, te, Yp, P))
        if name == "fusion_full":
            meta = {"scaler": sc, "ts_cols": cfg["ts_cols"], "ts_names": schema["ts_cols"],
                    "ehr_names": schema["ehr_cols"], "L": data["L"], "H": data["H"],
                    "data_dir": args.data, "seed": args.seed, "schema": schema}
            save_artifacts(model, meta, args.models, name)
            print("saved ->", Path(args.models) / f"{name}.pt")

    df = pd.DataFrame(rows).set_index("model")
    Path(args.results).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.results) / "metrics.csv")
    pd.set_option("display.width", 200)
    print("\n=== TEST RESULTS (mg/dL; 95% CI of RMSE_60 by bootstrap over records) ===")
    print(df[["RMSE_30", "RMSE_60", "RMSE_60_lo", "RMSE_60_hi", "RMSE_120", "MAE_60"]].round(2))
    print("\n=== EARLY WARNING (glucose currently in range) ===")
    print(df[["AUROC_hypo", "Sens@90Spec_hypo", "hypo_positives",
              "AUROC_hyper", "Sens@90Spec_hyper", "hyper_positives"]].round(3))


if __name__ == "__main__":
    main()
