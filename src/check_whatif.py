"""Sanity check of the what-if simulator on unseen (test) patients.

Adds 50 g carbs / 4 U fast insulin / 1000 steps at 'now' to random test windows and reports the average
change of the forecast. Physiologically: carbs should raise glucose, insulin and activity should lower it.

    python -m src.check_whatif --data data/shanghai --models models/shanghai
"""
import argparse

import numpy as np

from .data import prepare
from .model import load_artifacts
from .whatif import col_index, scenario


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic")
    ap.add_argument("--models", default="models")
    ap.add_argument("--per_record", type=int, default=25)
    args = ap.parse_args()

    model, meta = load_artifacts(args.models)
    d = prepare(args.data, seed=meta.get("seed", 0))
    L, H = meta["L"], meta["H"]
    rng = np.random.default_rng(0)
    pids = [p for p, s in d["assignment"].items() if s in ("test", "all")]
    tests = {"+50 g carbs": dict(carbs=50), "+4 U fast insulin": dict(insulin=4), "+1000 steps": dict(steps=1000)}
    present = {"+50 g carbs": ("carbs",), "+4 U fast insulin": ("insulin", "insulin_fast"), "+1000 steps": ("steps",)}
    res = {k: [] for k in tests}
    for pid in pids:
        arr = d["arrays"][pid]
        e = d["ehr"].loc[pid].to_numpy(float)
        if len(arr) < L + H + 2:
            continue
        for t in rng.integers(L, len(arr) - H - 1, args.per_record):
            w = arr[t - L + 1:t + 1].copy()
            if np.isnan(w[:, 0]).any():
                continue
            base = scenario(model, meta, w, e, n_samples=1)["mean"]
            for k, kw in tests.items():
                if col_index(meta, *present[k]) is not None:
                    res[k].append(scenario(model, meta, w, e, n_samples=1, **kw)["mean"] - base)
    for k, v in res.items():
        if not v:
            continue
        v = np.array(v)
        want = "up" if "carbs" in k else "down"
        ok = (v[:, 3] > 0).mean() if want == "up" else (v[:, 3] < 0).mean()
        print(f"{k}: mean change at +30/60/120 min = {v[:, 1].mean():+.1f} / {v[:, 3].mean():+.1f} / "
              f"{v[:, 7].mean():+.1f} mg/dL; expected direction ({want}) in {ok:.0%} of {len(v)} windows")


if __name__ == "__main__":
    main()
