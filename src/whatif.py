"""What-if simulation on the virtual patient.

Take the patient's current 6-hour window, apply a hypothetical intervention at
'now' (eat X g carbs, inject Y units insulin, walk Z steps) and ask the model for
the 2-hour glucose forecast with uncertainty and hypo/hyper risk.

NOTE: this is a LEARNED counterfactual (the network learned the effect of meals /
insulin / activity from data). It is a research prototype, not medical advice.
"""
import numpy as np
import torch

from .model import mc_predict

CARBS, STEPS, INSULIN = 1, 2, 3          # column indices in TS_COLS


def scenario(model, meta, window, ehr_vec, carbs=0.0, insulin=0.0, steps=0.0, n_samples=30):
    """window: raw (L, 6) array; ehr_vec: raw (8,) array. Returns forecast dict in mg/dL."""
    sc = meta["scaler"]
    w = window.copy()
    w[-1, CARBS] += carbs
    w[-1, INSULIN] += insulin
    w[-1, STEPS] += steps

    x = (w - np.array(sc["ts_mean"])) / np.array(sc["ts_std"])
    e = (ehr_vec - np.array(sc["ehr_mean"])) / np.array(sc["ehr_std"])
    x = torch.tensor(x[None][:, :, meta["ts_cols"]], dtype=torch.float32)
    e = torch.tensor(e[None], dtype=torch.float32)

    gm, gs = sc["ts_mean"][0], sc["ts_std"][0]
    mean, std, p = mc_predict(model, x, e, n_samples)
    return {
        "minutes": np.arange(1, meta["H"] + 1) * 15,
        "mean": mean[0].numpy() * gs + gm,
        "std": std[0].numpy() * gs,
        "p_hypo": float(p[0, 0]),
        "p_hyper": float(p[0, 1]),
    }
