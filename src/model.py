"""FusionNet: fuses dynamic signals (CGM, carbs, steps, insulin, time of day)
with static EHR features.

  Branch 1 (dynamic): Conv1d feature extractor -> LSTM over the last 6 hours
  Branch 2 (static):  small MLP over EHR features (age, BMI, HbA1c, ...)
  Heads: glucose at the next 8 steps (15..120 min) + [hypo, hyper] event logits
"""
from pathlib import Path

import torch
import torch.nn as nn

from .data import load_json, save_json


class FusionNet(nn.Module):
    def __init__(self, n_ts_features=6, n_ehr_features=8, n_horizons=8,
                 hidden=64, dropout=0.2, ehr_noise=0.0):
        super().__init__()
        # Gaussian noise added to the (standardized) EHR vector during TRAINING only. With few patients
        # the network could otherwise use the EHR vector as a patient ID and memorize that patient.
        self.ehr_noise = ehr_noise
        self.conv = nn.Sequential(
            nn.Conv1d(n_ts_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.ehr = nn.Sequential(
            nn.Linear(n_ehr_features, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + 32, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.glucose_head = nn.Linear(64, n_horizons)
        self.event_head = nn.Linear(64, 2)

    def forward(self, x_ts, x_ehr):
        # x_ts: (batch, time, n_ts_features), x_ehr: (batch, n_ehr_features)
        if self.training and self.ehr_noise > 0:
            x_ehr = x_ehr + self.ehr_noise * torch.randn_like(x_ehr)
        h = self.conv(x_ts.transpose(1, 2)).transpose(1, 2)
        _, (h_n, _) = self.lstm(h)
        z = self.fuse(torch.cat([h_n[-1], self.ehr(x_ehr)], dim=1))
        return self.glucose_head(z), self.event_head(z)


def compute_loss(glu_pred, evt_logits, glu_true, evt_true, w_event=0.5):
    mse = nn.functional.mse_loss(glu_pred, glu_true)
    bce = nn.functional.binary_cross_entropy_with_logits(evt_logits, evt_true)
    return mse + w_event * bce


@torch.no_grad()
def mc_predict(model, x_ts, x_ehr, n_samples=30):
    """Monte-Carlo Dropout: keep dropout ON at inference and sample repeatedly.

    Returns (glucose_mean, glucose_std, event_prob_mean); the first two are in
    scaled units (convert to mg/dL with the scaler).
    """
    model.eval()
    for m in model.modules():              # enable dropout only (BatchNorm stays in eval)
        if isinstance(m, nn.Dropout):
            m.train()
    glu, evt = [], []
    for _ in range(n_samples):
        g, e = model(x_ts, x_ehr)
        glu.append(g)
        evt.append(torch.sigmoid(e))
    glu, evt = torch.stack(glu), torch.stack(evt)
    model.eval()
    return glu.mean(0), glu.std(0), evt.mean(0)


def save_artifacts(model, meta, models_dir="models", name="fusion_full"):
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / f"{name}.pt")
    save_json(meta, d / f"{name}.json")


def load_artifacts(models_dir="models", name="fusion_full", device="cpu"):
    d = Path(models_dir)
    meta = load_json(d / f"{name}.json")
    model = FusionNet(n_ts_features=len(meta["ts_cols"]), n_ehr_features=len(meta["ehr_names"]),
                      n_horizons=meta["H"])
    model.load_state_dict(torch.load(d / f"{name}.pt", map_location=device))
    model.to(device).eval()
    return model, meta


if __name__ == "__main__":
    net = FusionNet()
    xt, xe = torch.randn(4, 24, 6), torch.randn(4, 8)
    g, e = net(xt, xe)
    print("glucose:", tuple(g.shape), "events:", tuple(e.shape))
    mean, std, p = mc_predict(net, xt, xe)
    print("uncertainty:", tuple(std.shape), "event probs:", tuple(p.shape))
