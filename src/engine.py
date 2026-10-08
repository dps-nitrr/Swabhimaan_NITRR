"""Training helpers shared by train.py and personalize.py."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import compute_loss


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(X, E, Y, EV, bs=256, shuffle=False):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(E),
                       torch.from_numpy(Y), torch.from_numpy(EV))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def make_consistency(scaler, ts_names, used_cols, carbs_g=50.0, insulin_u=4.0, steps=1000.0):
    """Physiological SIGN prior used as a training regulariser.

    Returns a list of (position in the model's input columns, extra amount in scaled units, direction):
      direction -1: glucose must NOT fall when this input is added   (carbs)
      direction +1: glucose must NOT rise when this input is added   (insulin, steps)
    Only the sign of the effect is constrained; its size is learned from data.
    """
    out = []
    for names, amount, direction in ((("carbs",), carbs_g, -1),
                                     (("insulin", "insulin_fast"), insulin_u, +1),
                                     (("steps",), steps, +1)):
        full = next((ts_names.index(n) for n in names if n in ts_names), None)
        if full is None or full not in used_cols:
            continue
        out.append((used_cols.index(full), amount / scaler["ts_std"][full], direction))
    return out


def train_epoch(model, loader, opt, device, freeze_conv=False, consistency=None, w_cons=1.0):
    model.train()
    if freeze_conv:
        model.conv.eval()          # keep BatchNorm statistics fixed too
    total = 0.0
    for x, e, y, ev in loader:
        x, e, y, ev = x.to(device), e.to(device), y.to(device), ev.to(device)
        opt.zero_grad()
        g, logit = model(x, e)
        loss = compute_loss(g, logit, y, ev)
        for j, delta, direction in (consistency or []):
            x2 = x.clone()
            x2[:, -1, j] += delta                       # "add this carbs / insulin / steps now"
            g2, _ = model(x2, e)
            loss = loss + w_cons * torch.relu(direction * (g2 - g)).mean()
        loss.backward()
        opt.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total = 0.0
    for x, e, y, ev in loader:
        x, e, y, ev = x.to(device), e.to(device), y.to(device), ev.to(device)
        g, logit = model(x, e)
        total += compute_loss(g, logit, y, ev).item() * len(x)
    return total / len(loader.dataset)


def fit(model, train, val, epochs=40, lr=1e-3, patience=6, bs=256,
        device="cpu", freeze_conv=False, verbose=True, consistency=None):
    """train / val are tuples (X, E, Y, EV). Early stopping on validation loss."""
    model.to(device)
    if freeze_conv:
        for p in model.conv.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    tr, va = make_loader(*train, bs=bs, shuffle=True), make_loader(*val, bs=2048)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        tl = train_epoch(model, tr, opt, device, freeze_conv, consistency)
        vl = eval_loss(model, va, device)
        if verbose:
            print(f"    epoch {ep + 1:02d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, E, scaler, device="cpu", bs=4096):
    """Returns (glucose forecast in mg/dL (N, H), event probabilities (N, 2))."""
    model.eval().to(device)
    gm, gs = scaler["ts_mean"][0], scaler["ts_std"][0]
    G, P = [], []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        e = torch.from_numpy(E[i:i + bs]).to(device)
        g, logit = model(x, e)
        G.append(g.cpu().numpy())
        P.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(G) * gs + gm, np.concatenate(P)
