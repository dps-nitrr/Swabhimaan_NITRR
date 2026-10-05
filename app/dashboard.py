"""Doctor-facing dashboard for the virtual patient.   Run:  streamlit run app/dashboard.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import EHR_COLS, HYPER, HYPO, load_raw, patient_arrays
from src.model import load_artifacts
from src.whatif import scenario

DATA_DIR = ROOT / "data" / "synthetic"
MODEL_DIR = ROOT / "models"
SPD = 96

st.set_page_config(page_title="GlucoTwin India", page_icon="🩺", layout="wide")


@st.cache_resource
def load_all():
    model, meta = load_artifacts(str(MODEL_DIR))
    cgm, ehr = load_raw(DATA_DIR)
    arrays = patient_arrays(cgm)
    start = cgm.groupby("patient_id")["timestamp"].min()
    return model, meta, arrays, ehr.set_index("patient_id"), start


def ffill(w):
    w = w.copy()
    g = pd.Series(w[:, 0]).ffill().bfill().to_numpy()
    w[:, 0] = g
    return w


@st.cache_data(show_spinner=False)
def triage_table(_model, _meta, _arrays, _ehr):
    """Risk list across ALL patients at their most recent moment."""
    L, H = _meta["L"], _meta["H"]
    rows = []
    for pid, arr in _arrays.items():
        t = len(arr) - H - 1
        w = ffill(arr[t - L + 1:t + 1])
        r = scenario(_model, _meta, w, _ehr.loc[pid, EHR_COLS].to_numpy(float), n_samples=10)
        rows.append({"patient": pid, "HbA1c": _ehr.loc[pid, "hba1c"],
                     "now (mg/dL)": round(w[-1, 0]), "min next 2h": round(r["mean"].min()),
                     "max next 2h": round(r["mean"].max()),
                     "P(hypo)": round(r["p_hypo"], 2), "P(hyper)": round(r["p_hyper"], 2)})
    df = pd.DataFrame(rows)
    df["risk"] = df[["P(hypo)", "P(hyper)"]].max(axis=1)
    return df.sort_values("risk", ascending=False).drop(columns="risk").reset_index(drop=True)


if not (MODEL_DIR / "fusion_full.pt").exists() or not (DATA_DIR / "cgm.csv").exists():
    st.error("Data or model missing. Run:  `python -m src.synth`  then  `python -m src.train`")
    st.stop()

model, meta, arrays, ehr, start = load_all()
L, H = meta["L"], meta["H"]

st.title("🩺 GlucoTwin India: virtual patient dashboard")
st.caption("Research prototype on SYNTHETIC data. Not a medical device and not medical advice.")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Virtual patient")
    pid = st.selectbox("Patient ID", sorted(arrays.keys()), index=0)
    arr = arrays[pid]
    n_days = len(arr) // SPD
    day = st.slider("Day (last days of record)", n_days - 4, n_days - 1, n_days - 2)
    times = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
    tod = st.select_slider("Time of day", options=times, value="12:00")
    t = day * SPD + times.index(tod)
    t = int(np.clip(t, L - 1, len(arr) - H - 1))

    st.header("What-if scenario")
    carbs = st.slider("Eat now: carbohydrates (g)", 0, 120, 0, 5)
    insulin = st.slider("Insulin bolus now (units)", 0.0, 10.0, 0.0, 0.5)
    walk = st.slider("Walk in next 15 min (steps)", 0, 2500, 0, 100)

window = ffill(arr[t - L + 1:t + 1])
e_vec = ehr.loc[pid, EHR_COLS].to_numpy(float)
base = scenario(model, meta, window, e_vec)
what = scenario(model, meta, window, e_vec, carbs=carbs, insulin=insulin, steps=walk)
changed = carbs > 0 or insulin > 0 or walk > 0

# ---------------- main view ----------------
now_ts = start[pid] + pd.Timedelta(minutes=15 * t)
hist_t = [now_ts - pd.Timedelta(minutes=15 * (L - 1 - i)) for i in range(L)]
fut_t = [now_ts + pd.Timedelta(minutes=int(m)) for m in base["minutes"]]
actual = arr[t + 1:t + 1 + H, 0]

left, right = st.columns([3, 1])
with left:
    fig = go.Figure()
    fig.add_hrect(y0=HYPO, y1=HYPER, fillcolor="green", opacity=0.07, line_width=0)
    fig.add_hline(y=HYPO, line_dash="dot", line_color="#d62728")
    fig.add_hline(y=HYPER, line_dash="dot", line_color="#ff7f0e")
    fig.add_trace(go.Scatter(x=hist_t, y=window[:, 0], name="CGM (past 6 h)", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=fut_t, y=actual, name="What actually happened (hidden from model)",
                             line=dict(color="gray", dash="dot")))
    for res, name, color in [(base, "Forecast: no change", "#2ca02c"), (what, "Forecast: what-if", "#9467bd")]:
        if res is what and not changed:
            continue
        up, lo = res["mean"] + 1.64 * res["std"], res["mean"] - 1.64 * res["std"]
        fig.add_trace(go.Scatter(x=fut_t + fut_t[::-1], y=list(up) + list(lo[::-1]), fill="toself",
                                 fillcolor=color, opacity=0.15, line_width=0, hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=fut_t, y=res["mean"], name=name, line=dict(color=color, width=3)))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Glucose (mg/dL)",
                      legend=dict(orientation="h", y=-0.2), title=f"Patient {pid}: {now_ts:%d %b %Y, %H:%M}")
    st.plotly_chart(fig, width="stretch")

    shown = what if changed else base
    if shown["p_hypo"] > 0.5:
        st.error(f"⚠️ High risk of LOW glucose (<{HYPO:.0f}) in the next 2 hours "
                 f"(P = {shown['p_hypo']:.0%}). Consider reducing insulin / having carbohydrates.")
    elif shown["p_hyper"] > 0.6:
        st.warning(f"⚠️ High risk of HIGH glucose (>{HYPER:.0f}) in the next 2 hours "
                   f"(P = {shown['p_hyper']:.0%}).")
    else:
        st.success("No high-risk event predicted for the next 2 hours.")

with right:
    st.subheader("Patient card")
    r = ehr.loc[pid]
    st.write(f"**Age** {r.age:.0f}   |   **Sex** {'F' if r.sex else 'M'}")
    st.write(f"**BMI** {r.bmi}   |   **HbA1c** {r.hba1c}%")
    st.write(f"**Diabetes** {r.diabetes_years} yrs   |   **BP** {r.sbp:.0f} mmHg")
    st.write(f"**Insulin** {'yes' if r.on_insulin else 'no'}   |   **Metformin** {'yes' if r.on_metformin else 'no'}")
    st.subheader("Risk (next 2 h)")
    st.metric("Low glucose", f"{shown['p_hypo']:.0%}",
              delta=f"{(what['p_hypo'] - base['p_hypo']):+.0%} vs no change" if changed else None, delta_color="inverse")
    st.metric("High glucose", f"{shown['p_hyper']:.0%}",
              delta=f"{(what['p_hyper'] - base['p_hyper']):+.0%} vs no change" if changed else None, delta_color="inverse")
    st.metric("Predicted at +60 min", f"{shown['mean'][3]:.0f} mg/dL",
              delta=f"{(what['mean'][3] - base['mean'][3]):+.0f}" if changed else None, delta_color="off")

st.divider()
st.subheader("Clinic triage: all patients ranked by predicted risk (latest moment)")
st.dataframe(triage_table(model, meta, arrays, ehr), width="stretch", hide_index=True)
