"""Doctor-facing dashboard for the virtual patient.   Run:  streamlit run app/dashboard.py

Works with any dataset that has a trained model (synthetic or real Shanghai T2DM)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data import HYPER, HYPO, load_raw, patient_frames, prepare
from src.model import load_artifacts
from src.whatif import col_index, scenario

DATASETS = {
    "Real data: Shanghai T2DM": (ROOT / "data" / "shanghai", ROOT / "models" / "shanghai"),
    "Synthetic data": (ROOT / "data" / "synthetic", ROOT / "models"),
}
NICE = {"age": "Age (yr)", "female": "Female (1=yes)", "sex": "Sex (1=F)", "bmi": "BMI", "hba1c": "HbA1c (%)",
        "diabetes_years": "Diabetes duration (yr)", "on_insulin": "On insulin", "on_metformin": "On metformin",
        "sbp": "Systolic BP", "fpg": "Fasting glucose", "ppg_2h": "2h post-meal glucose",
        "cpep_fasting": "Fasting C-peptide", "cpep_2h": "2h C-peptide", "egfr": "eGFR"}

st.set_page_config(page_title="GlucoTwin India", page_icon="🩺", layout="wide")

available = {k: v for k, v in DATASETS.items() if (v[1] / "fusion_full.pt").exists() and (v[0] / "cgm.csv").exists()}
if not available:
    st.error("Data or model missing. Run:  `python -m src.synth`  then  `python -m src.train`")
    st.stop()


@st.cache_resource
def load_all(name):
    data_dir, model_dir = available[name]
    model, meta = load_artifacts(str(model_dir))
    data = prepare(str(data_dir), L=meta["L"], H=meta["H"], seed=meta.get("seed", 0))
    cgm, _ = load_raw(data_dir)
    frames = patient_frames(cgm, data["schema"])
    return model, meta, data, frames


def ffill(w):
    w = w.copy()
    w[:, 0] = pd.Series(w[:, 0]).ffill().bfill().to_numpy()
    return w


@st.cache_data(show_spinner=False)
def triage_table(name):
    """Risk list across ALL records at their most recent moment."""
    model, meta, data, frames = load_all(name)
    L, H = meta["L"], meta["H"]
    rows = []
    for pid, arr in data["arrays"].items():
        t = len(arr) - H - 1
        if t < L - 1:
            continue
        w = ffill(arr[t - L + 1:t + 1])
        if np.isnan(w[:, 0]).any():
            continue
        r = scenario(model, meta, w, data["ehr"].loc[pid].to_numpy(float), n_samples=10)
        row = {"record": pid, "now (mg/dL)": round(w[-1, 0]), "min next 2h": round(r["mean"].min()),
               "max next 2h": round(r["mean"].max()),
               "P(hypo)": round(r["p_hypo"], 2), "P(hyper)": round(r["p_hyper"], 2)}
        if "hba1c" in data["ehr"].columns:
            row = {"record": pid, "HbA1c": round(data["ehr"].loc[pid, "hba1c"], 1), **{k: v for k, v in row.items() if k != "record"}}
        rows.append(row)
    df = pd.DataFrame(rows)
    df["risk"] = df[["P(hypo)", "P(hyper)"]].max(axis=1)
    return df.sort_values("risk", ascending=False).drop(columns="risk").reset_index(drop=True)


st.title("🩺 GlucoTwin India: virtual patient dashboard")

with st.sidebar:
    ds_name = st.radio("Dataset", list(available.keys()))
model, meta, data, frames = load_all(ds_name)
L, H = meta["L"], meta["H"]
schema = data["schema"]
by_patient = schema.get("split") == "by_patient"
if by_patient:
    st.caption("Research prototype trained on the public Shanghai T2DM dataset (Chinese hospital inpatients, "
               "used as a proxy: no public Indian CGM dataset exists). Not a medical device and not medical advice.")
else:
    st.caption("Research prototype on SYNTHETIC data. Not a medical device and not medical advice.")

usable = [p for p, a in data["arrays"].items() if len(a) >= L + H + 1]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Virtual patient")

    def label(p):
        tag = f" ({data['assignment'][p]})" if by_patient else ""
        return f"Record {p}{tag}"

    options = sorted(usable)
    first_test = next((i for i, p in enumerate(options) if by_patient and data["assignment"][p] == "test"), 0)
    pid = st.selectbox("Record", options, index=first_test, format_func=label)
    if by_patient:
        st.caption("'test' = person never seen in training (honest evaluation); 'train' = seen.")
    arr = data["arrays"][pid]
    idx = frames[pid].index
    lo_t, hi_t = L - 1, len(arr) - H - 1
    t = st.slider("Moment in the record", lo_t, hi_t, hi_t - (hi_t - lo_t) // 3,
                  help="Each step = 15 min. The model sees the 6 h before this moment.")
    st.caption(f"Selected time: {idx[t]:%d %b %Y, %H:%M}")

    st.header("What-if scenario")
    carbs = st.slider("Eat now: carbohydrates (g)", 0, 120, 0, 5)
    ins_idx = col_index(meta, "insulin", "insulin_fast")
    insulin = st.slider("Fast insulin now (units)", 0.0, 10.0, 0.0, 0.5) if ins_idx is not None else 0.0
    walk = st.slider("Walk in next 15 min (steps)", 0, 2500, 0, 100) if col_index(meta, "steps") is not None else 0

window = ffill(arr[t - L + 1:t + 1])
if np.isnan(window[:, 0]).any():
    st.warning("This moment has no recent CGM readings. Pick another moment.")
    st.stop()
e_vec = data["ehr"].loc[pid].to_numpy(float)
base = scenario(model, meta, window, e_vec)
what = scenario(model, meta, window, e_vec, carbs=carbs, insulin=insulin, steps=walk)
changed = carbs > 0 or insulin > 0 or walk > 0

# ---------------- main view ----------------
now_ts = idx[t]
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
                      legend=dict(orientation="h", y=-0.2), title=f"Record {pid}: {now_ts:%d %b %Y, %H:%M}")
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
    r = data["ehr"].loc[pid]
    for c in schema["ehr_cols"]:
        v = r[c]
        shown_v = "yes" if c in ("on_insulin", "on_metformin", "female", "sex") and v >= 0.5 else \
                  "no" if c in ("on_insulin", "on_metformin", "female", "sex") else f"{v:.1f}"
        st.write(f"**{NICE.get(c, c)}**: {shown_v}")
    st.subheader("Risk (next 2 h)")
    st.metric("Low glucose", f"{shown['p_hypo']:.0%}",
              delta=f"{(what['p_hypo'] - base['p_hypo']):+.0%} vs no change" if changed else None, delta_color="inverse")
    st.metric("High glucose", f"{shown['p_hyper']:.0%}",
              delta=f"{(what['p_hyper'] - base['p_hyper']):+.0%} vs no change" if changed else None, delta_color="inverse")
    st.metric("Predicted at +60 min", f"{shown['mean'][3]:.0f} mg/dL",
              delta=f"{(what['mean'][3] - base['mean'][3]):+.0f}" if changed else None, delta_color="off")

st.divider()
st.subheader("Clinic triage: all records ranked by predicted risk (latest moment)")
st.dataframe(triage_table(ds_name), width="stretch", hide_index=True)
