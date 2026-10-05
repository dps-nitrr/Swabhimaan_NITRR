"""Synthetic Type-2-diabetes cohort generator.

Creates
  1. a static EHR table (age, BMI, HbA1c, ...), and
  2. a 15-minute time-series table: glucose (CGM), meal carbs, steps, insulin.

Glucose comes from a small mechanistic glucose-insulin simulator whose
parameters depend on the EHR (HbA1c -> baseline glucose, BMI/duration/metformin
-> insulin sensitivity). So the EHR carries real signal, which lets us show the
benefit of fusing static + dynamic data.

IMPORTANT: this is SYNTHETIC data for a proof-of-concept. Say so in your README.

Run:  python -m src.synth --patients 40 --days 30 --out data/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STEP_MIN = 15                     # sampling interval (minutes)
SUB = 3                           # simulation sub-steps per sample
DT = STEP_MIN / SUB               # 5 min
SPD = 24 * 60 // STEP_MIN         # samples per day = 96

KE = 0.04                         # gastric emptying rate (1/min)
KA_BASE = 0.03                    # intestinal absorption rate (1/min)
KI = 1 / 50                       # subcutaneous insulin rate (1/min)
TAU_EX = 45.0                     # exercise effect time constant (min)
A_GAIN = 3.3                      # mg/dL rise per gram of absorbed carbs
EX_GAIN = 0.5                     # glucose lowering by activity (mg/dL/min)
INS_COVER = 2.0                   # insulin dose scale (tuned so insulin users get some lows)

# (hour, jitter_h, mean_carbs_g, sd_carbs_g, prob_walk_after)
MEALS = [
    (8.0, 0.75, 45, 12, 0.2),     # breakfast
    (13.0, 0.75, 70, 18, 0.4),    # lunch
    (20.0, 1.0, 75, 20, 0.5),     # dinner
]
SNACK = (16.5, 1.0, 25, 10, 0.1)


def make_ehr(n, rng):
    dur = rng.uniform(1, 22, n)
    p_ins = np.clip(0.3 + 0.02 * (dur - 10), 0.05, 0.9)
    return pd.DataFrame({
        "patient_id": np.arange(n),
        "age": rng.uniform(35, 78, n).round(0),
        "sex": rng.integers(0, 2, n),
        "bmi": rng.normal(28.5, 4.5, n).clip(20, 42).round(1),
        "hba1c": rng.normal(8.0, 1.2, n).clip(6.0, 12.0).round(1),
        "diabetes_years": dur.round(1),
        "on_insulin": (rng.random(n) < p_ins).astype(int),
        "on_metformin": (rng.random(n) < 0.8).astype(int),
        "sbp": rng.normal(132, 14, n).clip(95, 180).round(0),
    })


def simulate_patient(e, days, rng):
    n = days * SPD
    on_ins = bool(e["on_insulin"])

    # --- hidden physiology derived from the EHR ---
    gb = 0.80 * (28.7 * e["hba1c"] - 46.7) + rng.normal(0, 4)       # baseline glucose
    si = float(np.clip(1.0 - 0.03 * (e["bmi"] - 24) - 0.015 * e["diabetes_years"]
                       + 0.12 * e["on_metformin"] + rng.normal(0, 0.05), 0.3, 1.2))
    kc = 0.006 + 0.018 * si                                          # glucose clearance
    isf = 30 + 40 * si                                               # insulin sensitivity factor
    a_gain = A_GAIN * rng.uniform(0.85, 1.15)
    ka = KA_BASE * rng.uniform(0.7, 1.3)

    # --- daily schedule of meals, insulin and activity ---
    carbs, insulin, steps = np.zeros(n), np.zeros(n), np.zeros(n)
    for d in range(days):
        day0 = d * SPD
        a, b = day0 + 7 * 60 // STEP_MIN, day0 + 22 * 60 // STEP_MIN
        steps[a:b] += np.clip(rng.normal(250, 90, b - a), 0, None)   # background steps
        events = list(MEALS)
        if rng.random() < 0.4:
            events.append(SNACK)
        for hour, jit, mu, sd, walk_p in events:
            if rng.random() < 0.07:                                  # skipped meal
                continue
            t = int(np.clip(day0 + round((hour + rng.normal(0, jit)) * 60 / STEP_MIN), 0, n - 1))
            c = max(10.0, rng.normal(mu, sd))
            carbs[t] += c
            if on_ins and rng.random() > 0.1:                        # 10% missed boluses
                insulin[t] += c / (12 + 14 * si) * INS_COVER * rng.lognormal(0, 0.3)
            if rng.random() < walk_p:
                s = t + int(rng.integers(1, 4))
                seg = steps[s:s + int(rng.integers(1, 3))]
                seg += rng.normal(1500, 300, len(seg))

    # --- glucose-insulin simulation ---
    G, g1, g2, i1, i2, E = gb, 0.0, 0.0, 0.0, 0.0, 0.0
    glucose = np.empty(n)
    daily = 1.0
    for t in range(n):
        if t % SPD == 0:
            daily = rng.lognormal(0, 0.1)                            # day-to-day variability
        hour = (t % SPD) * STEP_MIN / 60
        g1 += carbs[t]
        i1 += insulin[t]
        rate = steps[t] / STEP_MIN / 100.0
        gb_t = gb + 8 * np.cos(2 * np.pi * (hour - 7) / 24)         # dawn phenomenon
        for _ in range(SUB):
            dg1 = -KE * g1
            dg2 = KE * g1 - ka * g2
            di1 = -KI * i1
            di2 = KI * i1 - KI * i2
            dE = (rate - E) / TAU_EX
            dG = (-kc * daily * (G - gb_t) + a_gain * ka * g2
                  - EX_GAIN * si * E - isf * KI * i2)
            g1 += DT * dg1
            g2 += DT * dg2
            i1 += DT * di1
            i2 += DT * di2
            E += DT * dE
            G = float(np.clip(G + DT * dG + rng.normal(0, 0.8), 35, 450))
        glucose[t] = G

    # --- CGM sensor: AR(1) noise, clipping, small random gaps ---
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 2.5)
    cgm = np.clip(glucose + noise, 40, 400)
    t = 0
    while t < n:
        if rng.random() < 0.004:
            L = int(rng.integers(1, 7))
            cgm[t:t + L] = np.nan
            t += L
        t += 1

    ts = pd.date_range("2026-01-01", periods=n, freq=f"{STEP_MIN}min")
    return pd.DataFrame({
        "timestamp": ts,
        "glucose": cgm.round(1),
        "carbs": carbs.round(1),
        "steps": steps.round(0),
        "insulin": insulin.round(2),
    })


def build_cohort(n_patients, days, seed):
    rng = np.random.default_rng(seed)
    ehr = make_ehr(n_patients, rng)
    frames = []
    for _, row in ehr.iterrows():
        df = simulate_patient(row, days, rng)
        df.insert(0, "patient_id", int(row["patient_id"]))
        frames.append(df)
    return ehr, pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=40)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synthetic")
    args = ap.parse_args()

    ehr, cgm = build_cohort(args.patients, args.days, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ehr.to_csv(out / "ehr.csv", index=False)
    cgm.to_csv(out / "cgm.csv", index=False)

    g = cgm["glucose"].dropna()
    print(f"saved {len(ehr)} patients, {len(cgm)} rows -> {out}")
    print(f"mean glucose {g.mean():.0f} mg/dL | TIR(70-180) {((g >= 70) & (g <= 180)).mean():.0%} "
          f"| TBR(<70) {(g < 70).mean():.1%} | TAR(>180) {(g > 180).mean():.0%}")


if __name__ == "__main__":
    main()
