"""
Build the ESP live-monitor dashboard for a brand-new well.

Loads the models frozen by Step 7 of ESP_Capstone_Colab.ipynb, simulates a well
the models have never seen (well 17 by default), scores it minute by minute with
those frozen models, and writes ESP_Live_Dashboard.html - one self-contained page
that replays the well as if it were a live SCADA feed, at any speed.

NOTHING HERE RE-TRAINS ANYTHING. The models are loaded from disk and only ever
predict. The replay is honest in the same way a real control room is:

  * Days 1-30 are a calibration window. The new well's baseline levels and its
    alarm threshold come from those 30 days alone - no labels, no future data.
    (The notebook's offline scorer takes the threshold from all six months,
    which a live system could not do, so the numbers here differ slightly.)
  * Every feature is backward-looking (rolling windows, time since last trip),
    so each minute's score uses only what had already happened.
  * The actual failure cause is revealed only when the pump actually trips.

Usage - from the notebook, so it runs in the same Python that trained the models:
    %run ../6_dashboard/build_dashboard.py
or from a terminal:
    python build_dashboard.py [--work ../data] [--well 17]
                              [--causes "UNDERLOAD:0.5;HIGH_DISCHARGE:0.5"]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "2_code")
BUCKET_MIN = 5          # dashboard resolution: one point per 5 minutes
SHOWCASE_MIN_LEAD = 120  # the guided demo slows down on a catch with at least this much warning

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.environ.get("ESP_WORK", os.path.join(os.path.dirname(HERE), "data")))
ap.add_argument("--well", type=int, default=17)
ap.add_argument("--causes", default="UNDERLOAD:0.5;HIGH_DISCHARGE:0.5")
ap.add_argument("--out", default=os.path.join(HERE, "ESP_Live_Dashboard.html"))
ap.add_argument("--breakeven", type=float, default=None,
                help="break-even downtime per trip (h) from Step 5, shown next to the frequency advice")
args = ap.parse_args()

# Frequency recommendation constants - the same rule as frequency_recommendation.py
DEFICIT_TRIGGER = 18.0   # psi below expected intake pressure before a change is advised
RECOVERY_MARGIN = 12.0   # psi of headroom to aim for above the trigger

STORE = os.path.join(args.work, "model_store", "esp_models.joblib")
if not os.path.exists(STORE):
    sys.exit(f"No frozen models at {STORE}.\n"
             "Run ESP_Capstone_Colab.ipynb through Step 7 (Save the trained models) first.")
store = joblib.load(STORE)
WELL = args.well
if WELL in [int(w) for w in store["trained_on_wells"]]:
    sys.exit(f"Well {WELL} was used for training - pick a well id the models have never seen.")


# ============================================================
# 1. Simulate the new well in a scratch folder.
# The project's own equipment_specs.py and generate_multiwell_data.py are used
# unchanged apart from the output path, so well 17 gets exactly the equipment and
# data it would get from the notebook. The training data is never touched.
# ============================================================
def simulate_new_well():
    tmp = tempfile.mkdtemp(prefix="esp_newwell_")
    tmp_fwd = tmp.replace("\\", "/")
    try:
        roster = pd.read_csv(os.path.join(args.work, "wells_config.csv"))
        roster = roster[roster["well_id"] != WELL]
        roster = pd.concat([roster, pd.DataFrame([{"well_id": WELL, "causes": args.causes,
                                                   "split": "test"}])], ignore_index=True)
        roster.to_csv(os.path.join(tmp, "wells_config.csv"), index=False)

        for name in ("equipment_specs.py", "generate_multiwell_data.py"):
            shutil.copy(os.path.join(CODE, name), os.path.join(tmp, name))

        # The generator only simulates wells missing from its output file, so listing
        # the existing wells in a stub makes it simulate the new one alone - exactly as
        # the notebook's "add a well" step does.
        cols = ["well_id", "split", "timestamp", "state", "frequency_hz", "motor_current_a",
                "voltage_v", "Ti_intake_temp_f", "Tm_motor_temp_f", "Pi_intake_psi",
                "Pd_discharge_psi", "vibration_g", "shutdown_event", "failure_cause",
                "onset_cause", "safe_min_hz", "safe_max_hz", "production_bpd"]
        stub = pd.DataFrame({"well_id": roster.loc[roster["well_id"] != WELL, "well_id"]})
        stub.reindex(columns=cols).to_csv(os.path.join(tmp, "multiwell_scada_10wells_6mo.csv"),
                                          index=False)

        # the scripts write to ESP_DATA and read the roster from ESP_CONFIG: both point at the scratch folder
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   ESP_DATA=tmp_fwd, ESP_CONFIG=os.path.join(tmp, "wells_config.csv"))
        for name in ("equipment_specs.py", "generate_multiwell_data.py"):
            print(f"  running {name} ...")
            p = subprocess.run([sys.executable, name], cwd=tmp, env=env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            if p.returncode != 0:
                sys.exit(p.stdout[-2000:] + p.stderr[-2000:])

        df = pd.read_csv(os.path.join(tmp, "multiwell_scada_10wells_6mo.csv"),
                         parse_dates=["timestamp"], low_memory=False)
        df = df[df["well_id"] == WELL].sort_values("timestamp").reset_index(drop=True)
        specs = pd.read_csv(os.path.join(tmp, "well_equipment_specs.csv")).set_index("well_id")
        return df, specs.loc[WELL]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 2. Features - identical to run_full_analysis.py (and score_new_well.py).
# ============================================================
def build_features(df, sensor_cols):
    for col in sensor_cols + ["frequency_hz"]:
        df[col] = df[col].astype("float32")
    for col in sensor_cols:
        s = df[col]
        df[f"{col}_roll15_mean"] = s.rolling(15, min_periods=1).mean().astype("float32")
        df[f"{col}_roll15_std"] = s.rolling(15, min_periods=1).std().fillna(0).astype("float32")
        df[f"{col}_roll60_mean"] = s.rolling(60, min_periods=1).mean().astype("float32")
        df[f"{col}_roll60_std"] = s.rolling(60, min_periods=1).std().fillna(0).astype("float32")
        df[f"{col}_roll240_mean"] = s.rolling(240, min_periods=1).mean().astype("float32")
        df[f"{col}_chg5"] = s.diff(5).fillna(0).astype("float32")
    arr = df["shutdown_event"].to_numpy()
    since = np.empty(len(arr), dtype="float32")
    last = None
    for i in range(len(arr)):
        since[i] = float(i) if last is None else float(i - last)
        if arr[i] == 1:
            last = i
    df["minutes_since_last_shutdown"] = since
    return df


def mode(values):
    values = [v for v in values if v is not None]
    return max(set(values), key=values.count) if values else None


print(f"[1/4] Simulating new well {WELL} ({args.causes}) - the models have never seen it")
df, spec = simulate_new_well()
print(f"      {len(df):,} minutes, {int(df['shutdown_event'].sum())} real shutdowns (held aside)")

print("[2/4] Building features")
df = build_features(df, store["sensor_cols"])
t0 = df["timestamp"].min()
df["minute"] = ((df["timestamp"] - t0).dt.total_seconds() // 60).astype(int)
N_MIN = int(df["minute"].max()) + 1
CAL_END = store["calib_days"] * 1440
running = df["state"].isin(["normal", "onset", "ramp"]).to_numpy()

# ============================================================
# 3. Score every minute with the FROZEN models.
# ============================================================
print(f"[3/4] Scoring with the frozen {store['stage1_name']} detector + root-cause model")
feats = store["all_features"]
run = df[running].copy()
run["risk"] = store["stage1_model"].predict_proba(run[feats].to_numpy(dtype="float32"))[:, 1]

# Live alarm threshold: this well's own top-X% score level, measured on the
# calibration window only - the only scores a live system would have had.
calib = run["minute"] < CAL_END
thr = float(run.loc[calib, "risk"].quantile(store["alarm_quantile"]))
thr_offline = float(run["risk"].quantile(store["alarm_quantile"]))

# Root cause: normalise against the calibration window, add the equipment spec sheet.
mean = run.loc[calib, feats].mean()
std = run.loc[calib, feats].std().combine(store["global_std_floor"], max).replace(0, 1)
Z = ((run[feats] - mean) / std).astype("float32")
for c in store["equip_features"]:
    Z[c] = float(spec[c])
cause_proba = store["stage2_model"].predict_proba(Z[store["stage2_features"]].to_numpy(dtype="float32"))
CLASSES = list(store["stage2_label_encoder"].classes_)
run["cause"] = cause_proba.argmax(1)
run["cause_p"] = cause_proba.max(1)

# Frequency advice for underload - Step 5's method, fitted on this well alone.
# Linear regression of intake pressure on frequency over the calibration window
# (normal/ramp minutes; post-trip ramps supply the frequency spread):
#     Pi = a - k * f          k = psi of drawdown per Hz
# Then at any minute: deficit = expected Pi - observed Pi, and giving that pressure
# back takes delta_f = (deficit + margin) / k, clipped to the equipment's safe band.
# Observed Pi is the 15-minute rolling mean, so one noisy reading cannot move the advice.
fit_rows = df[(df["minute"] < CAL_END) & df["state"].isin(["normal", "ramp"]) & (df["frequency_hz"] > 0)]
k_neg, a_fit = np.polyfit(fit_rows["frequency_hz"].to_numpy(float), fit_rows["Pi_intake_psi"].to_numpy(float), 1)
K_FIT = float(-k_neg)
f_now = run["frequency_hz"].to_numpy(float)
deficit = (a_fit - K_FIT * f_now) - run["Pi_intake_psi_roll15_mean"].to_numpy(float)
target = f_now - (deficit + RECOVERY_MARGIN) / max(K_FIT, 1e-6)
safe_min, safe_max = float(spec["safe_min_hz"]), float(spec["safe_max_hz"])
run["deficit"] = deficit
run["rec_hz"] = np.where(deficit >= DEFICIT_TRIGGER, np.clip(target, safe_min, safe_max), np.nan)
run["escalate"] = ((deficit >= DEFICIT_TRIGGER) & (target < safe_min)).astype(int)

# Alarms: N consecutive running minutes above threshold, live window only.
live = run[~calib]
above = (live["risk"].to_numpy() >= thr).astype(int)
n_p = int(store["alarm_persist_min"])
csum = np.convolve(above, np.ones(n_p, dtype=int), mode="full")[:len(above)]
fired = (csum == n_p).astype(int)
d = np.diff(np.concatenate(([0], fired, [0])))
starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
live_min = live["minute"].to_numpy()
live_cause = live["cause"].to_numpy()
alarm_flag = np.zeros(N_MIN, dtype=np.int8)
alarms = []
for s, e in zip(starts, ends):
    alarm_flag[live_min[s]:live_min[e] + 1] = 1
    # the cause shown when the alarm is raised uses only the minutes that raised it
    first = mode([int(c) for c in live_cause[max(0, s - n_p + 1):s + 1]])
    alarms.append({"s": int(live_min[s]), "e": int(live_min[e]), "cause": first})

# ============================================================
# Only now look at what actually happened.
# ============================================================
WINDOW = int(store["alert_window_min"])
USEFUL = float(store["min_useful_lead_min"])
trips = []
for _, r in df[df["shutdown_event"] == 1].iterrows():
    m, scored = int(r["minute"]), int(r["minute"]) >= CAL_END
    hits = [a for a in alarms if a["s"] < m and a["e"] >= m - WINDOW] if scored else []
    lead, pcause = None, None
    if hits:
        a0 = max(min(a["s"] for a in hits), m - WINDOW)
        lead = m - a0
        in_win = run[(run["minute"] >= m - WINDOW) & (run["minute"] < m)]
        in_win = in_win[alarm_flag[in_win["minute"].to_numpy()] == 1]
        pcause = mode([int(c) for c in in_win["cause"]])
        for a in hits:
            a["matched"] = True
    trips.append({"m": m, "cause": str(r["failure_cause"]), "scored": scored,
                  "caught": bool(hits), "lead": lead,
                  "pcause": pcause, "useful": bool(lead is not None and lead >= USEFUL)})
for a in alarms:
    a["matched"] = bool(a.get("matched", False))
    a["resolve"] = a["e"] + WINDOW     # when you can know it was a nuisance alarm

scored = [t for t in trips if t["scored"]]
caught = [t for t in scored if t["caught"]]
nuisance = [a for a in alarms if not a["matched"]]
live_days = (N_MIN - CAL_END) / 1440
onset = run[run["onset_cause"].fillna("").astype(str).isin(CLASSES)]
cause_hits = int((np.array(CLASSES)[onset["cause"].to_numpy()] == onset["onset_cause"].astype(str).to_numpy()).sum())
cause_at_trip = [t for t in caught if t["pcause"] is not None]
summary = {
    "trips": len(scored), "caught": len(caught),
    "useful": sum(t["useful"] for t in caught), "missed": len(scored) - len(caught),
    "median_lead": float(np.median([t["lead"] for t in caught])) if caught else None,
    "nuisance": len(nuisance), "nuisance_per_month": round(len(nuisance) / (live_days / 30), 1),
    "cause_minutes_ok": cause_hits, "cause_minutes": len(onset),
    "cause_at_trip_ok": sum(CLASSES[t["pcause"]] == t["cause"] for t in cause_at_trip),
    "cause_at_trip_n": len(cause_at_trip),
    "calib_trips": len(trips) - len(scored),
}

# ============================================================
# 4. Export: 5-minute buckets over the whole six months.
# ============================================================
print("[4/4] Writing the dashboard")
df["b"] = df["minute"] // BUCKET_MIN
run["b"] = run["minute"] // BUCKET_MIN
g = df.groupby("b")
agg = pd.DataFrame({
    "freq": g["frequency_hz"].mean(), "cur": g["motor_current_a"].mean(),
    "pi": g["Pi_intake_psi"].mean(), "pd": g["Pd_discharge_psi"].mean(),
    "tm": g["Tm_motor_temp_f"].mean(), "vib": g["vibration_g"].mean(),
    "down": g["state"].agg(lambda s: int((s == "shutdown").any())),
}).reindex(range(N_MIN // BUCKET_MIN + 1))
rg = run.groupby("b")
agg["risk"] = rg["risk"].max()
agg["cause"] = rg["cause"].agg(lambda s: int(s.mode().iloc[0]))
agg["cause_p"] = rg["cause_p"].mean()
agg["deficit"] = rg["deficit"].mean()
agg["rec"] = rg["rec_hz"].mean()
agg["esc"] = rg["escalate"].max()
agg["alarm"] = [int(alarm_flag[i * BUCKET_MIN:(i + 1) * BUCKET_MIN].any()) for i in range(len(agg))]


def col(name, nd):
    v = agg[name].round(nd)
    return [None if pd.isna(x) else (int(x) if nd == 0 else float(x)) for x in v]


showcase = next((t for t in trips if t["scored"] and t["caught"] and t["lead"] >= SHOWCASE_MIN_LEAD), None)
data = {
    "well": WELL, "start": t0.strftime("%Y-%m-%dT%H:%M:%S"), "n_minutes": N_MIN,
    "B": BUCKET_MIN, "calib_end": CAL_END, "threshold": round(thr, 4),
    "classes": CLASSES, "scenario": args.causes,
    "equipment": {
        "motor": str(spec["motor_model"]), "protector": str(spec["protector_model"]),
        "separator": f"{spec['gas_separator_model']} ({int(spec['gas_separator_efficiency_pct'])}%)",
        "pump": str(spec["pump_model"]), "safe_min": float(spec["safe_min_hz"]),
        "safe_max": float(spec["safe_max_hz"]), "bep": float(spec["pump_bep_hz"]),
    },
    "model": {
        "detector": store["stage1_name"],
        "trained_on": [int(w) for w in store["trained_on_wells"]],
        "top_pct": round(100 * (1 - store["alarm_quantile"]), 1),
        "persist": n_p, "window": WINDOW, "horizon": int(store["horizon_min"]),
    },
    "series": {k: col(k, nd) for k, nd in [("risk", 3), ("freq", 1), ("cur", 1), ("pi", 0),
                                          ("pd", 0), ("tm", 1), ("vib", 3), ("down", 0),
                                          ("alarm", 0), ("cause", 0), ("cause_p", 2),
                                          ("deficit", 0), ("rec", 1), ("esc", 0)]},
    "freq_advice": {"k": round(K_FIT, 2), "a": round(float(a_fit), 0),
                    "trigger": DEFICIT_TRIGGER, "margin": RECOVERY_MARGIN,
                    "breakeven_h": args.breakeven},
    "alarms": alarms, "trips": trips, "summary": summary,
    "showcase": showcase["m"] if showcase else None,
}

template = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read()
html = template.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
open(args.out, "w", encoding="utf-8").write(html)

s = summary
print(f"\nWell {WELL} - live replay, scored after the {store['calib_days']}-day calibration window:")
print(f"  shutdowns warned  : {s['caught']} of {s['trips']} "
      f"({s['useful']} with >= {USEFUL:.0f} min warning), median warning {s['median_lead']} min")
print(f"  nuisance alarms   : {s['nuisance']} (~{s['nuisance_per_month']} per month)")
print(f"  root cause        : {s['cause_minutes_ok']} of {s['cause_minutes']} onset minutes correct")
print(f"  alarm threshold   : {thr:.3f} from calibration (offline scorer would use {thr_offline:.3f})")
print(f"  drawdown fit      : Pi = {a_fit:.0f} - {K_FIT:.2f} x Hz (days 1-30); frequency advice on underload alarms"
      + (f", break-even {args.breakeven:.0f} h/trip" if args.breakeven else ""))
print(f"\nSaved {args.out}")
print(f"Size  {os.path.getsize(args.out) / 1e6:.1f} MB - open it in a browser.")
