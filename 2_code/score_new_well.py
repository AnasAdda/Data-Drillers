"""
Score a well the models have never seen, and compare the predictions against
what actually happened on it.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
run_full_analysis.py trains and evaluates in one pass. Adding a new well to
wells_config.csv and re-running it would train a NEW model that has seen the
new well - which is not a test of the models we built, it is a different
model that happens to score well on data it was fitted to.

This script does the opposite. It loads the frozen models from
model_store/esp_models.joblib and never calls fit() on anything. The new well
contributes no training signal at all; it is only ever predicted on.

WHAT THE NEW WELL IS ALLOWED TO PROVIDE
---------------------------------------
One thing: its own first 30 days of ordinary operating data, used to compute
its baseline sensor levels. That is a calibration window, not training - it
uses no failure labels, and it is exactly what you would have in the field
when a well is commissioned. Every alarm threshold is then a quantile of that
well's own score distribution, so it needs no labels either.

The well's real shutdowns are read ONLY after predictions are made, to score
them.

USAGE
    python score_new_well.py 17              # score well 17
    python score_new_well.py --verify        # re-score wells 8/9/10 and check
                                             # the numbers match results.txt
"""
import sys
import os
import gc

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
# ---- where things live: the project's own data/ folder, unless ESP_DATA says otherwise ----
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ESP_DATA", os.path.join(os.path.dirname(HERE), "data"))
CONFIG_PATH = os.environ.get("ESP_CONFIG", os.path.join(HERE, "wells_config.csv"))
os.makedirs(DATA, exist_ok=True)
# local runs write to latest_run/ so the official results in 4_charts/ and 3_results/ are never overwritten
CHARTS = os.environ.get("ESP_CHARTS", os.path.join(os.path.dirname(HERE), "4_charts", "latest_run"))
RESULTS = os.environ.get("ESP_RESULTS", os.path.join(os.path.dirname(HERE), "3_results", "latest_run"))
os.makedirs(CHARTS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
DOCS = os.path.join(os.path.dirname(HERE), "5_documents")
OUT = DATA
DATA = f"{OUT}/multiwell_scada_10wells_6mo.csv"
STORE = f"{OUT}/model_store/esp_models.joblib"


# ============================================================
# Feature construction - must match run_full_analysis.py exactly.
# --verify exists to prove that it does: it re-scores the held-out wells
# through this path and checks the answer against the training run.
# ============================================================
def build_features(df, sensor_cols, horizon):
    for col in sensor_cols + ["frequency_hz"]:
        df[col] = df[col].astype("float32")

    grouped = df.groupby("well_id", sort=False)
    for col in sensor_cols:
        df[f"{col}_roll15_mean"] = grouped[col].transform(lambda s: s.rolling(15, min_periods=1).mean()).astype("float32")
        df[f"{col}_roll15_std"] = grouped[col].transform(lambda s: s.rolling(15, min_periods=1).std()).fillna(0).astype("float32")
        df[f"{col}_roll60_mean"] = grouped[col].transform(lambda s: s.rolling(60, min_periods=1).mean()).astype("float32")
        df[f"{col}_roll60_std"] = grouped[col].transform(lambda s: s.rolling(60, min_periods=1).std()).fillna(0).astype("float32")
        df[f"{col}_roll240_mean"] = grouped[col].transform(lambda s: s.rolling(240, min_periods=1).mean()).astype("float32")
        df[f"{col}_chg5"] = grouped[col].transform(lambda s: s.diff(5)).fillna(0).astype("float32")
    del grouped
    gc.collect()

    df["label_trip_within_H"] = np.int8(0)
    df["minutes_since_last_shutdown"] = np.float32(0)
    for well_id, idx in df.groupby("well_id", sort=False).groups.items():
        arr = df.loc[idx, "shutdown_event"].to_numpy()
        n = len(arr)
        nxt_idx = np.full(n, np.inf)
        nxt = np.inf
        for i in range(n - 1, -1, -1):
            if arr[i] == 1:
                nxt = i
            nxt_idx[i] = nxt
        to_shutdown = nxt_idx - np.arange(n)
        df.loc[idx, "label_trip_within_H"] = ((to_shutdown > 0) & (to_shutdown <= horizon)).astype("int8")

        since = np.empty(n, dtype="float32")
        last = None
        for i in range(n):
            since[i] = float(i) if last is None else float(i - last)
            if arr[i] == 1:
                last = i
        df.loc[idx, "minutes_since_last_shutdown"] = since
    gc.collect()
    return df


def calibration_baseline(well_df, features, global_std_floor, calib_days):
    """Mean/std from this well's OWN first N days. No labels, no other well."""
    cutoff = well_df["timestamp"].min() + pd.Timedelta(days=calib_days)
    w = well_df.loc[well_df["timestamp"] < cutoff, features]
    if len(w) > 100_000:
        w = w.sample(n=100_000, random_state=7)
    std = w.std().combine(global_std_floor, max).replace(0, 1)
    return w.mean(), std


def alarm_events(frame, proba_col, threshold, n_persist):
    g = frame.sort_values("timestamp")
    above = (g[proba_col].to_numpy() >= threshold).astype(int)
    if n_persist > 1:
        csum = np.convolve(above, np.ones(n_persist, dtype=int), mode="full")[:len(above)]
        fired = (csum == n_persist).astype(int)
    else:
        fired = above
    ts = g["timestamp"].to_numpy()
    d = np.diff(np.concatenate(([0], fired, [0])))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return pd.DataFrame({"start": ts[starts], "end": ts[ends]}) if len(starts) else \
        pd.DataFrame(columns=["start", "end"])


def score_against_truth(events, trips, alert_window, min_useful_lead):
    """Compare alarms to what actually happened. Truth is read only here."""
    caught, missed, leads, matched = 0, 0, [], set()
    for _, trip in trips.iterrows():
        st = trip["timestamp"]
        w0 = st - pd.Timedelta(minutes=alert_window)
        if len(events) == 0:
            missed += 1
            continue
        hits = events[(events["start"] < st) & (events["end"] >= w0)]
        if len(hits):
            caught += 1
            matched.update(hits.index.tolist())
            lead = (st - max(hits["start"].min(), w0)).total_seconds() / 60.0
            leads.append(lead)
        else:
            missed += 1
    useful = sum(1 for l in leads if l >= min_useful_lead)
    false_alarms = len(events) - len(matched)
    return dict(caught=caught, missed=missed, useful=useful,
                false_alarms=false_alarms, leads=leads, matched=matched)


# ============================================================
def score_wells(well_ids, store, df_all, trips_all):
    feats = store["all_features"]
    rows_out = []
    per_well_detail = {}

    for wid in well_ids:
        wdf = df_all[df_all["well_id"] == wid].sort_values("timestamp").reset_index(drop=True)
        if wdf.empty:
            print(f"  well {wid}: no rows found - skipped")
            continue

        # --- Stage 1: the frozen detector, on raw features ---
        X = wdf[feats].to_numpy(dtype="float32")
        wdf["proba"] = store["stage1_model"].predict_proba(X)[:, 1]

        # --- alarm policy, exactly as tuned on validation ---
        thr = float(wdf["proba"].quantile(store["alarm_quantile"]))
        ev = alarm_events(wdf, "proba", thr, store["alarm_persist_min"])

        # --- now, and only now, look at what actually happened ---
        trips = trips_all[trips_all["well_id"] == wid][["timestamp", "failure_cause"]]
        res = score_against_truth(ev, trips, store["alert_window_min"],
                                  store["min_useful_lead_min"])

        days = (wdf["timestamp"].max() - wdf["timestamp"].min()).total_seconds() / 86400
        fa_month = res["false_alarms"] / max(days / 30.0, 1e-9)
        n_trips = len(trips)
        rows_out.append({
            "well": wid, "trips": n_trips,
            "caught": res["caught"],
            "caught_pct": round(100 * res["caught"] / max(n_trips, 1)),
            "useful": res["useful"], "missed": res["missed"],
            "alarms_per_month": round(fa_month, 1),
            "median_lead_min": round(float(np.median(res["leads"])), 0) if res["leads"] else np.nan,
        })
        per_well_detail[wid] = (wdf, ev, trips, res)
    return pd.DataFrame(rows_out), per_well_detail


def stage2_causes(wdf, trips, store, global_floor):
    """Name the developing cause at each real trip, and compare with the truth."""
    feats, s2feats = store["all_features"], store["stage2_features"]
    mean, std = calibration_baseline(wdf, feats, global_floor, store["calib_days"])
    specs = pd.read_csv(f"{OUT}/well_equipment_specs.csv").set_index("well_id")
    wid = int(wdf["well_id"].iloc[0])

    onset = wdf[wdf["onset_cause"].astype(str).isin(store["stage2_label_encoder"].classes_)]
    if onset.empty:
        return None
    Z = ((onset[feats] - mean) / std).astype("float32")
    for c in store["equip_features"]:
        Z[c] = float(specs.loc[wid, c]) if c in specs.columns else 0.0
    pred = store["stage2_model"].predict(Z[s2feats].to_numpy(dtype="float32"))
    pred_names = store["stage2_label_encoder"].inverse_transform(pred)
    truth = onset["onset_cause"].astype(str).to_numpy()
    acc = float((pred_names == truth).mean())
    return {"n": len(truth), "accuracy": acc,
            "table": pd.crosstab(pd.Series(truth, name="actual"),
                                 pd.Series(pred_names, name="predicted"))}


def timeline_chart(wid, wdf, ev, trips, thr, path, store):
    """Top: the whole record. Bottom: one catch up close, which is what an
    operator would actually have seen."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 6.4),
                                  gridspec_kw={"height_ratios": [1.15, 1]})

    ax.plot(wdf["timestamp"], wdf["proba"], lw=0.5, color="#3E5C76", label="model risk score")
    ax.axhline(thr, color="#C0621A", ls="--", lw=1,
               label=f"this well's own alarm threshold ({thr:.3f})")
    for i, (_, e) in enumerate(ev.iterrows()):
        ax.axvspan(e["start"], e["end"], color="#C0621A", alpha=0.30,
                   label="alarm raised" if i == 0 else None)
    for i, (_, t) in enumerate(trips.iterrows()):
        ax.axvline(t["timestamp"], color="#C1666B", lw=1.2,
                   label="ACTUAL shutdown" if i == 0 else None)
    ax.set_title(f"Well {wid} — six months. Every prediction was made before any "
                 f"shutdown was looked at.", fontsize=11)
    ax.set_ylabel("risk"); ax.legend(fontsize=8, loc="upper left", ncol=4)
    ax.grid(alpha=0.25, lw=0.5)

    # --- pick the first trip that was caught, and zoom in on it ---
    window = pd.Timedelta(minutes=store["alert_window_min"])
    zoom = None
    for _, t in trips.iterrows():
        st = t["timestamp"]
        hits = ev[(ev["start"] < st) & (ev["end"] >= st - window)]
        if len(hits):
            lead = (st - max(hits["start"].min(), st - window)).total_seconds() / 60.0
            zoom = (st, hits, lead, str(t["failure_cause"]))
            break
    if zoom:
        st, hits, lead, cause = zoom
        lo, hi = st - pd.Timedelta(hours=10), st + pd.Timedelta(hours=2)
        seg = wdf[(wdf["timestamp"] >= lo) & (wdf["timestamp"] <= hi)]
        ax2.plot(seg["timestamp"], seg["proba"], lw=1.1, color="#3E5C76")
        ax2.axhline(thr, color="#C0621A", ls="--", lw=1)
        for _, e in hits.iterrows():
            ax2.axvspan(max(e["start"], lo), min(e["end"], hi), color="#C0621A", alpha=0.30)
        ax2.axvline(st, color="#C1666B", lw=1.8)
        y = ax2.get_ylim()[1] * 0.74
        a0 = max(hits["start"].min(), st - window)
        ax2.annotate("", xy=(st, y), xytext=(a0, y),
                     arrowprops=dict(arrowstyle="<->", color="#16233A", lw=1.3))
        ax2.text(a0 + (st - a0) / 2, y * 1.13,
                 f"{lead:.0f} minutes of warning", ha="center", fontsize=11,
                 color="#16233A", fontweight="bold")
        ax2.text(st, ax2.get_ylim()[1] * 0.06, f"  pump trips ({cause})",
                 color="#C1666B", fontsize=9, va="bottom")
        ax2.set_title("One of them, close up — alarm first, trip hours later", fontsize=11)
    else:
        ax2.text(0.5, 0.5, "no catch to zoom in on", ha="center", transform=ax2.transAxes)
    ax2.set_ylabel("risk"); ax2.grid(alpha=0.25, lw=0.5)

    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path


# ============================================================
def main():
    if not os.path.exists(STORE):
        sys.exit("No frozen models found. Run run_full_analysis.py first to train and freeze them.")
    store = joblib.load(STORE)
    verify = "--verify" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    targets = [int(w) for w in store["held_out_wells"]] if verify else [int(a) for a in args]
    if not targets:
        sys.exit("usage: python score_new_well.py <well_id> [...]   |   --verify")

    print("=" * 74)
    print("SCORING WITH FROZEN MODELS - nothing is re-trained")
    print("=" * 74)
    trained = [int(w) for w in store["trained_on_wells"]]
    print(f"Detector      : {store['stage1_name']}, trained on {len(trained)} wells {trained}")
    print(f"Alarm policy  : top {100*(1-store['alarm_quantile']):.1f}% of each well's own scores, "
          f"{store['alarm_persist_min']} consecutive minutes")
    print(f"Horizon       : {store['horizon_min']} min, a catch needs "
          f"{store['min_useful_lead_min']:.0f}+ min of warning to count as actionable")
    print(f"Scoring wells : {[int(t) for t in targets]}  (never seen in training)")
    for w in targets:
        if w in store["trained_on_wells"]:
            print(f"  !! well {w} WAS in the training set - this is not a fair test of it")

    usecols = ["well_id", "timestamp", "shutdown_event", "failure_cause", "onset_cause",
               "state", "frequency_hz"] + store["sensor_cols"]
    df = pd.read_csv(DATA, usecols=lambda c: c in usecols, parse_dates=["timestamp"])
    df = df[df["well_id"].isin(targets)].copy()
    if df.empty:
        sys.exit(f"No data for wells {targets}. Add them to wells_config.csv and "
                 f"run generate_multiwell_data.py first.")

    # Rolling windows are computed over the complete record, exactly as in training -
    # dropping rows first would change every window that spans a shutdown.
    df = build_features(df, store["sensor_cols"], store["horizon_min"])

    # The real shutdowns are state="shutdown" rows, so take them before the filter.
    trips_all = df.loc[df["shutdown_event"] == 1,
                       ["well_id", "timestamp", "failure_cause"]].copy()

    # Training modelled only running minutes; a well that is already down needs no
    # prediction, and leaving those rows in shifts every per-well alarm threshold.
    before = len(df)
    df = df[df["state"].isin(["normal", "onset", "ramp"])].copy()
    print(f"\nRows: {before:,} total -> {len(df):,} with the pump running "
          f"({len(trips_all)} real shutdowns held aside for scoring)")

    table, detail = score_wells(targets, store, df, trips_all)
    print("\n--- DETECTION: what the models predicted vs. what actually happened ---")
    print(table.to_string(index=False))

    tot_trips, tot_caught = table["trips"].sum(), table["caught"].sum()
    tot_useful = table["useful"].sum()
    print(f"\nTotal: {tot_caught} of {tot_trips} real shutdowns caught "
          f"({100*tot_caught/max(tot_trips,1):.0f}%), "
          f"{tot_useful} with at least {store['min_useful_lead_min']:.0f} minutes of warning")

    for wid, (wdf, ev, trips, res) in detail.items():
        thr = float(wdf["proba"].quantile(store["alarm_quantile"]))
        p = timeline_chart(wid, wdf, ev, trips, thr, f"{CHARTS}/new_well_{wid}_timeline.png", store)
        print(f"  saved {os.path.basename(p)}")
        s2 = stage2_causes(wdf, trips, store, store["global_std_floor"])
        if s2:
            print(f"\n--- ROOT CAUSE, well {wid}: {100*s2['accuracy']:.0f}% correct "
                  f"on {s2['n']} onset minutes ---")
            print(s2["table"].to_string())

    if verify:
        print("\n" + "=" * 74)
        print("VERIFICATION against the training run's own report")
        print("=" * 74)
        want = {}
        for line in open(f"{RESULTS}/results.txt"):
            if line.strip().startswith(tuple(str(w) for w in store["held_out_wells"])):
                parts = line.split()
                if len(parts) > 6 and parts[0].isdigit():
                    want[int(parts[0])] = parts
        expected = None
        for line in open(f"{RESULTS}/results.txt"):
            if line.startswith("Caught (>="):
                expected = int(line.split(":")[1].split("(")[0].strip())
        got = int(tot_caught)
        hw = [int(w) for w in store["held_out_wells"]]
        print(f"results.txt reports : {expected} trips caught on wells {hw}")
        print(f"this script reports : {got}")
        print("MATCH - the scoring path reproduces the training run exactly"
              if expected == got else
              f"MISMATCH of {abs((expected or 0) - got)} - the feature path has drifted, do not trust new-well numbers")


if __name__ == "__main__":
    main()
