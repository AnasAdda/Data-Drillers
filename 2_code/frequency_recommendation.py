"""
UNDERLOAD frequency recommendation, and a closed-loop back-test of it.

WHY THIS EXISTS
---------------
Underload (pump-off / inflow starvation) happens when the pump's throughput
exceeds what the reservoir can deliver: the fluid level over the pump falls,
intake pressure drops, the motor unloads and the drive eventually trips.

The standard field remedy is exactly a frequency change - slow the pump until
its rate matches the well's inflow. That is what a pump-off controller does.
An earlier version of this project classified UNDERLOAD as "not
frequency-related", which was wrong; it is the textbook frequency-adjustable
failure mode, and this script implements the recommendation.

THE RULE
--------
For each well, fit the normal-operation relationship between intake pressure
and frequency from that well's own early history:

    Pi_expected(f) = a - k * f          (k = psi of drawdown per Hz)

Frequency variation during post-trip ramps is what makes k identifiable from
ordinary operating data - no test or step-rate survey needed.

At any moment the drawdown deficit is how far intake pressure sits below what
that frequency should give:

    deficit = Pi_expected(f_now) - Pi_observed

Because dPi/df = -k, giving back `deficit` psi of intake pressure requires

    delta_f = (deficit + margin) / k

and the recommendation is f_now - delta_f, clipped to the well's own
equipment-rated band. If the clip binds - i.e. even the minimum safe frequency
cannot recover the deficit - frequency alone cannot save the well and the
recommendation escalates to inspection instead of pretending otherwise.

THE TEST
--------
The recommendation is back-tested closed-loop: each held-out well is
re-simulated from the same random seed, once with control off and once with the
controller acting on what a SCADA feed would actually expose. Because the
simulator's drawdown now responds to pump rate, slowing the pump genuinely
changes the outcome, so the comparison measures something real:

    - underload trips avoided
    - production given up to run slower (Q proportional to f, zero while down)
    - net effect on total produced volume
"""
import numpy as np
import pandas as pd

import generate_multiwell_data as sim

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
CALIB_DAYS = 30          # per-well calibration window - same one the detector uses
DEFICIT_TRIGGER = 18.0   # psi below expected before the controller acts
RECOVERY_MARGIN = 12.0   # psi of headroom to aim for above the trigger
RESTORE_STEP = 0.02      # Hz per minute crept back toward BEP once recovered

specs = pd.read_csv(f"{OUT}/well_equipment_specs.csv").set_index("well_id")
df = pd.read_csv(f"{OUT}/esp_scada_16wells_6months.csv", parse_dates=["timestamp"],
                 usecols=["well_id", "split", "timestamp", "state", "frequency_hz",
                          "Pi_intake_psi", "shutdown_event", "failure_cause"])

TEST_WELLS = sorted(df.loc[df["split"] == "test", "well_id"].unique())
print("Held-out wells:", TEST_WELLS)


# ============================================================
# 1. Fit each well's drawdown coefficient from its own early history
# ============================================================
def fit_drawdown(well_df):
    """Pi = a - k*f on normal-operation rows in the calibration window."""
    cutoff = well_df["timestamp"].min() + pd.Timedelta(days=CALIB_DAYS)
    w = well_df[(well_df["timestamp"] < cutoff) &
                (well_df["state"].isin(["normal", "ramp"])) &
                (well_df["frequency_hz"] > 0)]
    f = w["frequency_hz"].to_numpy(dtype=float)
    pi = w["Pi_intake_psi"].to_numpy(dtype=float)
    if len(f) < 500 or f.std() < 0.5:
        return None
    k_neg, a = np.polyfit(f, pi, 1)      # pi = k_neg*f + a, expect k_neg negative
    return {"a": float(a), "k": float(-k_neg), "n": len(f), "f_std": float(f.std())}


calib = {}
print("\n=== Per-well drawdown coefficient, fitted on each well's own first 30 days ===")
print(f"{'well':>5} {'k (psi/Hz)':>12} {'intercept a':>13} {'rows':>9} {'Hz spread':>10}")
for wid in TEST_WELLS:
    fit = fit_drawdown(df[df["well_id"] == wid])
    calib[wid] = fit
    if fit:
        print(f"{wid:>5} {fit['k']:>12.2f} {fit['a']:>13.0f} {fit['n']:>9,} {fit['f_std']:>10.2f}")
    else:
        print(f"{wid:>5}   not identifiable (insufficient frequency variation)")
print(f"\n(the simulator's true value is {sim.K_DRAWDOWN:.1f} psi/Hz - the fit only sees sensor data)")


# ============================================================
# 2. The recommendation itself
# ============================================================
def recommend_frequency(k, a, f_now, pi_now, safe_min, safe_max, bep):
    """Returns (recommended Hz, deficit psi, escalate flag, reason)."""
    pi_expected = a - k * f_now
    deficit = pi_expected - pi_now

    if deficit < DEFICIT_TRIGGER:
        # healthy - creep back toward best-efficiency point rather than sitting slow
        return min(f_now + RESTORE_STEP, bep), deficit, False, "healthy: restoring toward BEP"

    delta_f = (deficit + RECOVERY_MARGIN) / max(k, 1e-6)
    target = f_now - delta_f
    if target < safe_min:
        # even the equipment floor cannot recover the deficit
        return safe_min, deficit, True, "frequency alone insufficient - escalate to inspection"
    return target, deficit, False, f"reduce {delta_f:.1f} Hz to recover {deficit:.0f} psi"


def make_controller(well_id):
    fit = calib.get(well_id)
    if not fit:
        return None
    state = {"escalations": 0, "interventions": 0, "min_hz": None}

    def controller(well_id, minute, freq, pi, current, safe_min, safe_max, bep):
        rec, deficit, escalate, _ = recommend_frequency(
            fit["k"], fit["a"], freq, pi, safe_min, safe_max, bep)
        if deficit >= DEFICIT_TRIGGER:
            state["interventions"] += 1
            state["min_hz"] = rec if state["min_hz"] is None else min(state["min_hz"], rec)
        if escalate:
            state["escalations"] += 1
        return rec

    controller.state = state
    return controller


# ============================================================
# 3. Closed-loop back-test: same well, same seed, control off vs on
# ============================================================
def summarise(frame, events):
    running = frame[frame["state"].isin(["normal", "onset", "ramp"])]
    # Affinity law: rate is proportional to frequency, and zero while shut down,
    # so summed frequency over the whole period is a clean production proxy.
    production = float(frame["frequency_hz"].sum())
    causes = pd.Series([c for _, c in events]).value_counts().to_dict() if events else {}
    return {
        "trips_total": len(events),
        "trips_underload": int(causes.get("UNDERLOAD", 0)),
        "production_proxy": production,
        "downtime_min": int((frame["frequency_hz"] == 0).sum()),
        "mean_hz_running": float(running["frequency_hz"].mean()),
    }


rows = []
print("\n=== Closed-loop back-test (same seed, control OFF vs ON) ===")
for wid in TEST_WELLS:
    profile = sim.WELL_PROFILES[wid]["causes"]
    if "UNDERLOAD" not in profile:
        print(f"well {wid}: no underload in its cause profile - skipped")
        continue

    base_df, base_ev = sim.simulate_well(wid, profile, seed=1000 + wid)
    ctrl = make_controller(wid)
    ctrl_df, ctrl_ev = sim.simulate_well(wid, profile, seed=1000 + wid, controller=ctrl)

    b, c = summarise(base_df, base_ev), summarise(ctrl_df, ctrl_ev)
    prod_delta = 100.0 * (c["production_proxy"] / b["production_proxy"] - 1.0)
    rows.append({
        "well": wid,
        "underload_trips_before": b["trips_underload"],
        "underload_trips_after": c["trips_underload"],
        "all_trips_before": b["trips_total"],
        "all_trips_after": c["trips_total"],
        "downtime_hr_before": round(b["downtime_min"] / 60, 1),
        "downtime_hr_after": round(c["downtime_min"] / 60, 1),
        "mean_hz_before": round(b["mean_hz_running"], 2),
        "mean_hz_after": round(c["mean_hz_running"], 2),
        "production_change_pct": round(prod_delta, 2),
        "interventions": ctrl.state["interventions"],
        "escalations": ctrl.state["escalations"],
    })
    print(f"well {wid}: underload {b['trips_underload']} -> {c['trips_underload']} trips, "
          f"mean {b['mean_hz_running']:.2f} -> {c['mean_hz_running']:.2f} Hz, "
          f"production {prod_delta:+.2f}%")

res = pd.DataFrame(rows)

print("\n=== Result ===")
print(res.to_string(index=False))

tot_before = res["underload_trips_before"].sum()
tot_after = res["underload_trips_after"].sum()
dt_before = res["downtime_hr_before"].sum()
dt_after = res["downtime_hr_after"].sum()
prod = res["production_change_pct"].mean()

print(f"\nUnderload trips: {tot_before} -> {tot_after} "
      f"({100*(tot_before-tot_after)/max(tot_before,1):.0f}% avoided)")
print(f"Total downtime: {dt_before:.0f} h -> {dt_after:.0f} h ({dt_before-dt_after:+.0f} h)")
print(f"Mean production change across wells: {prod:+.2f}%")

with open(f"{RESULTS}/frequency_recommendation_results.txt", "w") as f:
    f.write("UNDERLOAD FREQUENCY RECOMMENDATION - closed-loop back-test\n")
    f.write("=" * 68 + "\n\n")
    f.write("Rule: fit Pi = a - k*f on each well's own first 30 days, then reduce\n")
    f.write("frequency by (deficit + margin)/k whenever intake pressure falls more than\n")
    f.write(f"{DEFICIT_TRIGGER:.0f} psi below what the current frequency should deliver.\n")
    f.write("Every recommendation is clipped to the well's own equipment-rated band; if\n")
    f.write("the floor binds, the well escalates to inspection instead.\n\n")
    f.write("Per-well drawdown coefficients fitted from sensor data only:\n")
    for wid in TEST_WELLS:
        if calib.get(wid):
            f.write(f"  well {wid}: k = {calib[wid]['k']:.2f} psi/Hz, a = {calib[wid]['a']:.0f} psi\n")
    f.write(f"  (simulator ground truth: {sim.K_DRAWDOWN:.1f} psi/Hz)\n\n")
    f.write("Closed-loop back-test, same wells and same random seeds:\n")
    f.write(res.to_string(index=False) + "\n\n")
    f.write(f"Underload trips: {tot_before} -> {tot_after} "
            f"({100*(tot_before-tot_after)/max(tot_before,1):.0f}% avoided)\n")
    f.write(f"Total downtime: {dt_before:.0f} h -> {dt_after:.0f} h ({dt_before-dt_after:+.0f} h)\n")
    f.write(f"Mean production change: {prod:+.2f}%\n")
print("\nSaved frequency_recommendation_results.txt")
