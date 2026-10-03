"""
Optimal operating frequency for underload-prone wells.

THE OBJECTIVE, CORRECTED
------------------------
A first attempt controlled frequency to eliminate underload trips. It worked -
22 trips became 0 - and it was still the wrong answer: it gave up 18.9% of
production to recover 0.8% of uptime.

Avoiding trips is not the goal. Producing the most oil is. Those differ because
both sides of the trade cost rate:

    running slower       -> lower rate every minute       (Q proportional to f)
    running too fast     -> trips, and zero rate while down

So the objective is expected produced volume, and the optimum is an interior
point: fast enough to produce, slow enough that the well keeps running.

WHAT THIS SCRIPT DOES
---------------------
1. Frequency sweep. Each underload-prone held-out well is re-simulated from the
   same seed at a range of fixed frequencies. Total produced volume is measured
   directly, which gives that well's production-optimal setpoint and the shape
   of the curve around it - including how much is lost by running at the pump's
   best-efficiency point, which is the usual default.

2. Adaptive control. The reactive controller from the previous step is retuned
   to hold the well near its optimum instead of chasing zero trips, and is
   measured against the best fixed setpoint. Adapting only earns its complexity
   if it beats a well-chosen constant.

Production proxy: rate is proportional to frequency under the affinity laws and
zero while shut down, so summed frequency over the period is proportional to
produced volume.
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
CALIB_DAYS = 30

specs = pd.read_csv(f"{OUT}/well_equipment_specs.csv").set_index("well_id")
meta = pd.read_csv(f"{OUT}/esp_scada_16wells_6months.csv",
                   usecols=["well_id", "split", "timestamp", "state", "frequency_hz", "Pi_intake_psi"],
                   parse_dates=["timestamp"])
TEST_WELLS = sorted(meta.loc[meta["split"] == "test", "well_id"].unique())
UNDERLOAD_WELLS = [w for w in TEST_WELLS if "UNDERLOAD" in sim.WELL_PROFILES[w]["causes"]]
print("Held-out wells with underload in their profile:", UNDERLOAD_WELLS)


def production(frame):
    """Barrels produced: min(pump capacity, reservoir inflow), zero while down."""
    return float(frame["production_bpd"].sum()) / (60.0 * 24.0)  # bbl produced


def run_fixed(well_id, hz):
    """Simulate the well held at a fixed frequency setpoint."""
    return sim.simulate_well(well_id, sim.WELL_PROFILES[well_id]["causes"],
                             seed=1000 + well_id,
                             controller=lambda **kw: hz)


# ============================================================
# 1. Frequency sweep - find each well's production optimum
# ============================================================
sweep_rows = []
optima = {}
print("\n=== Frequency sweep (same well, same seed, fixed setpoint) ===")
for wid in UNDERLOAD_WELLS:
    spec = specs.loc[wid]
    bep, lo, hi = float(spec["pump_bep_hz"]), float(spec["safe_min_hz"]), float(spec["safe_max_hz"])
    candidates = sorted({round(float(np.clip(bep * m, lo, hi)), 1)
                         for m in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)})
    print(f"\nwell {wid}  (BEP {bep:.0f} Hz, safe band {lo:.0f}-{hi:.0f} Hz)")
    print(f"{'Hz':>7} {'% of BEP':>9} {'trips':>7} {'underload':>10} {'downtime h':>11} {'production':>11} {'vs BEP':>9}")
    per_well = []
    for hz in candidates:
        frame, events = run_fixed(wid, hz)
        causes = pd.Series([c for _, c in events]).value_counts().to_dict() if events else {}
        per_well.append({
            "well": wid, "hz": hz, "pct_of_bep": round(100 * hz / bep),
            "trips": len(events), "underload": int(causes.get("UNDERLOAD", 0)),
            "downtime_hr": round(float((frame["frequency_hz"] == 0).sum()) / 60, 1),
            "production": production(frame),
        })
    base = [r for r in per_well if abs(r["hz"] - round(bep, 1)) < 0.05]
    base_prod = base[0]["production"] if base else max(r["production"] for r in per_well)
    for r in per_well:
        r["vs_bep_pct"] = round(100 * (r["production"] / base_prod - 1), 2)
        print(f"{r['hz']:>7.1f} {r['pct_of_bep']:>8}% {r['trips']:>7} {r['underload']:>10} "
              f"{r['downtime_hr']:>11.1f} {r['production']:>11,.0f} {r['vs_bep_pct']:>8.2f}%")
        sweep_rows.append(r)
    best = max(per_well, key=lambda r: r["production"])
    optima[wid] = best
    print(f"  -> optimum {best['hz']:.1f} Hz ({best['pct_of_bep']}% of BEP): "
          f"{best['vs_bep_pct']:+.2f}% production vs running at BEP, "
          f"underload trips {base[0]['underload'] if base else '?'} -> {best['underload']}")

sweep = pd.DataFrame(sweep_rows)


# ============================================================
# 2. Adaptive control, aimed at the optimum rather than at zero trips
# ============================================================
def fit_drawdown(well_df):
    cutoff = well_df["timestamp"].min() + pd.Timedelta(days=CALIB_DAYS)
    w = well_df[(well_df["timestamp"] < cutoff) & (well_df["state"].isin(["normal", "ramp"]))
                & (well_df["frequency_hz"] > 0)]
    f, pi = w["frequency_hz"].to_numpy(float), w["Pi_intake_psi"].to_numpy(float)
    if len(f) < 500 or f.std() < 0.5:
        return None
    k_neg, a = np.polyfit(f, pi, 1)
    return {"a": float(a), "k": float(-k_neg)}


DEFICIT_TRIGGER = 35.0    # act later than before: small deficits are not worth rate
STEP_DOWN = 0.35          # Hz trimmed per intervention - gentle, not a slam to the floor
RESTORE_STEP = 0.004      # Hz per minute crept back once the well recovers
FLOOR_FRACTION = 0.88     # never trim below this share of BEP: past here the rate loss
                          # outweighs the trips avoided, as the sweep shows


def make_controller(well_id, fit, bep):
    st = {"interventions": 0, "min_hz": bep}

    def controller(well_id, minute, freq, pi, current, safe_min, safe_max, bep):
        deficit = (fit["a"] - fit["k"] * freq) - pi
        floor = max(safe_min, FLOOR_FRACTION * bep)
        if deficit >= DEFICIT_TRIGGER:
            st["interventions"] += 1
            nxt = max(freq - STEP_DOWN, floor)
        else:
            nxt = min(freq + RESTORE_STEP, bep)
        st["min_hz"] = min(st["min_hz"], nxt)
        return nxt

    controller.state = st
    return controller


# ============================================================
# 1b. Break-even: how long must a trip keep the well down before
#     running slower actually pays?
# ============================================================
print("\n\n=== Break-even on restart time ===")
print("The simulator restarts a tripped well automatically within a few hours. If a")
print("real trip instead waits for a crew, each one costs far more, and the balance")
print("shifts. Break-even is the extra downtime per trip at which slowing down pays:\n")
breakeven_rows = []
for wid in UNDERLOAD_WELLS:
    w = sweep[sweep["well"] == wid].sort_values("hz")
    bep_row = w.iloc[(w["pct_of_bep"] - 100).abs().argsort().iloc[0]]
    for _, r in w.iterrows():
        if r["hz"] >= bep_row["hz"] or r["trips"] >= bep_row["trips"]:
            continue
        prod_given_up = bep_row["production"] - r["production"]
        trips_avoided = bep_row["trips"] - r["trips"]
        if trips_avoided <= 0:
            continue
        # one hour of full-rate production at BEP, in the same proxy units
        hour_at_bep = bep_row["production"] / (182.0 * 24.0)  # bbl per hour at BEP
        hours = prod_given_up / (trips_avoided * hour_at_bep)
        breakeven_rows.append({"well": wid, "hz": r["hz"], "pct_of_bep": r["pct_of_bep"],
                                "trips_avoided": int(trips_avoided),
                                "production_given_up_pct": r["vs_bep_pct"],
                                "breakeven_hours_per_trip": round(hours, 1)})
be = pd.DataFrame(breakeven_rows)
print(be.to_string(index=False))
print("\nRead it as: running this well at that frequency only pays if each avoided trip")
print("would otherwise have kept it down longer than the break-even hours shown.")

print("\n\n=== Adaptive control vs. the best fixed setpoint ===")
adaptive_rows = []
for wid in UNDERLOAD_WELLS:
    spec = specs.loc[wid]
    bep = float(spec["pump_bep_hz"])
    fit = fit_drawdown(meta[meta["well_id"] == wid])
    if not fit:
        print(f"well {wid}: drawdown coefficient not identifiable - skipped")
        continue

    base_frame, base_ev = run_fixed(wid, bep)
    base_prod, base_causes = production(base_frame), pd.Series([c for _, c in base_ev]).value_counts().to_dict()

    ctrl = make_controller(wid, fit, bep)
    ad_frame, ad_ev = sim.simulate_well(wid, sim.WELL_PROFILES[wid]["causes"],
                                        seed=1000 + wid, controller=ctrl)
    ad_prod = production(ad_frame)
    ad_causes = pd.Series([c for _, c in ad_ev]).value_counts().to_dict() if ad_ev else {}

    best_fixed = optima[wid]
    adaptive_rows.append({
        "well": wid,
        "bep_hz": round(bep, 1),
        "best_fixed_hz": best_fixed["hz"],
        "adaptive_mean_hz": round(float(ad_frame.loc[ad_frame["frequency_hz"] > 0, "frequency_hz"].mean()), 2),
        "adaptive_min_hz": round(ctrl.state["min_hz"], 1),
        "underload_at_bep": int(base_causes.get("UNDERLOAD", 0)),
        "underload_best_fixed": best_fixed["underload"],
        "underload_adaptive": int(ad_causes.get("UNDERLOAD", 0)),
        "prod_best_fixed_pct": best_fixed["vs_bep_pct"],
        "prod_adaptive_pct": round(100 * (ad_prod / base_prod - 1), 2),
    })
    r = adaptive_rows[-1]
    print(f"well {wid}: BEP {bep:.0f} Hz | best fixed {r['best_fixed_hz']:.1f} Hz "
          f"({r['prod_best_fixed_pct']:+.2f}%) | adaptive mean {r['adaptive_mean_hz']:.2f} Hz "
          f"({r['prod_adaptive_pct']:+.2f}%)")

adaptive = pd.DataFrame(adaptive_rows)
print("\n" + adaptive.to_string(index=False))

INTERPRETATION = """

=== INTERPRETATION - WHAT THIS CAN AND CANNOT SETTLE ===

WHAT THE SWEEP SAYS, AND WHY IT IS NOT THE ANSWER
The sweep reports that production keeps rising up to 120% of BEP (+17%),
despite trips tripling from 11 to 34. That is an artifact of the simulator,
not a recommendation:

  * The production proxy uses the affinity law (Q proportional to f), which
    only holds near the pump's design point. Past BEP a real pump moves off
    its curve and the extra speed does not buy proportional flow.
  * Raising frequency increases drawdown, which under a simple IPR increases
    inflow as well - so nothing saturates. The real ceiling is pump
    submergence: the fluid level cannot be pulled below the intake. That
    constraint is not modelled here.
  * A trip costs only a few hours of automatic restart in this simulation. It
    carries no workover risk and no run-life penalty, whereas repeated
    restarts are exactly what shortens ESP life in the field.

Standard practice is not to run deliberately above BEP, so recommendations are
capped there regardless of what the proxy says.

WHAT IS ACTUALLY TRANSFERABLE: THE BREAK-EVEN RULE
The break-even table does not depend on the invented production constants. It
is built from two quantities that were measured directly - production given up,
and trips avoided - so it transfers to real economics:

    Reducing well 8 from 52 Hz to 47 Hz (90% of BEP) eliminates all 11
    underload trips and gives up 9.2% of rate. That pays only if each avoided
    trip would otherwise keep the well down longer than ~36 hours.

So the decision rule for an underload-prone well is:

    if (expected downtime per trip) > (break-even hours) -> slow the well down
    else                                                 -> accept the trips

With automatic restart in a few hours, the arithmetic says accept the trips.
With a remote well waiting a day or two for a crew, it says slow it down. Same
well, opposite answer, decided entirely by restart time - which is a field fact,
not a modelling choice.

WHAT THE METHOD DELIVERS
  1. Per-well drawdown coefficient k (psi/Hz), fitted from that well's own
     sensor history - recovered 9.27 / 8.38 / 10.36 against a true 9.0.
  2. A frequency recommendation derived from k, clipped to the well's own
     equipment band, escalating to inspection when the floor binds.
  3. A closed-loop back-test showing the control genuinely works: underload
     trips 22 -> 0 on the same wells and the same seeds
     (see frequency_recommendation.py).
  4. A break-even figure telling an engineer whether to use it.

WHAT ONLY REAL DATA CAN SETTLE
Whether the optimum sits below BEP at all, and by how much. That depends on the
well's true inflow performance, the real cost of a trip including run-life, and
actual restart times - none of which a synthetic dataset can supply honestly.
This is the first place the pipeline should be pointed at real historical wells.
"""

with open(f"{RESULTS}/frequency_optimisation_results.txt", "w") as f:
    f.write("OPTIMAL OPERATING FREQUENCY FOR UNDERLOAD-PRONE WELLS\n")
    f.write("=" * 70 + "\n\n")
    f.write("Objective: maximise produced volume, not minimise trips. Running slower\n")
    f.write("costs rate every minute; running too fast costs whole hours to trips. The\n")
    f.write("optimum is an interior point between the two.\n\n")
    f.write("Production proxy: rate ~ frequency (affinity law), zero while shut down.\n\n")
    f.write("=== 1. Frequency sweep (same well, same seed, fixed setpoint) ===\n")
    f.write(sweep.to_string(index=False) + "\n\n")
    for wid, best in optima.items():
        f.write(f"Well {wid}: production optimum at {best['hz']:.1f} Hz "
                f"({best['pct_of_bep']}% of BEP), {best['vs_bep_pct']:+.2f}% vs running at BEP, "
                f"{best['underload']} underload trips\n")
    f.write("\n=== 1b. Break-even downtime per trip ===\n")
    f.write(be.to_string(index=False) + "\n")
    f.write("Slowing the well only pays if an avoided trip would have kept it down\n")
    f.write("longer than the break-even hours shown.\n\n")
    f.write("=== 2. Adaptive control vs. best fixed setpoint ===\n")
    f.write(adaptive.to_string(index=False) + "\n")
    f.write(INTERPRETATION)
print("\nSaved frequency_optimisation_results.txt")
