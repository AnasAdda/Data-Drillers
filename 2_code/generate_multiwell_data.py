"""
Multi-well ESP SCADA simulation, 6 months each, 1-minute resolution,
with MIXED failure causes across wells (not all wells face the same
failure mode) - matching the team's real-data plan.

Well roster (which wells exist, their cause profile, train/test split) is
read from wells_config.csv - the single shared config also used by
equipment_specs.py. To add a well (e.g. more data to test against later),
edit that file and re-run this script - see HOW_TO_ADD_A_WELL.md. This
script is INCREMENTAL: it only simulates well_ids that are in
wells_config.csv but not yet in the output CSV, then appends them, so
adding one well is fast - it does not re-simulate everything.

Failure causes modeled (each well is assigned 1-2 dominant causes,
so the dataset has a realistic mix, and some wells are "healthy/rare
failure" wells):

  1. GAS_LOCK        - gas interference at intake; LOW current, erratic
                        pressure; frequency-related (reducing Hz helps)
  2. UNDERLOAD       - pump off-design / low fluid supply; LOW current;
                        NOT frequency-related in the same way - often a
                        supply/inflow issue, not fixed by changing Hz
  3. HIGH_TEMP       - inadequate cooling / high ambient + high load;
                        motor temp climbs steadily; frequency-related
                        (running harder raises heat)
  4. HIGH_DISCHARGE  - downstream restriction (scale/choke); discharge
                        pressure climbs, current climbs; frequency-related
  5. VIBRATION       - mechanical wear / misalignment; vibration climbs
                        independent of frequency choice; NOT solved by
                        changing frequency alone (mechanical issue)
  6. LOW_VOLTAGE     - surface power supply sag; voltage drops, current
                        becomes unstable; NOT frequency-related at all -
                        an electrical supply problem; this one is an
                        abrupt external-grid event, not a gradual
                        degradation, so (realistically) it has little to
                        no leading indicator - see precursor leak below.

PRECURSOR LEAK (new): previously, sensors carried literally zero signal
before the state machine flipped from "normal" to "onset" - detection had
no genuine early-warning signal to find for a meaningful chunk of the
60-minute pre-trip window, which was part of why the detector's recall was
weak. GAS_LOCK, HIGH_TEMP, HIGH_DISCHARGE and VIBRATION each have a latent
degradation index that was already accumulating during "normal" state
(gas_accum, thermal_load, fouling_index, wear_index) but never leaked into
the sensors until onset officially began. Now a small, physically-motivated
fraction of each index leaks into its associated raw sensor throughout
"normal" state too, growing as the index approaches its trigger threshold -
representing the gradual degradation a real SCADA feed would actually show.
UNDERLOAD and LOW_VOLTAGE are sporadic/external triggers with no
accumulator, so they intentionally keep no precursor - that is a realistic
difference between a wearing-out failure and a sudden-onset one, not a bug.

Output: one CSV with a `well_id`, `split` (train/test) column, a
`failure_cause` label on the exact trip-minute row, and an `onset_cause`
label on every row while that well is actively in the "onset" state (the
clean, non-ambiguous signal Stage 2 root-cause classification trains on).
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

MINUTES_PER_DAY = 24 * 60
N_DAYS = 182
N = N_DAYS * MINUTES_PER_DAY

CAUSES = ["GAS_LOCK", "UNDERLOAD", "HIGH_TEMP", "HIGH_DISCHARGE", "VIBRATION", "LOW_VOLTAGE"]
FREQ_RELATED = {"GAS_LOCK": True, "UNDERLOAD": False, "HIGH_TEMP": True,
                 "HIGH_DISCHARGE": True, "VIBRATION": False, "LOW_VOLTAGE": False}

import os
# ---- where things live: the project's own data/ folder, unless ESP_DATA says otherwise ----
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ESP_DATA", os.path.join(os.path.dirname(HERE), "data"))
CONFIG_PATH = os.environ.get("ESP_CONFIG", os.path.join(HERE, "wells_config.csv"))
os.makedirs(DATA, exist_ok=True)
OUT_DIR = DATA
OUT_PATH = f"{OUT_DIR}/esp_scada_16wells_6months.csv"


def load_well_profiles(path=CONFIG_PATH):
    cfg = pd.read_csv(path)
    profiles = {}
    for _, row in cfg.iterrows():
        pairs = str(row["causes"]).split(";")
        causes = {}
        for p in pairs:
            name, weight = p.split(":")
            causes[name] = float(weight)
        profiles[int(row["well_id"])] = {"causes": causes, "split": row["split"]}
    return profiles


WELL_PROFILES = load_well_profiles()

P_RES = 2600.0
K_DRAWDOWN = 9.0
# Pump capacity per Hz (bbl/d/Hz), sized so that at a typical well's
# best-efficiency point the pump is roughly matched to what the reservoir can
# deliver. This is what makes "optimal frequency" a real question: past the
# inflow limit, extra pump speed produces nothing - it only pulls the fluid
# level down faster and brings the next pump-off trip forward.
C_PUMP = 15.0
K_BOOST = 13.5

# Loaded from well_equipment_specs.csv (run equipment_specs.py first) - each
# well's operating point and reduced-frequency restart point are now derived
# from ITS OWN equipment-rated safe band, not a shared global constant.
_specs = pd.read_csv(f"{OUT_DIR}/well_equipment_specs.csv").set_index("well_id")


def simulate_well(well_id, causes_profile, seed, controller=None):
    """controller(well_id, minute, freq, pi, current, safe_min, safe_max, bep) -> commanded Hz.

    Optional closed-loop frequency control. It sees only what a SCADA feed exposes -
    the current frequency and sensor readings - never the simulator's latent state.
    Used to back-test the underload frequency recommendation against the same well
    and the same random seed with control off.
    """
    local_rng = np.random.default_rng(seed)
    rows = []
    shutdown_events = []

    spec = _specs.loc[well_id]
    SAFE_MIN_HZ = float(spec["safe_min_hz"])
    SAFE_MAX_HZ = float(spec["safe_max_hz"])
    FREQ_BASE = float(spec["pump_bep_hz"])              # normal operating point = pump's best-efficiency Hz
    FREQ_RESTART = SAFE_MIN_HZ + 2.0                     # reduced-freq restart, just above the equipment floor
    gas_sep_efficiency = float(spec["gas_separator_efficiency_pct"])
    # weaker gas separator -> faster gas accumulation for wells prone to GAS_LOCK
    gas_lock_multiplier = (100.0 - gas_sep_efficiency) / 75.0

    reservoir_pressure = local_rng.uniform(1800, 3200)
    productivity_index = local_rng.uniform(0.8, 2.5)
    baseline_supply_voltage = local_rng.uniform(4100, 4200)

    state = "normal"
    freq = FREQ_BASE
    gas_accum = local_rng.uniform(0.05, 0.15)
    wear_index = local_rng.uniform(0.05, 0.15)      # mechanical wear (vibration cause)
    thermal_load = local_rng.uniform(0.05, 0.15)    # cumulative thermal stress (high temp cause)
    fouling_index = local_rng.uniform(0.05, 0.15)   # discharge restriction buildup
    drawdown_index = local_rng.uniform(0.05, 0.15)  # fluid level falling toward the pump (underload)
    onset_minutes = 0
    shutdown_minutes = 0
    shutdown_target = None
    ramp_minutes = 0
    ramp_duration = None
    current_cause = None
    commanded_freq = FREQ_BASE      # what a controller has asked for (BEP when uncontrolled)
    Tm, Ti = 178.0, 182.0

    def days_to_next_onset():
        return local_rng.uniform(3.0, 9.0) * MINUTES_PER_DAY

    accel_rate = 1.0 / days_to_next_onset()

    for i in range(N):
        cause = current_cause

        # ---------------- state machine ----------------
        if state == "normal":
            freq = commanded_freq
            gas_accum += accel_rate * gas_lock_multiplier * (freq / FREQ_BASE) ** 2 * local_rng.normal(1.0, 0.1) * 0.5
            wear_index += accel_rate * local_rng.normal(1.0, 0.15) * 0.35
            thermal_load += accel_rate * local_rng.normal(1.0, 0.15) * 0.35
            fouling_index += accel_rate * local_rng.normal(1.0, 0.15) * 0.35
            # Drawdown builds only while the pump outruns the well's inflow, so it must
            # depend on pump rate. Affinity law: Q proportional to frequency, so demand
            # scales with (f/f_bep)^2 against a fixed inflow capacity. At the BEP the
            # pump slightly outruns inflow and the level falls; roughly 10-15% slower and
            # the level recovers. Without this term, reducing frequency could not fix
            # underload in the simulation - which is the whole point of a pump-off control.
            _pump_demand = (freq / FREQ_BASE) ** 2
            drawdown_index += accel_rate * 0.35 * ((_pump_demand - 0.80) / 0.20) * local_rng.normal(1.0, 0.15)
            gas_accum, wear_index = max(gas_accum, 0), max(wear_index, 0)
            thermal_load, fouling_index = max(thermal_load, 0), max(fouling_index, 0)
            drawdown_index = max(drawdown_index, 0)

            voltage_sag = local_rng.random() < 0.0001  # rare abrupt supply sag event
            triggers = {
                "GAS_LOCK": gas_accum >= 1.0,
                # PHYSICS CORRECTION: underload was modelled as an instantaneous random
                # event, which made it unpredictable by construction (0-sigma precursor,
                # 6% catch rate). Real inflow starvation / pump-off develops gradually as
                # the fluid level over the pump falls - which is exactly why pump-off
                # controllers exist. It now accumulates like the other degradation modes.
                "UNDERLOAD": drawdown_index >= 1.0,
                "HIGH_TEMP": thermal_load >= 1.0,
                "HIGH_DISCHARGE": fouling_index >= 1.0,
                "VIBRATION": wear_index >= 1.0,
                "LOW_VOLTAGE": voltage_sag,
            }
            active = [c for c in CAUSES if triggers[c] and c in causes_profile]
            if active:
                weights = np.array([causes_profile[c] for c in active])
                current_cause = local_rng.choice(active, p=weights / weights.sum())
                state = "onset"
                onset_minutes = 0

        elif state == "onset":
            freq = commanded_freq
            onset_minutes += 1
            p_trip = min(0.02 * onset_minutes, 0.9)
            if local_rng.random() < p_trip:
                state = "shutdown"
                shutdown_minutes = 0
                shutdown_target = local_rng.uniform(90, 240)
                shutdown_events.append((i, current_cause))

        elif state == "shutdown":
            freq = 0.0
            shutdown_minutes += 1
            if shutdown_minutes >= shutdown_target:
                state = "ramp"
                ramp_minutes = 0
                ramp_duration = local_rng.uniform(300, 600)
                freq = FREQ_RESTART
                # reset the specific stressor that caused this trip
                if current_cause == "GAS_LOCK":
                    gas_accum = local_rng.uniform(0.05, 0.15)
                elif current_cause == "VIBRATION":
                    wear_index = local_rng.uniform(0.05, 0.15)
                elif current_cause == "HIGH_TEMP":
                    thermal_load = local_rng.uniform(0.05, 0.15)
                elif current_cause == "HIGH_DISCHARGE":
                    fouling_index = local_rng.uniform(0.05, 0.15)
                elif current_cause == "UNDERLOAD":
                    drawdown_index = local_rng.uniform(0.05, 0.15)
                accel_rate = 1.0 / days_to_next_onset()

        elif state == "ramp":
            ramp_minutes += 1
            frac = min(ramp_minutes / ramp_duration, 1.0)
            freq = FREQ_RESTART + frac * (commanded_freq - FREQ_RESTART)
            if ramp_minutes >= ramp_duration:
                state = "normal"
                current_cause = None

        onset_severity = min(onset_minutes / 40.0, 1.0) if state == "onset" else 0.0
        c = cause if state in ("onset", "shutdown") else None

        # ---------------- precursor leak (pre-onset, gradual causes only) ----------------
        # A small, physically-motivated fraction of each latent index leaks
        # into its sensor even during "normal" state, growing as the index
        # approaches its trigger threshold (>=1.0). Gives detection a real
        # early-warning signal instead of a flat baseline until the instant
        # onset officially begins. UNDERLOAD/LOW_VOLTAGE are sporadic/
        # external triggers with no accumulator - intentionally no precursor.
        precursor_current_dip = 0.05 * min(gas_accum, 1.3)              # GAS_LOCK precursor
        precursor_vibration = 0.10 * min(wear_index, 1.3)                # VIBRATION precursor
        precursor_tm = 3.0 * min(thermal_load, 1.3)                      # HIGH_TEMP precursor
        # Scale/fouling raises discharge pressure by tens of psi as it builds - the
        # previous 12 psi sat below the 15 psi sensor noise, i.e. 0.8 sigma, which is
        # why HIGH_DISCHARGE was effectively invisible (10% catch rate).
        precursor_pd = 55.0 * min(fouling_index, 1.3)                    # HIGH_DISCHARGE precursor
        # Falling fluid level: intake pressure declines and the starved pump draws less
        # current. Both are standard pump-off indicators on a real SCADA feed.
        precursor_pi_drop = 45.0 * min(drawdown_index, 1.3)              # UNDERLOAD precursor
        precursor_cur_sag = 0.04 * min(drawdown_index, 1.3)              # UNDERLOAD precursor

        # ---------------- sensor synthesis ----------------
        if state == "shutdown":
            current = max(local_rng.normal(0.2, 0.1), 0)
            voltage = max(local_rng.normal(30, 15), 0)
            vibration = max(local_rng.normal(0.02, 0.01), 0)
            pi = P_RES - 5 * (1 - min(shutdown_minutes / shutdown_target, 1.0)) + local_rng.normal(0, 8)
            pd_ = max(local_rng.normal(40, 10), 0)
            tm_target = 90 + 90 * (1 - min(shutdown_minutes / 200, 1.0))
            ti_target = 184 + 2 * min(shutdown_minutes / 200, 1.0)
        else:
            base_current = 29 + 0.52 * freq
            base_voltage = baseline_supply_voltage
            vib_base = 0.14 + 0.004 * (freq - 40) + 0.5 * wear_index if wear_index > 1 else 0.14 + 0.004 * (freq - 40)
            pi_penalty = 0.0
            pd_extra = 0.0
            efficiency = 1.0
            volt_drop = 0.0
            current_mult = 1.0

            if c == "GAS_LOCK":
                dip = onset_severity * rng.uniform(0.15, 0.45)
                current_mult *= (1 - dip)
                vibration_extra = onset_severity * rng.uniform(0.6, 1.8)
                pi_penalty = (max(gas_accum - 0.6, 0) ** 1.5) * 55 + onset_severity * rng.uniform(0, 90)
                efficiency = 1 - 0.55 * onset_severity * rng.uniform(0.7, 1.0)
            elif c == "UNDERLOAD":
                dip = onset_severity * rng.uniform(0.25, 0.55)
                current_mult *= (1 - dip)
                pi_penalty = onset_severity * rng.uniform(40, 120)
                efficiency = 1 - 0.3 * onset_severity
                vibration_extra = onset_severity * rng.uniform(0.05, 0.2)
            elif c == "HIGH_TEMP":
                vibration_extra = onset_severity * rng.uniform(0.1, 0.3)
                current_mult *= (1 + 0.05 * onset_severity)
                efficiency = 1 - 0.1 * onset_severity
            elif c == "HIGH_DISCHARGE":
                pd_extra = onset_severity * rng.uniform(150, 400) + fouling_index * 30 if fouling_index > 1 else onset_severity * rng.uniform(150, 400)
                current_mult *= (1 + 0.15 * onset_severity)
                vibration_extra = onset_severity * rng.uniform(0.1, 0.3)
                efficiency = 1 - 0.2 * onset_severity
            elif c == "VIBRATION":
                vibration_extra = onset_severity * rng.uniform(1.0, 2.5) + max(wear_index - 1, 0) * 0.8
                current_mult *= (1 + 0.03 * onset_severity)
                efficiency = 1 - 0.1 * onset_severity
            elif c == "LOW_VOLTAGE":
                volt_drop = onset_severity * rng.uniform(300, 900)
                current_mult *= (1 + 0.35 * onset_severity)  # current climbs to compensate
                vibration_extra = onset_severity * rng.uniform(0.05, 0.15)
                efficiency = 1 - 0.15 * onset_severity
            else:
                vibration_extra = 0.0

            # precursor leak applies whenever the well is actually running
            # (normal/onset/ramp), on top of any onset-specific effect above
            current_mult *= (1 - precursor_current_dip) * (1 - precursor_cur_sag)
            vibration_extra += precursor_vibration
            pd_extra += precursor_pd

            current = base_current * current_mult + local_rng.normal(0, 0.9)
            voltage = base_voltage - volt_drop + local_rng.normal(0, 12)
            vibration = max(vib_base + vibration_extra + local_rng.normal(0, 0.05), 0)

            pi = P_RES - K_DRAWDOWN * freq - pi_penalty - precursor_pi_drop + local_rng.normal(0, 12)
            pd_ = pi + K_BOOST * freq * efficiency + pd_extra + local_rng.normal(0, 15)

            tm_target = (165 + 0.85 * freq + onset_severity * (25 if c == "HIGH_TEMP" else 10)
                         + max(thermal_load - 1, 0) * 15 + precursor_tm)
            ti_target = 182 + onset_severity * 3 + local_rng.normal(0, 0.3)

        Tm += (tm_target - Tm) * 0.02 + local_rng.normal(0, 0.15)
        Ti += (ti_target - Ti) * 0.05 + local_rng.normal(0, 0.1)

        # ---------------- production ----------------
        # Inflow performance: what the reservoir can deliver at this drawdown.
        # Pump capacity: affinity law, proportional to speed.
        # Actual rate is the lesser of the two - a pump cannot produce fluid the
        # reservoir does not supply.
        if state == "shutdown":
            production_bpd = 0.0
        else:
            q_pump = C_PUMP * freq
            q_inflow = productivity_index * max(P_RES - pi, 0.0)
            production_bpd = min(q_pump, q_inflow)

        if controller is not None and state in ("normal", "onset"):
            commanded_freq = float(np.clip(
                controller(well_id=well_id, minute=i, freq=freq, pi=pi, current=current,
                           safe_min=SAFE_MIN_HZ, safe_max=SAFE_MAX_HZ, bep=FREQ_BASE),
                SAFE_MIN_HZ, SAFE_MAX_HZ))

        rows.append((
            well_id, i, state, c if c else "", round(freq, 2), round(current, 2), round(voltage, 1),
            round(Ti, 2), round(Tm, 2), round(pi, 1), round(pd_, 1), round(vibration, 3),
            round(production_bpd, 1),
        ))

    df = pd.DataFrame(rows, columns=[
        "well_id", "minute", "state", "onset_cause", "frequency_hz", "motor_current_a", "voltage_v",
        "Ti_intake_temp_f", "Tm_motor_temp_f", "Pi_intake_psi", "Pd_discharge_psi", "vibration_g",
        "production_bpd",
    ])
    df["shutdown_event"] = 0
    df["failure_cause"] = ""
    for idx, cause in shutdown_events:
        df.loc[idx, "shutdown_event"] = 1
        df.loc[idx, "failure_cause"] = cause
    df["safe_min_hz"] = SAFE_MIN_HZ
    df["safe_max_hz"] = SAFE_MAX_HZ
    return df, shutdown_events


# ============================================================
# Incremental generation: only simulate wells missing from the output CSV
# ============================================================
existing_well_ids = set()
existing_df = None
if os.path.exists(OUT_PATH):
    existing_df = pd.read_csv(OUT_PATH, usecols=["well_id"])
    existing_well_ids = set(existing_df["well_id"].unique().tolist())
    del existing_df

wells_to_simulate = {wid: p for wid, p in WELL_PROFILES.items() if wid not in existing_well_ids}

if not wells_to_simulate:
    print("No new wells to simulate - wells_config.csv already fully generated in", OUT_PATH)
else:
    print(f"Simulating {len(wells_to_simulate)} well(s) not yet in the dataset: {sorted(wells_to_simulate)}")
    new_dfs = []
    summary = []
    for well_id, profile in wells_to_simulate.items():
        df_well, events = simulate_well(well_id, profile["causes"], seed=1000 + well_id)
        df_well["split"] = profile["split"]
        start = pd.Timestamp("2026-01-01 00:00:00")
        df_well["timestamp"] = start + pd.to_timedelta(df_well["minute"], unit="m")
        new_dfs.append(df_well)
        cause_counts = pd.Series([c for _, c in events]).value_counts().to_dict()
        summary.append({"well_id": well_id, "split": profile["split"], "n_trips": len(events), **cause_counts})
        print(f"well {well_id:2d} [{profile['split']}] causes={list(profile['causes'].keys())}: {len(events)} trips -> {cause_counts}")

    new_full = pd.concat(new_dfs, ignore_index=True)
    new_full = new_full[[
        "well_id", "split", "timestamp", "state", "frequency_hz", "motor_current_a", "voltage_v",
        "Ti_intake_temp_f", "Tm_motor_temp_f", "Pi_intake_psi", "Pd_discharge_psi", "vibration_g",
        "shutdown_event", "failure_cause", "onset_cause", "safe_min_hz", "safe_max_hz",
        "production_bpd",
    ]]

    # QA: confirm no simulated frequency ever violated its own well's safe band
    violations = new_full[(new_full["frequency_hz"] > 0) &
                           ((new_full["frequency_hz"] < new_full["safe_min_hz"] - 0.01) |
                            (new_full["frequency_hz"] > new_full["safe_max_hz"] + 0.01))]
    print(f"\nSafe-band QA (new wells): {len(violations)} rows violate the well's equipment-rated Hz band "
          f"(0 expected; freq=0 rows during shutdown are excluded from this check).")

    if os.path.exists(OUT_PATH):
        new_full.to_csv(OUT_PATH, mode="a", header=False, index=False)
        print(f"\nAppended {len(new_full):,} rows to existing:", OUT_PATH)
    else:
        new_full.to_csv(OUT_PATH, index=False)
        print(f"\nSaved {len(new_full):,} rows to new file:", OUT_PATH)

    print("\nSummary (new wells only):")
    print(pd.DataFrame(summary).fillna(0))

full = pd.read_csv(OUT_PATH, usecols=["well_id", "split", "shutdown_event"])
print("\nFull dataset now covers", full["well_id"].nunique(), "wells,", len(full), "rows")
print(full.groupby(["well_id", "split"])["shutdown_event"].sum().rename("n_trips"))
