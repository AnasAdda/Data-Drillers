"""
ESP equipment configuration per well: Motor, Protector, Gas Separator, Pump.

These are REPRESENTATIVE/TYPICAL specs for a training exercise (not pulled
from a real manufacturer catalog) - realistic in shape and magnitude for a
standard 5.13"-housing ESP string, used to derive a per-well SAFE OPERATING
FREQUENCY BAND (Min Hz / Max Hz), per the instructor's feedback that the
recommended frequency must respect real equipment limits, not just look
statistically optimal.

Safe band logic (intersection of component-rated ranges):
  - Motor: nameplate rated Hz (60), plus a manufacturer-derated max Hz
    (running faster than this risks winding/bearing overheat) and a min Hz
    (below this, motor cooling by fluid flow-by becomes inadequate).
  - Protector: thrust-bearing capacity implies a max Hz (higher Hz -> more
    axial thrust from the pump stages); expressed here directly as a max Hz.
  - Pump: manufacturer curve valid range (min/max Hz the pump stages are
    rated for) - going outside it risks upthrust/downthrust damage.
  - Gas Separator: type + efficiency does not change the Hz band directly,
    but wells with a less effective (or no) gas separator are more
    susceptible to gas lock at a given drawdown - noted per well.

Well Safe Hz Band = [max of all component mins, min of all component maxs]

Well roster (well_id, cause profile, train/test split) now lives in
wells_config.csv, the single shared source of truth for both this script
and generate_multiwell_data.py. To add a well, edit that file - see
HOW_TO_ADD_A_WELL.md.
"""
import numpy as np
import pandas as pd
import os
# ---- where things live: the project's own data/ folder, unless ESP_DATA says otherwise ----
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ESP_DATA", os.path.join(os.path.dirname(HERE), "data"))
CONFIG_PATH = os.environ.get("ESP_CONFIG", os.path.join(HERE, "wells_config.csv"))
os.makedirs(DATA, exist_ok=True)

rng = np.random.default_rng(77)

MOTOR_CATALOG = [
    # (model, rated_hp, rated_v, rated_a, min_hz, max_hz)
    ("HS 456 Series 562 Motor", 456, 2450, 68, 35, 70),
    ("HS 562 Series 562 Motor", 562, 2790, 72, 35, 75),
    ("HD 375 Series 456 Motor", 375, 2100, 61, 30, 68),
    ("HT 500 Series 562 Motor", 500, 2600, 70, 35, 72),
]

PROTECTOR_CATALOG = [
    # (model, thrust_rating_lb, max_hz)
    ("Tandem Protector - Labyrinth/Bag, 2500 lb", 2500, 68),
    ("Tandem Protector - Positive Seal, 3200 lb", 3200, 72),
    ("Single Protector - Labyrinth, 1800 lb", 1800, 65),
]

GAS_SEPARATOR_CATALOG = [
    # (model, type, efficiency_pct)
    ("Rotary Gas Separator, RGS-538", "Rotary", 85),
    ("Reverse-Flow Gas Separator, RFS-400", "Reverse-Flow", 65),
    ("None (natural separation only)", "None", 25),
]

PUMP_CATALOG = [
    # (model, stage_type, n_stages, min_hz, max_hz, bep_hz)
    ("Mixed-Flow Pump, GC-2700, 92 stages", "Mixed-Flow", 92, 40, 65, 52),
    ("Mixed-Flow Pump, GC-3300, 78 stages", "Mixed-Flow", 78, 38, 67, 54),
    ("Radial-Flow Pump, D-1750, 130 stages", "Radial-Flow", 130, 35, 63, 50),
    ("Mixed-Flow Pump, GC-4000, 65 stages", "Mixed-Flow", 65, 40, 70, 55),
]


def load_wells_config(path=CONFIG_PATH):
    cfg = pd.read_csv(path)
    causes_by_well = {}
    for _, row in cfg.iterrows():
        pairs = str(row["causes"]).split(";")
        causes_by_well[int(row["well_id"])] = [p.split(":")[0] for p in pairs]
    return causes_by_well


WELL_CAUSES = load_wells_config()

rows = []
for well_id, causes in WELL_CAUSES.items():
    motor = MOTOR_CATALOG[rng.integers(len(MOTOR_CATALOG))]
    protector = PROTECTOR_CATALOG[rng.integers(len(PROTECTOR_CATALOG))]
    pump = PUMP_CATALOG[rng.integers(len(PUMP_CATALOG))]

    # Wells with GAS_LOCK in their cause profile are deliberately given a
    # weaker gas separator (or none) - this is WHY they're gas-lock-prone,
    # tying the equipment config to the failure mode rather than being arbitrary
    if "GAS_LOCK" in causes:
        gas_sep = GAS_SEPARATOR_CATALOG[rng.choice([1, 2])]  # Reverse-Flow or None
    else:
        gas_sep = GAS_SEPARATOR_CATALOG[rng.choice([0, 1], p=[0.7, 0.3])]  # mostly good Rotary

    min_hz = max(motor[4], pump[3])
    max_hz = min(motor[5], protector[2], pump[4])

    rows.append({
        "well_id": well_id,
        "motor_model": motor[0], "motor_rated_hp": motor[1], "motor_rated_v": motor[2], "motor_rated_a": motor[3],
        "motor_min_hz": motor[4], "motor_max_hz": motor[5],
        "protector_model": protector[0], "protector_thrust_rating_lb": protector[1], "protector_max_hz": protector[2],
        "gas_separator_model": gas_sep[0], "gas_separator_type": gas_sep[1], "gas_separator_efficiency_pct": gas_sep[2],
        "pump_model": pump[0], "pump_stage_type": pump[1], "pump_n_stages": pump[2],
        "pump_min_hz": pump[3], "pump_max_hz": pump[4], "pump_bep_hz": pump[5],
        "safe_min_hz": min_hz, "safe_max_hz": max_hz,
        "dominant_causes": ", ".join(causes),
    })

specs = pd.DataFrame(rows).sort_values("well_id").reset_index(drop=True)

# sanity check: every well needs a valid non-empty band, and BEP should fall inside it
assert (specs["safe_min_hz"] < specs["safe_max_hz"]).all(), "invalid band for some well"
specs["bep_in_band"] = specs.apply(lambda r: r["pump_bep_hz"] >= r["safe_min_hz"] and r["pump_bep_hz"] <= r["safe_max_hz"], axis=1)
print(specs[["well_id", "safe_min_hz", "safe_max_hz", "pump_bep_hz", "bep_in_band", "gas_separator_type", "gas_separator_efficiency_pct"]].to_string(index=False))

specs.to_csv(f"{DATA}/well_equipment_specs.csv", index=False)
print(f"\nSaved well_equipment_specs.csv ({len(specs)} wells)")
