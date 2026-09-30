# Adding a well (e.g. new/future test data)

All 16 wells are defined in one place: `wells_config.csv`. Nothing else needs
to change to add a well.

## 1. Add a row to `wells_config.csv`

```
well_id,causes,split
17,HIGH_TEMP:1.0,test
```

- `well_id`: any unused integer.
- `causes`: one or more `CAUSE:weight` pairs separated by `;`. Weights don't
  need to sum to 1 (they're normalized automatically). Valid causes:
  `GAS_LOCK`, `UNDERLOAD`, `HIGH_TEMP`, `HIGH_DISCHARGE`, `VIBRATION`,
  `LOW_VOLTAGE`.
- `split`: `train` or `test`. For "add more data to test against later",
  use `test` — the well is simulated but the model never trains on it.

## 2. Regenerate equipment specs (fast, all wells)

```
python equipment_specs.py
```

This re-derives every well's safe Hz band (Motor ∩ Protector ∩ Pump), so the
new well gets a valid band automatically.

## 3. Generate SCADA data for the new well only

```
python generate_multiwell_data.py
```

This script is incremental: it checks which `well_id`s are already in
`data/multiwell_scada_10wells_6mo.csv` and only simulates the ones that are
missing, then appends them. Adding one well takes a fraction of the time of
a full regeneration.

## 4. Re-run the analysis

```
python run_full_analysis.py
```

Train/test wells are read directly from the `split` column, so a new
`test`-split well is automatically included in held-out evaluation with no
other code changes — this is the hook for testing the model against future
data (synthetic or, later, real historical well data) without touching the
modeling code at all.

## Using real historical data instead of synthetic

Point `generate_multiwell_data.py`'s output path at a CSV with the same
schema (`well_id, split, timestamp, state, frequency_hz, motor_current_a,
voltage_v, Ti_intake_temp_f, Tm_motor_temp_f, Pi_intake_psi,
Pd_discharge_psi, vibration_g, shutdown_event, failure_cause, onset_cause,
safe_min_hz, safe_max_hz`) for the real well(s), append/concat it into
`data/multiwell_scada_10wells_6mo.csv`, add a matching row to `wells_config.csv`
(so equipment specs and train/test bookkeeping stay consistent), and run
step 4. No model code changes needed either way.
