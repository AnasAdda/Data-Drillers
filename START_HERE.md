# ESP Failure Prediction & Optimal Operation Frequency — Code and Data

Samsung Innovation Campus, AI Capstone · Group-3 "Data Drillers"

This package holds the complete code, the dataset, and the full experiment: the results, the
charts, the trained models and the live dashboard produced by running the code.

---

## Start here

| To… | Open |
|---|---|
| Run the whole project | `1_notebooks/ESP_Capstone_Full_Pipeline.ipynb` |
| See the live demo | `5_dashboard/ESP_Live_Dashboard.html` → press **Guided demo** (no code, no internet) |
| Check the results | `3_results/results.txt` (detection and root cause), `3_results/frequency_*.txt` |
| See the charts | `4_charts/` |
| Read the project overview | `README.md` |

## Headline results (held-out wells 8, 9, 10: never used in training)

| | |
|---|---|
| Trips caught before they happened | **33 of 69 (48%)**, 31 with ≥ 30 min warning |
| Median warning time | **180 min** |
| Nuisance alarms | **18.7** per well per month |
| Root-cause accuracy | **68.6%** |
| Underload trips under frequency control | 22 → 0 (pays only if a trip costs > ~34 h of downtime) |
| Live dashboard, new well 17 | 9 of 17 trips warned, median 180 min |

---

## Running the code

### Option 1: Google Colab (~16 min, no installation)
1. Upload `1_notebooks/ESP_Capstone_Full_Pipeline.ipynb` to Google Colab.
2. **Runtime → Run all.**

On Colab the notebook generates the dataset itself in Step 3. The simulator is seeded, so the
generated file is identical to `data/esp_scada_16wells_6months.csv`.

### Option 2: your own computer (Python 3.12)
Open the notebook from inside this folder (VS Code or Jupyter) and run all cells. It uses `data/`,
which already holds the dataset and the trained models, so the 5-minute data generation is skipped.

Or run the scripts:
```
pip install -r requirements.txt
python 2_code/run_full_analysis.py            # features, clustering, Stage 1 + 2, alarm policy, charts, saves models
python 2_code/frequency_recommendation.py     # per-well drawdown fit + closed-loop back-test
python 2_code/frequency_optimisation.py       # frequency sweep + break-even
python 2_code/score_new_well.py --verify      # re-scores wells 8-10 with the saved models
python 5_dashboard/build_dashboard.py --well 17 --breakeven 34.7   # rebuilds the live dashboard
```
To regenerate the dataset from scratch, delete `data/esp_scada_16wells_6months.csv` and run
`python 2_code/generate_multiwell_data.py` (~5 min).

Local runs write to `4_charts/latest_run/` and `3_results/latest_run/`, so they never overwrite
the official results.

---

## Contents

```
START_HERE.md          This file
README.md              Project overview: scope, key finding, what was tried and rejected, limitations
requirements.txt       Python packages

1_notebooks/           ESP_Capstone_Full_Pipeline.ipynb — the whole pipeline in one notebook (Steps 0–8)
                       ESP_New_Well_Demo_Colab.ipynb — short Colab demo: trains, freezes the models,
                       then scores a brand-new well (17) without retraining
2_code/                The same pipeline as scripts:
  wells_config.csv               well roster (16 wells, causes, train/test split)
  equipment_specs.py             ESP equipment per well and its safe frequency band
  generate_multiwell_data.py     physics-inspired SCADA simulator
  run_full_analysis.py           main pipeline (44 features, K-Means/DBSCAN, LogReg/RF/XGBoost,
                                 alarm policy, Stage 2 root cause, charts, model freeze)
  score_new_well.py              scores an unseen well with the frozen models
  frequency_recommendation.py    drawdown regression Pi = a − k·Hz + closed-loop back-test
  frequency_optimisation.py      frequency sweep + break-even downtime
  build_report.py                PDF summary builder
  HOW_TO_ADD_A_WELL.md
3_results/             Official results; comparisons/ = earlier configurations (incl. before the
                       physics fix); latest_run/ = a local re-run
4_charts/              The 11 official charts + dashboard screenshot; latest_run/ = local re-run charts,
                       including new_well_8/9/10_timeline.png from score_new_well.py
5_dashboard/           build_dashboard.py, dashboard_template.html, ESP_Live_Dashboard.html
data/                  esp_scada_16wells_6months.csv (4.19M rows), wells_config.csv,
                       well_equipment_specs.csv, model_store/esp_models.joblib (trained models)
```

## Dataset

`data/esp_scada_16wells_6months.csv`: 16 wells × 182 days × 1 minute = 4,193,280 rows.
Columns: well_id, split, timestamp, state, frequency_hz, motor_current_a, voltage_v,
Ti_intake_temp_f, Tm_motor_temp_f, Pi_intake_psi, Pd_discharge_psi, vibration_g, shutdown_event,
failure_cause, onset_cause, safe_min_hz, safe_max_hz, production_bpd.

## Note on versions
The official results in `3_results/` and `4_charts/` come from the Colab run. A local re-run with
newer XGBoost (3.4) and scikit-learn (1.9) catches 29 of 69 trips (42%); the data, Stage 2 and every
frequency result reproduce exactly. The saved models in `data/model_store/` and the dashboard come
from that local run, so load them with the same library versions.
