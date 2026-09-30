# ESP Failure Prediction & Optimal Operation Frequency

Samsung Innovation Campus — AI Capstone · Group-3 "Data Drillers"

Predicts electrical submersible pump shutdowns from 1-minute SCADA data, identifies which
failure mode is developing, and selects a safe operating frequency for underload-prone wells.

---

## Headline result

Measured on wells 8, 9 and 10, which were excluded from training entirely.

| | |
|---|---|
| Real shutdowns caught before they happened | **33 of 69 (48%)** |
| Caught with at least 30 minutes of usable warning | 31 of 69 (45%) |
| Median warning time on a caught trip | 180 minutes |
| Nuisance alarms | 18.7 per well per month |
| Root-cause accuracy once a trip is developing | 68.6% |
| Downtime the alarms could have avoided | ~99 hours over 6 months, 3 wells |

The alarm settings behind these numbers were chosen on validation data and then left alone.
The test set was never used to make a single decision.

---

## Where to start

**If you just want to run it:** open `1_run_in_colab/ESP_Capstone_Colab.ipynb` in Google
Colab and run the cells in order. Nothing needs downloading — the notebook generates the
data from scratch. About 12 minutes on a free CPU runtime, plus 4 more for the frequency
work in Step 5.

**If you run it on your own computer:** open the same notebook from inside this folder
(VS Code or Jupyter) with the Python 3.12 kernel and run the cells in order. It works in
`data/`, which already holds the dataset and the trained models, so Step 3 skips the data
generation.

> **Cloned from GitHub?** The dataset (`data/multiwell_scada_10wells_6mo.csv`, 431 MB) is
> not in the repository because it exceeds GitHub's 100 MB file limit. Rebuild it once with
> `python 2_code/generate_multiwell_data.py` (about 5 minutes). The simulator is seeded, so
> the file is identical. The trained models and everything else are included.

**If you just want to see the live demo:** double-click
`6_dashboard/ESP_Live_Dashboard.html` and press **Guided demo**. No code, no internet.

**If you want to read what was done:** `5_documents/ESP_Capstone_Summary.pdf` is the full
16-page technical write-up, including the results, the charts, and what was tried and
rejected.

**If you want the raw numbers:** `3_results/results.txt`.

---

## What is in this archive

```
1_run_in_colab/     Self-contained notebook — the whole project in one file (Steps 0–8)
2_code/             The scripts, in the order they run
3_results/          Official results (the Colab run the report quotes), earlier configurations
                    in comparisons/, and latest_run/ (written by every local training run)
4_charts/           Official charts, the dashboard screenshot, and latest_run/ (charts from
                    every local training run)
5_documents/        Report, presentation (with the demo video), team guide, PDF summary, action plan
6_dashboard/        Live monitor: build_dashboard.py, its template, and ESP_Live_Dashboard.html
data/               Everything the code reads: the 4.19M-row dataset (431 MB, not on GitHub), well roster,
                    equipment specs, and the trained models (model_store/esp_models.joblib)
```

Every script and the notebook read the dataset and models from `data/` and write charts and
result files to `4_charts/latest_run/` and `3_results/latest_run/`, so a local run never
overwrites the official results. To point them elsewhere, set `ESP_DATA`, `ESP_CHARTS` or
`ESP_RESULTS`. On Colab everything stays in `/content/esp_multiwell`.

The official results in `3_results/` and `4_charts/` come from the Colab run the report and
presentation quote (33 of 69 trips, 48%). A local run on newer XGBoost and scikit-learn
releases catches 29 of 69 (42%) on the same data; Stage 2 and every frequency result are
identical. The saved models in `data/model_store/` and the live dashboard come from that
local run.

### 2_code — what each script does

| File | What it does |
|---|---|
| `wells_config.csv` | The well roster. **This is the only file you edit to add a well.** |
| `equipment_specs.py` | Assigns each well a motor, protector, gas separator and pump, and derives its safe Hz band as the intersection of their ratings |
| `generate_multiwell_data.py` | Simulates 16 wells × 6 months × 1 minute of SCADA data (4.19M rows) from a state machine with physics-inspired sensors |
| `run_full_analysis.py` | The main pipeline: clustering, Stage 1 detection, Stage 2 root cause, baselines, alarm policy, all charts; ends by saving the trained models to `data/model_store/` |
| `score_new_well.py` | Scores a well the models never saw with the saved models, without re-training (`python score_new_well.py 17`, or `--verify` to re-check wells 8–10) |
| `frequency_recommendation.py` | Fits each well's drawdown coefficient from its own history and back-tests the frequency recommendation closed-loop |
| `frequency_optimisation.py` | Frequency sweep, adaptive-vs-fixed comparison, and the break-even calculation |
| `build_report.py` | Builds the PDF summary |

Run order: `equipment_specs` → `generate_multiwell_data` → `run_full_analysis` →
`frequency_recommendation` → `frequency_optimisation`.

### 3_results/comparisons — why these are kept

These are earlier configurations, kept so the effect of each change is visible rather than
asserted:

- `results_6cause_baseline.txt` — before narrowing from six failure causes to four
- `results_4cause_4sensor.txt` — the experiment that removed the vibration and voltage
  sensor channels; gas-lock classification collapsed from 0.68 to 0.20, so the channels
  were kept
- `results_before_physics_fix.txt` — the 19% catch rate, before the two simulator physics
  errors were found

---

## Scope: four failure causes, not six

| Cause | Status | Why |
|---|---|---|
| GAS_LOCK | In scope | Gas accumulates at the intake over hours |
| UNDERLOAD | In scope | Inflow starvation develops as the fluid level falls — and is correctable by frequency |
| HIGH_TEMP | In scope | Motor temperature climbs measurably ahead of the thermal trip |
| HIGH_DISCHARGE | In scope | Scale or restriction builds discharge pressure gradually |
| LOW_VOLTAGE | **Removed** | A grid sag is an external supply event. Well data cannot anticipate it, and the drive's own undervoltage protection already handles it |
| VIBRATION | **Removed as a label** | A pump does not shut down *because* it vibrates — it is a condition indicator. Kept as a **sensor input**; removing the channel was tested and measurably hurt gas-lock classification |

Dropping a *label* while keeping the *signal* is the distinction that mattered here.

---

## The finding that moved the project furthest

Detection sat at 19% for a long stretch and no amount of feature work, model substitution
or threshold tuning moved it. What broke the deadlock was not a modelling idea: measuring,
for each cause, how large its precursor was compared with the sensor noise it had to be
seen through.

| Cause | Precursor vs. noise | Catch rate at the time |
|---|---|---|
| GAS_LOCK | 6 σ | 45% |
| HIGH_TEMP | 3.1 σ | 43% |
| HIGH_DISCHARGE | 0.8 σ | **10%** |
| UNDERLOAD | 0 σ (no precursor at all) | **6%** |

Catch rate tracked signal-to-noise almost exactly. The bottleneck was the data, not the
model — and two errors in the simulator's physics were the cause. UNDERLOAD was firing as
an instantaneous random event when inflow starvation in fact develops gradually as the
fluid level over the pump falls. HIGH_DISCHARGE carried a 12 psi precursor against 15 psi
of noise, which is mathematically undetectable.

Correcting both took UNDERLOAD from 6% to 59%, HIGH_DISCHARGE from 10% to 55%, and the
overall catch rate from 19% to 48%. Nothing was paid for it elsewhere — nuisance alarms
fell and median warning time lengthened at the same time. When every axis improves
together, the signal was genuinely there and had simply been buried.

**The transferable part is the diagnostic itself.** Before spending another week on model
architecture, measure whether the precursor you are asking the model to find is larger than
the noise it is hidden in. On real field data the same check answers a more important
question: whether the sensor set is adequate at all, or whether the fix is instrumentation
rather than software.

---

## On the frequency work

The control provably works — underload trips went 22 → 0 closed-loop, on the same wells
with the same random seeds. It also cost 18.9% of production to recover 0.8% of uptime, so
under these conditions it is the wrong thing to do.

The deliverable is therefore not a setpoint but a decision rule:

> Slow the well down if the expected downtime per trip exceeds the break-even hours
> (~33–36 h for these wells); otherwise accept the trips.

With automatic restart in a few hours, the arithmetic says accept them. With a remote well
waiting a day or two for a crew, it says slow it down. Same well, opposite answer, decided
entirely by restart time — which is a field fact, not a modelling choice. The break-even
figure is built only from directly measured quantities, so it survives the simulator's
invented constants.

**One result in there is deliberately not being acted on.** The frequency sweep claims
production keeps rising to 120% of BEP. That is an artifact — the production proxy assumes
flow rises with speed without limit, pump submergence is not modelled, and a trip carries
no run-life penalty. Recommendations are capped at BEP, as standard practice requires. It
is recorded as a limitation rather than tuned until it gave the wanted answer.

---

## Already tried and rejected

Don't redo these without a new angle. Each was implemented, measured and reverted:

- **Frequency-detrended residual features** — catch rate fell 49% → 28%; detrending removed
  real signal along with the trend
- **Relative-only features** — closed the validation-to-test gap from the wrong end; catch
  rate 14%
- **F1-maximised alarm thresholds** — degenerate at 1% positives, selects near-zero recall
- **Dropping vibration/voltage as sensor channels** — gas-lock classification collapsed
- **Counting false alarms per minute** — flattered results badly; a "5.4% false alarm rate"
  was really ~208 alarms per well per month

---

## Honest limitations

- The data is simulated. Every number here is a method demonstration, not a field result.
- 52% of trips are still missed. This supplements existing protection; it does not replace it.
- HIGH_TEMP collapses to zero recall in Stage 2, its cases absorbed into UNDERLOAD.
- A validation-to-test gap persists (ROC-AUC 0.951 vs 0.768). Well 10, which carries three
  interleaved failure modes, reaches only 21% and shows this most clearly.

Several figures in this project are *worse* than earlier reported ones — the 39.1%
root-cause accuracy, the 49% catch rate, the 5.4% false-alarm rate. In each case the earlier
number was measuring something other than what it claimed, and it was corrected in place
rather than quietly kept.

---

## Next steps, in priority order

1. **Cause-specific features for HIGH_TEMP** — the one class failing outright
2. **Point the pipeline at real historical wells** — the only thing that can settle whether
   the optimum sits below BEP at all, and whether the precursors survive real
   instrumentation. Run the signal-to-noise diagnostic on them first.
3. **Per-cause alarm thresholds** — GAS_LOCK at 29% vs UNDERLOAD at 59% under one shared policy
4. **Submergence and pump-curve constraints in the simulator** — without them the frequency
   sweep produces an answer nobody should act on

---

## Adding a well

Edit `2_code/wells_config.csv`, add a row, re-run the pipeline:

```
well_id,causes,split
17,GAS_LOCK:0.6;HIGH_TEMP:0.4,test
```

Only the new well is simulated; the existing 16 are not re-generated, so the comparison
stays clean. See `2_code/HOW_TO_ADD_A_WELL.md`.
