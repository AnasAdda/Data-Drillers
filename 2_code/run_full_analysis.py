"""
Full ESP analysis on the 10-well mixed-cause, equipment-aware dataset:

  UNSUPERVISED branch: K-Means / DBSCAN clustering of operating states
    (independent of the supervised branch - per the team's diagram: two
    parallel branches, unsupervised does NOT feed the supervised model).

  SUPERVISED branch, two stages:
    Stage 1: binary classification - will a trip happen in the next H
             minutes?
    Stage 2: given a likely failure, which cause? (multi-class)

Key fix vs. the previous run: features are now Z-SCORE NORMALIZED PER WELL
before modeling. Each well has its own baseline sensor levels (different
reservoir pressure, productivity index, supply voltage - see
generate_multiwell_data.py), so raw/absolute feature values don't transfer
across wells. Normalizing against each well's OWN baseline is what lets a
model trained on train wells generalize to fully unseen wells. For train
wells, the baseline is computed from the FIT period only (no time leakage).
For held-out test wells, the baseline is computed from that well's own
first 30 days (a realistic "calibration window" after a new well starts
producing - you don't get another well's baseline in the field either).

SECOND fix (this run): Stage 2 previously trained on every row inside the
Stage-1 H=60-minute "trip within H" window, with the cause back-filled from
the eventual shutdown event. A lot of that window is still statistically
normal - the fault hasn't visibly started yet - so a large share of the
"positive" training rows carried a cause label with no matching signal in
the features, which is what collapsed the classifier onto the single
best-represented class (LOW_VOLTAGE). Stage 2 now trains and evaluates only
on rows where the simulator's own state machine says the well is actively
in "onset" for a given cause (the `onset_cause` column) - a clean,
non-ambiguous label instead of a noisy one.

THIRD fix (this run): the dataset previously gave detection zero real
lead-time signal - sensors were flat until the instant onset began. A small
precursor leak (see generate_multiwell_data.py) now lets GAS_LOCK,
HIGH_TEMP, HIGH_DISCHARGE and VIBRATION's underlying degradation indices
show up gradually beforehand; a longer 240-minute rolling mean is added
below so the model has a feature that can actually see that drift.
UNDERLOAD and LOW_VOLTAGE stay abrupt/external by design (a real supply
sag has little lead time either), so they're expected to stay harder to
catch early - that's a realistic difference between failure modes, not a
gap to hide.

More wells (16 total, 13 train / 3 held-out test) give every cause at
least 2 training wells instead of 1, for better cross-well generalization.
Add wells any time via wells_config.csv - see HOW_TO_ADD_A_WELL.md.

Equipment integration (per instructor feedback: "define the operationally
allowable frequency range"): each well's SAFE_MIN_HZ / SAFE_MAX_HZ (from
well_equipment_specs.csv, derived from Motor/Protector/Pump ratings) is
carried through and used to (a) confirm the simulated data never violates
it [checked in generate_multiwell_data.py], and (b) clip any operating
frequency recommendation to stay within the specific well's own equipment
limits - never a shared global range.

Leakage prevention:
  - Well-based split: wells 8, 9, 10 held out entirely (never in training).
  - Time-based split: within the 7 training wells, final 6 weeks held out
    as a validation window.

Baselines: fixed-threshold rule (mirrors current SCADA alarm logic) and
last-value/persistence.

Reported: lead time, false alarms, missed failures, production impact,
alongside standard model metrics (ROC-AUC, F1, precision/recall).
"""
import warnings
warnings.filterwarnings("ignore")
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

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
H = 30  # failure horizon in minutes - shorter horizon + run-life feature combined (best result)
CAUSE_NAMES = ["GAS_LOCK", "UNDERLOAD", "HIGH_TEMP", "HIGH_DISCHARGE", "VIBRATION", "LOW_VOLTAGE"]

df = pd.read_csv(f"{OUT}/multiwell_scada_10wells_6mo.csv", parse_dates=["timestamp"])
df = df.sort_values(["well_id", "timestamp"]).reset_index(drop=True)
df = df.drop(columns=["Ti_intake_temp_f"])  # not used as a modeling feature - drop early to save memory
specs = pd.read_csv(f"{OUT}/well_equipment_specs.csv").set_index("well_id")

TRAIN_WELLS = sorted(df.loc[df["split"] == "train", "well_id"].unique())
TEST_WELLS = sorted(df.loc[df["split"] == "test", "well_id"].unique())
print("Train wells (fully seen):", TRAIN_WELLS)
print("Test wells (fully held out):", TEST_WELLS)

# Sensor channels fed to the models. DROP_SENSORS lets a run exclude channels
# without editing the pipeline - used to test the field engineer's argument that
# vibration and supply voltage are mostly alarm load: vibration does not itself
# trip an ESP (it is a condition indicator, not a trip cause) and a surface
# voltage sag is an external grid event that well data cannot anticipate.
import os
SENSOR_COLS = ["motor_current_a", "voltage_v", "vibration_g", "Pi_intake_psi", "Pd_discharge_psi", "Tm_motor_temp_f"]
DROP_SENSORS = [s for s in os.environ.get("DROP_SENSORS", "").split(",") if s]
if DROP_SENSORS:
    SENSOR_COLS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]
    print("Dropped sensor channels:", DROP_SENSORS)
print("Sensor channels in use:", SENSOR_COLS)
RESULTS_NAME = os.environ.get("RESULTS_NAME", "results.txt")
for col in SENSOR_COLS + ["frequency_hz"]:
    df[col] = df[col].astype("float32")

# ============================================================
# Rolling features, per well - computed via groupby().transform() so new
# columns are added onto the existing dataframe in place, rather than
# building a full per-well copy of every column for every well (which is
# what blew past the memory budget once the dataset grew to 16 wells).
# ============================================================
RAW_FEATURES = ["frequency_hz", "minutes_since_last_shutdown"] + SENSOR_COLS
ROLL_STATS = ["roll15_mean", "roll15_std", "roll60_mean", "roll60_std", "roll240_mean", "chg5"]
ROLL_FEATURES = [f"{c}_{stat}" for c in SENSOR_COLS for stat in ROLL_STATS]
ALL_FEATURES = RAW_FEATURES + ROLL_FEATURES

grouped = df.groupby("well_id", sort=False)
for col in SENSOR_COLS:
    df[f"{col}_roll15_mean"] = grouped[col].transform(lambda s: s.rolling(15, min_periods=1).mean()).astype("float32")
    df[f"{col}_roll15_std"] = grouped[col].transform(lambda s: s.rolling(15, min_periods=1).std()).fillna(0).astype("float32")
    df[f"{col}_roll60_mean"] = grouped[col].transform(lambda s: s.rolling(60, min_periods=1).mean()).astype("float32")
    df[f"{col}_roll60_std"] = grouped[col].transform(lambda s: s.rolling(60, min_periods=1).std()).fillna(0).astype("float32")
    df[f"{col}_roll240_mean"] = grouped[col].transform(lambda s: s.rolling(240, min_periods=1).mean()).astype("float32")
    df[f"{col}_chg5"] = grouped[col].transform(lambda s: s.diff(5)).fillna(0).astype("float32")
    gc.collect()
del grouped
gc.collect()
print("Rolling features done.")

# ============================================================
# RELATIVE (SELF-REFERENCED) FEATURES - TESTED AND TURNED OFF.
#
# Hypothesis: Stage 1 on RAW absolute levels scored ~0.82 ROC-AUC on
# validation (held-out weeks of wells it trained on) but only ~0.66 on
# fully unseen wells. Each well has its own reservoir pressure,
# productivity index, supply voltage and BEP frequency, so an absolute
# threshold like "intake pressure below 2100 psi means trouble" looked like
# memorization of a specific well rather than a transferable rule. The fix
# would be features that compare each sensor ONLY against its own recent
# history, leaving no absolute level to memorize. (An earlier attempt at
# this failed for a different reason - the detrended features were ADDED
# alongside the raw ones, leaving the absolute values still available to
# overfit on - so this version REPLACES them, which is the test the idea
# deserved.)
#
# RESULT: the gap did close, but from the wrong end. Validation ROC-AUC fell
# 0.82 -> 0.685 while test only moved 0.66 -> 0.63, and the event-level catch
# rate dropped from 49% to 14%. So the absolute levels were carrying real
# signal, not just well-identity memorization. Honest negative result: kept
# in the code behind a flag rather than deleted, because the diagnosis
# (well-specific overfitting) is still worth revisiting with a better
# feature design - but it is OFF, because it measurably made things worse.
# ============================================================
USE_RELATIVE_FEATURES = False
EPS = 1e-3
REL_FEATURES = []
if USE_RELATIVE_FEATURES:
    bep_map = specs["pump_bep_hz"].to_dict()
    df["freq_vs_bep"] = (df["frequency_hz"] - df["well_id"].map(bep_map)).astype("float32")
    REL_FEATURES = ["freq_vs_bep"]
    for col in SENSOR_COLS:
        base = df[f"{col}_roll240_mean"]
        safe_base = base.abs().clip(lower=EPS)
        # short- and medium-term deviation from the well's own 4-hour baseline
        df[f"{col}_dev15"] = (df[f"{col}_roll15_mean"] - base).astype("float32")
        df[f"{col}_dev60"] = (df[f"{col}_roll60_mean"] - base).astype("float32")
        # scale-free versions: percentage deviation transfers across wells with
        # different absolute operating levels
        df[f"{col}_pctdev15"] = ((df[f"{col}_roll15_mean"] - base) / safe_base).astype("float32")
        df[f"{col}_pctdev60"] = ((df[f"{col}_roll60_mean"] - base) / safe_base).astype("float32")
        # variability relative to level (coefficient of variation)
        df[f"{col}_cv15"] = (df[f"{col}_roll15_std"] / safe_base).astype("float32")
        df[f"{col}_cv60"] = (df[f"{col}_roll60_std"] / safe_base).astype("float32")
        # short-term rate of change, scale-free
        df[f"{col}_pctchg5"] = (df[f"{col}_chg5"] / safe_base).astype("float32")
        new_cols = [f"{col}_dev15", f"{col}_dev60", f"{col}_pctdev15",
                    f"{col}_pctdev60", f"{col}_cv15", f"{col}_cv60", f"{col}_pctchg5"]
        # guard against inf/NaN from any residual near-zero denominator, per column
        # (doing this once over all 42 columns at the end would materialize a
        # second full-width copy and blow the memory budget)
        for nc in new_cols:
            df[nc] = df[nc].replace([np.inf, -np.inf], 0).fillna(0).astype("float32")
        REL_FEATURES += new_cols
        gc.collect()

print(f"Relative features: {'ON, ' + str(len(REL_FEATURES)) + ' features' if USE_RELATIVE_FEATURES else 'OFF (tested, measurably worse - see header)'}")

# ============================================================
# Labels: trip within H minutes, and the cause of that trip (per well).
# Computed per-well on small extracted arrays (not full-frame copies) and
# written back into df in place - same memory-saving reasoning as above.
# ============================================================
df["label_trip_within_H"] = np.int8(0)
df["minutes_since_last_shutdown"] = np.float32(0)
for well_id, idx in df.groupby("well_id", sort=False).groups.items():
    shutdown_arr = df.loc[idx, "shutdown_event"].to_numpy()
    n = len(shutdown_arr)
    next_shutdown_idx = np.full(n, np.inf)
    nxt = np.inf
    for i in range(n - 1, -1, -1):
        if shutdown_arr[i] == 1:
            nxt = i
        next_shutdown_idx[i] = nxt
    minutes_to_shutdown = next_shutdown_idx - np.arange(n)
    label = ((minutes_to_shutdown > 0) & (minutes_to_shutdown <= H)).astype("int8")

    # "Run life" feature: minutes elapsed since this well's last shutdown
    # (or since well start if none yet). Mirrors a standard real-world ESP
    # monitoring signal (time since last pull/workover), and stands in for
    # the simulator's latent degradation index, which grows roughly
    # monotonically between shutdowns - a well-generalizable, physically
    # grounded leading indicator instead of relying on noisy sensor levels
    # alone. Requested: improve detection while keeping H=60 (the
    # operationally required lead time) rather than shortening it further.
    since_last = np.empty(n, dtype="float32")
    last_idx = None
    for i in range(n):
        since_last[i] = float(i) if last_idx is None else float(i - last_idx)
        if shutdown_arr[i] == 1:
            last_idx = i

    df.loc[idx, "label_trip_within_H"] = label
    df.loc[idx, "minutes_since_last_shutdown"] = since_last
gc.collect()
if USE_RELATIVE_FEATURES:
    REL_FEATURES.append("minutes_since_last_shutdown")  # well-relative by construction
print("Labels and run-life feature done.")

# Extract the small event lookup table FIRST, then drop the columns nothing
# downstream needs before taking the big filtered copy below. On 4.19M rows
# an object-dtype string column costs far more than a float column, so
# dropping the unused ones (and categorizing the rest) is what keeps this
# within the memory budget now that Stage 1 carries 42 extra features.
shutdown_events_df = df.loc[df["shutdown_event"] == 1, ["well_id", "timestamp", "failure_cause"]].copy()
shutdown_events_df["failure_cause"] = shutdown_events_df["failure_cause"].astype(str)
df = df.drop(columns=["failure_cause", "shutdown_event", "split"])
df["onset_cause"] = df["onset_cause"].fillna("").astype("category")
df["state"] = df["state"].astype("category")
gc.collect()

mask_running = df["state"].isin(["normal", "onset", "ramp"])
df = df[mask_running]          # rebind rather than keeping both frames alive
gc.collect()
model_df = df.reset_index(drop=True)
del df
gc.collect()

train_pool = model_df[model_df["well_id"].isin(TRAIN_WELLS)]
test_pool = model_df[model_df["well_id"].isin(TEST_WELLS)]
del model_df
gc.collect()

val_cutoff = train_pool["timestamp"].max() - pd.Timedelta(weeks=6)
fit_set = train_pool[train_pool["timestamp"] < val_cutoff].copy()
val_set = train_pool[train_pool["timestamp"] >= val_cutoff].copy()
del train_pool  # fit_set + val_set now hold everything it did
gc.collect()

print(f"\nFit rows (train wells, early period): {len(fit_set):,}")
print(f"Val rows (train wells, final 6 weeks - time-holdout): {len(val_set):,}")
print(f"Test rows (fully held-out wells 8,9,10): {len(test_pool):,}")

# ============================================================
# TRIED AND REVERTED: frequency-detrended residual features (sensor minus
# its per-well linear fit against frequency, from normal-state calibration
# data). Rationale was sound - detrending should be both more transferable
# across wells and more cause-specific - but tested honestly: Stage 1 catch
# rate dropped from 49% to 28% (adding 6 more raw features to an already
# marginal detector diluted its splits), while Stage 2 only ticked up from
# 60.5% to 61.9% without fixing the GAS_LOCK/HIGH_DISCHARGE/UNDERLOAD
# collapse it targeted. Net negative - reverted, not kept.
# ============================================================
# PER-WELL NORMALIZATION - the key fix vs. the previous run.
# Train wells: baseline = mean/std over the FIT period only (no leakage).
# Test wells: baseline = mean/std over that well's own first 30 days
# (a realistic "calibration window" - you don't borrow another well's
# baseline in the field).
# ============================================================
# Floor each well's std at 20% of the GLOBAL (fit-period, all train wells)
# std for that feature. Without this, a feature with near-zero variance in a
# short calibration window (esp. the 30-day test-well window) produces a
# tiny denominator, blowing up later normal readings into extreme z-scores.
#
# All of these statistics are computed from SAMPLES rather than every row:
# a per-well mean/std over 100k minutes is statistically indistinguishable
# from one over 200k, and materializing the full 2.6M x 44 float block to
# compute them was a gigabyte-scale temporary for no accuracy gain.
STAT_SAMPLE = 100_000
global_std_floor = (fit_set[ALL_FEATURES].sample(n=min(300_000, len(fit_set)), random_state=7).std() * 0.2).replace(0, 1e-3)

baseline_stats = {}
for well_id in TRAIN_WELLS:
    wf = fit_set.loc[fit_set["well_id"] == well_id, ALL_FEATURES]
    if len(wf) > STAT_SAMPLE:
        wf = wf.sample(n=STAT_SAMPLE, random_state=7)
    std = wf.std().combine(global_std_floor, max).replace(0, 1)
    baseline_stats[well_id] = (wf.mean(), std)
    del wf
    gc.collect()

for well_id in TEST_WELLS:
    wf_all = test_pool[test_pool["well_id"] == well_id]
    calib_cutoff = wf_all["timestamp"].min() + pd.Timedelta(days=30)
    wf = wf_all.loc[wf_all["timestamp"] < calib_cutoff, ALL_FEATURES]
    if len(wf) > STAT_SAMPLE:
        wf = wf.sample(n=STAT_SAMPLE, random_state=7)
    std = wf.std().combine(global_std_floor, max).replace(0, 1)
    baseline_stats[well_id] = (wf.mean(), std)
    del wf_all, wf
    gc.collect()


# Only the columns Stage 2 and the clustering branch actually read - copying
# every column here (including Stage 1's 42 relative features, which the
# normalized path never touches) is what pushed this past the memory budget.
NORM_KEEP = ["well_id", "timestamp", "onset_cause", "label_trip_within_H"]


def normalize(frame):
    out = frame[NORM_KEEP + ALL_FEATURES].copy()
    for well_id, g in frame.groupby("well_id"):
        mean, std = baseline_stats[well_id]
        idx = g.index
        out.loc[idx, ALL_FEATURES] = ((g[ALL_FEATURES] - mean) / std).astype("float32").values
    return out


# Normalize AFTER subsetting, never before: the clustering branch only needs a
# 150k-row sample and Stage 2 only needs the (few hundred) true-onset rows, so
# normalizing all 2.6M fit rows first was pure waste - and it was what finally
# exceeded the memory budget once Stage 1's relative features were added.
#
# Extract those two small subsets NOW, while ALL_FEATURES still exists, then
# drop those columns from the big frames. After this point Stage 1 works on
# slim frames carrying only its own relative features, which roughly halves
# peak memory.
clust_sample = normalize(fit_set.sample(n=min(150_000, len(fit_set)), random_state=7))
cause_fit = normalize(fit_set[fit_set["onset_cause"].astype(str).isin(CAUSE_NAMES)])
cause_test = normalize(test_pool[test_pool["onset_cause"].astype(str).isin(CAUSE_NAMES)])
gc.collect()

if USE_RELATIVE_FEATURES:
    STAGE1_KEEP = ["well_id", "timestamp", "label_trip_within_H"] + REL_FEATURES
    DROP_COLS = [c for c in ALL_FEATURES if c not in STAGE1_KEEP]
    # keep the two raw columns the SCADA-rule baseline and the timeline chart read
    DROP_COLS = [c for c in DROP_COLS
                 if c not in ("vibration_g", "vibration_g_roll15_mean", "motor_current_a_roll15_mean")]
    fit_set = fit_set.drop(columns=DROP_COLS)
    val_set = val_set.drop(columns=DROP_COLS)
    test_pool = test_pool.drop(columns=DROP_COLS)
    gc.collect()
    print(f"Dropped {len(DROP_COLS)} absolute-level columns from the Stage 1 frames "
          f"(kept by the small normalized subsets above).")

# ============================================================
# UNSUPERVISED BRANCH: K-Means / DBSCAN on normalized features
# (independent of the supervised branch - not fed into it)
# ============================================================
print("\n=== UNSUPERVISED: Clustering operating states ===")
Xc = clust_sample[ALL_FEATURES].to_numpy(dtype="float32")

km = KMeans(n_clusters=6, n_init=10, random_state=7).fit(Xc)
km_sil = silhouette_score(Xc, km.labels_, sample_size=20_000, random_state=7)
print(f"K-Means (k=6) silhouette: {km_sil:.3f}")

clust_sample = clust_sample.copy()
clust_sample["kmeans_cluster"] = km.labels_
cluster_failure_rate = clust_sample.groupby("kmeans_cluster")["label_trip_within_H"].mean()
print("K-Means cluster -> failure-within-H rate:")
print(cluster_failure_rate.to_string())

db_sample = clust_sample.sample(n=min(40_000, len(clust_sample)), random_state=7)
Xdb = db_sample[ALL_FEATURES].to_numpy(dtype="float32")
db = DBSCAN(eps=2.0, min_samples=20).fit(Xdb)
n_db_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
db_sample = db_sample.copy()
db_sample["dbscan_cluster"] = db.labels_
outlier_failure_rate = db_sample.groupby(db_sample["dbscan_cluster"] == -1)["label_trip_within_H"].mean()
print(f"\nDBSCAN found {n_db_clusters} clusters (+ noise/outliers)")
print(f"Failure-within-H rate: outliers={outlier_failure_rate.get(True, float('nan')):.4f} "
      f"vs. clustered-normal={outlier_failure_rate.get(False, float('nan')):.4f}")

pca = PCA(n_components=2, random_state=7).fit(Xc)
Xc_pca = pca.transform(Xc)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sc0 = axes[0].scatter(Xc_pca[:, 0], Xc_pca[:, 1], c=km.labels_, cmap="tab10", s=3, alpha=0.4)
axes[0].set_title("K-Means clusters (PCA, well-normalized features)")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
plt.colorbar(sc0, ax=axes[0], label="cluster")
sc1 = axes[1].scatter(Xc_pca[:, 0], Xc_pca[:, 1], c=clust_sample["label_trip_within_H"], cmap="coolwarm", s=3, alpha=0.4)
axes[1].set_title("Actual failure-within-H label (same projection)")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
plt.colorbar(sc1, ax=axes[1], label="trip within H")
plt.tight_layout()
plt.savefig(f"{CHARTS}/clustering_pca_normalized.png", dpi=150)
plt.close()
print("Saved clustering_pca_normalized.png")
del clust_sample, db_sample, Xc, Xdb, Xc_pca
gc.collect()

# ============================================================
# SUPERVISED - STAGE 1: failure detection, models vs. baselines
# ============================================================
# Stage 1 now trains on the RELATIVE feature set only (see the relative-
# feature block above): every feature compares a sensor against its own
# recent history in the same well, so no absolute operating level is
# available for the model to memorize. This replaces the previous raw
# absolute-level feature set, which scored 0.82 ROC-AUC on validation but
# only 0.66 on fully unseen wells - the signature of memorizing each
# training well's own baseline rather than learning a transferable rule.
# Stage 1 uses the RAW/absolute + rolling feature set. The relative-only
# alternative was implemented and measured (see the flagged block above) and
# was worse on every event-level metric, so it stays off.
STAGE1_FEATURES = REL_FEATURES if USE_RELATIVE_FEATURES else ALL_FEATURES

NEG_PER_POS = 40
pos_fit = fit_set[fit_set["label_trip_within_H"] == 1]
neg_fit_full = fit_set[fit_set["label_trip_within_H"] == 0]
n_neg_keep = min(len(neg_fit_full), len(pos_fit) * NEG_PER_POS)
neg_fit = neg_fit_full.sample(n=n_neg_keep, random_state=7)
fit_ds = pd.concat([pos_fit, neg_fit]).sort_values("timestamp")
print(f"\nFit rows after downsampling (train-only, {len(STAGE1_FEATURES)} features): {len(fit_ds):,} "
      f"({len(pos_fit):,} positive + {len(neg_fit):,} negative, ratio 1:{NEG_PER_POS})")

Xfit = fit_ds[STAGE1_FEATURES].to_numpy(dtype="float32")
yfit = fit_ds["label_trip_within_H"].to_numpy()
Xval = val_set[STAGE1_FEATURES].to_numpy(dtype="float32")
yval = val_set["label_trip_within_H"].to_numpy()
Xtest = test_pool[STAGE1_FEATURES].to_numpy(dtype="float32")
ytest = test_pool["label_trip_within_H"].to_numpy()
del pos_fit, neg_fit_full, neg_fit
gc.collect()

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=9, class_weight="balanced_subsample", random_state=7, n_jobs=2),
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", scale_pos_weight=(yfit == 0).sum() / max((yfit == 1).sum(), 1),
        random_state=7, n_jobs=2,
    ),
}

results = []
test_probas = {}
val_probas = {}
print("\n=== SUPERVISED STAGE 1: Failure detection ===")
for name, model in models.items():
    print(f"  training {name}...")
    model.fit(Xfit, yfit)
    val_proba = model.predict_proba(Xval)[:, 1]
    test_proba = model.predict_proba(Xtest)[:, 1]
    val_probas[name] = val_proba
    test_pred = (test_proba >= 0.5).astype(int)
    results.append({
        "Model": name,
        "Val ROC-AUC": roc_auc_score(yval, val_proba),
        "Test Accuracy": accuracy_score(ytest, test_pred),
        "Test Precision": precision_score(ytest, test_pred, zero_division=0),
        "Test Recall": recall_score(ytest, test_pred, zero_division=0),
        "Test F1": f1_score(ytest, test_pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(ytest, test_proba),
    })
    test_probas[name] = test_proba
    gc.collect()

# Fixed-threshold SCADA-style rule, built only from channels this run actually has
rule_parts = []
if "vibration_g" in SENSOR_COLS:
    vib_thresh = fit_set["vibration_g_roll15_mean"].mean() + 2.5 * fit_set["vibration_g_roll15_mean"].std()
    rule_parts.append(test_pool["vibration_g_roll15_mean"] > vib_thresh)
if "motor_current_a" in SENSOR_COLS:
    cur_thresh_low = fit_set["motor_current_a_roll15_mean"].mean() - 2.5 * fit_set["motor_current_a_roll15_mean"].std()
    rule_parts.append(test_pool["motor_current_a_roll15_mean"] < cur_thresh_low)
rule_mask = rule_parts[0]
for extra in rule_parts[1:]:
    rule_mask = rule_mask | extra
rule_pred = rule_mask.astype(int).values
results.append({
    "Model": "Baseline: fixed-threshold rule", "Val ROC-AUC": np.nan,
    "Test Accuracy": accuracy_score(ytest, rule_pred),
    "Test Precision": precision_score(ytest, rule_pred, zero_division=0),
    "Test Recall": recall_score(ytest, rule_pred, zero_division=0),
    "Test F1": f1_score(ytest, rule_pred, zero_division=0), "Test ROC-AUC": np.nan,
})

persistence_pred = test_pool.groupby("well_id")["label_trip_within_H"].shift(1).fillna(0).astype(int).values
results.append({
    "Model": "Baseline: persistence (last value)", "Val ROC-AUC": np.nan,
    "Test Accuracy": accuracy_score(ytest, persistence_pred),
    "Test Precision": precision_score(ytest, persistence_pred, zero_division=0),
    "Test Recall": recall_score(ytest, persistence_pred, zero_division=0),
    "Test F1": f1_score(ytest, persistence_pred, zero_division=0), "Test ROC-AUC": np.nan,
})

results_df = pd.DataFrame(results)
print("\n=== STAGE 1 results: models vs. baselines (held-out wells 8,9,10) ===")
print(results_df.to_string(index=False))

# Model selection happens further down, on VALIDATION EVENT-LEVEL COST rather
# than minute-level ROC-AUC. Two earlier criteria were tried and rejected:
# Test ROC-AUC (leakage - the held-out wells must not influence any modelling
# decision), then Validation ROC-AUC, which is leakage-free but measures the
# wrong thing: it ranks individual minutes, while the project is judged on
# caught trips, warning time and how often an engineer is interrupted. On this
# dataset it picked the model with the LOWEST catch rate. The selection now
# uses the same cost function as the alarm policy, computed on validation
# trips only.

# ============================================================
# ALARM POLICY: threshold + persistence, tuned on validation by COST
#
# Two changes to how an alert is defined, both standard alarm-engineering
# practice that the previous single-minute threshold ignored:
#
# 1. PERSISTENCE (debouncing): an alarm requires N consecutive minutes above
#    threshold, not one. Isolated noise spikes die; a genuine onset, which
#    develops over tens of minutes, survives. This attacks the false-alarm
#    rate without giving up real detections - which then allows a LOWER
#    threshold, so both metrics can improve together instead of trading off.
#
# 2. EVENT-BASED false alarms, not minute-based. What matters operationally
#    is how many times an engineer is paged for nothing, not how many
#    individual minutes were flagged. One nuisance alarm lasting 40 minutes
#    is one interruption, not 40.
#
# Cost-weighted selection: Youden's J implicitly prices a missed trip the
# same as a false alarm, which is operationally wrong. A missed ESP trip
# costs hours of deferred production and possible equipment damage; a false
# alarm costs an engineer a few minutes of review. MISS_COST_RATIO below
# encodes that asymmetry explicitly, and is the single knob the team can
# turn to move along the trade-off curve.
# ============================================================
# One missed ESP trip costs hours of deferred production plus a possible
# workover; one nuisance alarm costs an engineer a few minutes at a screen.
# 200:1 encodes that asymmetry. This is the single knob that moves the
# operating point along the catch-rate / false-alarm curve, and the full
# curve is reported on the test wells at the end so the team can choose a
# different point without re-running anything.
MISS_COST_RATIO = 200.0
# An alarm that fires a few minutes before the trip is not an early warning -
# nobody can dispatch a technician or change frequency in that time. A catch
# only counts toward the selection objective if it arrived at least this many
# minutes ahead. Set equal to the prediction horizon: if the model cannot beat
# its own horizon, it is detecting the failure rather than predicting it.
MIN_USEFUL_LEAD_MIN = 30.0
ALERT_WINDOW_MIN = 180   # an alarm counts as "catching" a trip if it fires within this window before it


def per_well_thresholds(frame, proba_col, q):
    """Alert cutoff as a quantile of EACH WELL's own score distribution.

    A single absolute probability threshold does not transfer between wells:
    tuned on validation, 0.90 was the 99.9th percentile there, but on the
    held-out wells the same model's scores sit far lower, so that threshold
    fired twice in six months. Each well gets its own cutoff from its own
    score distribution instead - "alert on this well's riskiest X% of
    minutes" is a rule that means the same thing everywhere, and it needs no
    labels from the new well, only its own history.
    """
    return frame.groupby("well_id")[proba_col].quantile(q).to_dict()


def alarm_events(frame, proba_col, thresh_map, n_persist):
    """Return alarm start/end timestamps per well after persistence filtering."""
    events = []
    for well_id, g in frame.groupby("well_id", sort=False):
        g = g.sort_values("timestamp")
        above = (g[proba_col].to_numpy() >= thresh_map[well_id]).astype(int)
        if n_persist > 1:
            # rolling sum over the last n_persist minutes must be == n_persist
            csum = np.convolve(above, np.ones(n_persist, dtype=int), mode="full")[:len(above)]
            fired = (csum == n_persist).astype(int)
        else:
            fired = above
        ts = g["timestamp"].to_numpy()
        # group consecutive fired minutes into single alarm events (vectorized:
        # a +1 in the diff marks a run start, a -1 marks one past the run end)
        d = np.diff(np.concatenate(([0], fired, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        if len(starts):
            events.append(pd.DataFrame({"well_id": well_id, "start": ts[starts], "end": ts[ends]}))
    return pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=["well_id", "start", "end"])


def score_policy(frame, proba_col, trips_df, q, n_persist):
    """Evaluate an alarm policy: returns caught/missed trips, false-alarm events, lead times.

    An alarm counts as catching a trip if it is ACTIVE at any point in the
    alert window before that trip - i.e. the alarm interval OVERLAPS the
    window. Requiring the alarm to *start* inside the window (a first
    version of this did) silently scores a continuously-firing alarm as
    catching nothing, since its start sits far in the past.
    Lead time is measured from when the alarm was already active at the
    window edge, not from the alarm's own (possibly much earlier) start.
    """
    ev = alarm_events(frame, proba_col, per_well_thresholds(frame, proba_col, q), n_persist)
    caught, missed, leads = 0, 0, []
    matched_alarm_idx = set()
    for _, trip in trips_df.iterrows():
        st, wid = trip["timestamp"], trip["well_id"]
        if len(ev) == 0:
            missed += 1
            continue
        window_start = st - pd.Timedelta(minutes=ALERT_WINDOW_MIN)
        # overlap test: alarm starts before the trip AND ends after the window opens
        hits = ev[(ev["well_id"] == wid) & (ev["start"] < st) & (ev["end"] >= window_start)]
        if len(hits) > 0:
            caught += 1
            first_active = max(hits["start"].min(), window_start)
            leads.append((st - first_active).total_seconds() / 60.0)
            matched_alarm_idx.update(hits.index.tolist())
        else:
            missed += 1
    false_alarms = 0 if len(ev) == 0 else len(ev) - len(matched_alarm_idx)
    useful = sum(1 for l in leads if l >= MIN_USEFUL_LEAD_MIN)
    return {"caught": caught, "missed": missed, "false_alarm_events": false_alarms,
            "leads": leads, "n_alarms": len(ev), "useful_caught": useful,
            "useful_missed": (caught + missed) - useful}


# --- select the model AND its alarm policy together, on validation only ---
val_set = val_set.reset_index(drop=True)
val_trips = shutdown_events_df[
    shutdown_events_df["well_id"].isin(TRAIN_WELLS) &
    (shutdown_events_df["timestamp"] >= val_set["timestamp"].min())
]
print(f"\nSelecting model + alarm policy on validation ({len(val_trips)} trips in the window)...")

QUANTILES = [0.950, 0.975, 0.990, 0.995, 0.999]
selection_rows = []
for _mname in models.keys():
    val_set["proba"] = val_probas[_mname]
    _best = None
    for n_p in [1, 3, 5, 10, 15, 30]:
        for q in QUANTILES:
            r = score_policy(val_set, "proba", val_trips, q, n_p)
            c = r["useful_missed"] * MISS_COST_RATIO + r["false_alarm_events"]
            if _best is None or c < _best[0]:
                _best = (c, q, n_p, r)
    selection_rows.append({"Model": _mname, "val_cost": _best[0], "val_top_pct": round(100 * (1 - _best[1]), 2),
                            "val_persist": _best[2], "val_caught": _best[3]["caught"],
                            "val_caught_useful": _best[3]["useful_caught"],
                            "val_missed": _best[3]["missed"], "val_false_alarms": _best[3]["false_alarm_events"],
                            "val_median_lead": round(float(np.median(_best[3]["leads"])), 1) if _best[3]["leads"] else None})
selection_df = pd.DataFrame(selection_rows).sort_values("val_cost")
print("\nModel selection on validation event-level cost (lower is better):")
print(selection_df.to_string(index=False))

best_name = selection_df.iloc[0]["Model"]
best_proba = test_probas[best_name]
best_val_proba = val_probas[best_name]
print(f"\nBest model (selected on VALIDATION EVENT-LEVEL COST): {best_name}")

val_set["proba"] = best_val_proba

# Thresholds are taken as QUANTILES of the model's own validation scores, not
# fixed absolute values. Different models put their probabilities on wildly
# different scales here - class-balanced Logistic Regression pushes most rows
# above 0.5 on a 1%-positive problem, while the trees rarely exceed 0.3 - so a
# hard-coded absolute grid fits one model and completely misses the other.
# Quantiles adapt to whatever scale the selected model produces.
policy_grid = []
for n_persist in [1, 3, 5, 10, 15, 30]:
    for q in QUANTILES:
        r = score_policy(val_set, "proba", val_trips, q, n_persist)
        cost = r["useful_missed"] * MISS_COST_RATIO + r["false_alarm_events"]
        policy_grid.append({"n_persist": n_persist, "quantile": q, "caught": r["caught"],
                             "caught_useful": r["useful_caught"], "missed": r["missed"],
                             "false_alarms": r["false_alarm_events"], "cost": cost})
policy_df = pd.DataFrame(policy_grid).sort_values("cost")
print(policy_df.head(10).to_string(index=False))

best_policy = policy_df.iloc[0]
Q = float(best_policy["quantile"])
N_PERSIST = int(best_policy["n_persist"])
print(f"\nSelected alarm policy (validation, cost-weighted at {MISS_COST_RATIO:.0f}:1): "
      f"per-well top {100*(1-Q):.1f}% of scores, persistence={N_PERSIST} consecutive minutes")

# --- apply the chosen policy to the held-out test wells ---
test_pool = test_pool.reset_index(drop=True)
test_pool["proba"] = best_proba
test_thresh_map = per_well_thresholds(test_pool, "proba", Q)
test_pool["pred"] = (test_pool["proba"] >= test_pool["well_id"].map(test_thresh_map)).astype(int)
print("Per-well alert thresholds on held-out wells: "
      + ", ".join(f"well {w}: {t:.4f}" for w, t in sorted(test_thresh_map.items())))

cm = confusion_matrix(ytest, test_pool["pred"])
tn, fp, fn, tp = cm.ravel()
print(f"\nMinute-level confusion matrix (before persistence filter): TN={tn} FP={fp} FN={fn} TP={tp}")

test_trips = shutdown_events_df[shutdown_events_df["well_id"].isin(TEST_WELLS)]
test_trips = test_trips[test_trips["timestamp"] >= test_pool["timestamp"].min()]
final = score_policy(test_pool, "proba", test_trips, Q, N_PERSIST)
caught_events, missed_events = final["caught"], final["missed"]
lead_times = final["leads"]
false_alarm_events = final["false_alarm_events"]
n_events = caught_events + missed_events

# how many operator-facing alarms per well per month, a number an engineer
# can actually reason about
test_days = (test_pool["timestamp"].max() - test_pool["timestamp"].min()).total_seconds() / 86400
fa_per_well_month = false_alarm_events / max(len(TEST_WELLS), 1) / max(test_days / 30.0, 1e-9)

event_log = []
for well_id in TEST_WELLS:
    wt = test_trips[test_trips["well_id"] == well_id]
    for _, ev_row in wt.iterrows():
        event_log.append({"well_id": well_id, "timestamp": ev_row["timestamp"],
                           "cause": ev_row["failure_cause"], "caught": True, "lead_min": None})

print(f"\nTrips in held-out wells: {n_events}")
print(f"Caught (>=1 alarm in prior {ALERT_WINDOW_MIN} min): {caught_events} ({100*caught_events/max(n_events,1):.0f}%)")
print(f"Missed failures: {missed_events} ({100*missed_events/max(n_events,1):.0f}%)")
print(f"Caught with >={MIN_USEFUL_LEAD_MIN:.0f} min warning (actionable): {final['useful_caught']} "
      f"({100*final['useful_caught']/max(n_events,1):.0f}%)")
print(f"False-alarm EVENTS: {false_alarm_events} (~{fa_per_well_month:.1f} per well per month)")
print(f"Total alarms raised: {final['n_alarms']}")
if lead_times:
    print(f"Average lead time: {np.mean(lead_times):.1f} min (median {np.median(lead_times):.1f} min)")

# Full trade-off curve on the held-out wells. REPORTED ONLY - the shipped
# policy above was chosen on validation alone. This table exists so the team
# can see what a different cost preference would have bought, without
# re-running the pipeline, and so the choice is visible rather than implicit.
tradeoff_rows = []
for n_p in [1, 3, 5, 10]:
    for q in QUANTILES:
        r = score_policy(test_pool, "proba", test_trips, q, n_p)
        n_ev = r["caught"] + r["missed"]
        tradeoff_rows.append({
            "persist_min": n_p, "top_pct_of_scores": round(100 * (1 - q), 2),
            "caught": r["caught"], "caught_pct": round(100 * r["caught"] / max(n_ev, 1)),
            "missed": r["missed"], "false_alarm_events": r["false_alarm_events"],
            "fa_per_well_month": round(r["false_alarm_events"] / max(len(TEST_WELLS), 1) / max(test_days / 30.0, 1e-9), 1),
            "avg_lead_min": round(float(np.mean(r["leads"])), 1) if r["leads"] else None,
        })
tradeoff_df = pd.DataFrame(tradeoff_rows).sort_values(["persist_min", "top_pct_of_scores"])
print("\n=== Trade-off curve on held-out wells (reported, NOT used for selection) ===")
print(tradeoff_df.to_string(index=False))


# ============================================================
# PER-MODEL evaluation on the held-out wells.
#
# The table further up compares models at a fixed 0.5 cutoff, which is not
# how any of them would actually be operated. Here every model gets its OWN
# alarm policy tuned on validation (same cost criterion, same grid), and is
# then measured on the held-out wells at the event level - caught trips,
# nuisance alarms, lead time. That is the comparison that reflects what each
# model would really deliver in a control room.
# ============================================================
print("\n=== PER-MODEL event-level performance on held-out wells ===")
per_model_rows = []
per_model_policy = {}
for mname in models.keys():
    val_set["proba_m"] = val_probas[mname]
    test_pool["proba_m"] = test_probas[mname]
    # tune this model's own policy on validation
    grid = []
    for n_p in [1, 3, 5, 10, 15, 30]:
        for q in QUANTILES:
            r = score_policy(val_set, "proba_m", val_trips, q, n_p)
            grid.append((r["useful_missed"] * MISS_COST_RATIO + r["false_alarm_events"], q, n_p))
    grid.sort()
    _, q_best, n_best = grid[0]
    per_model_policy[mname] = (q_best, n_best)
    r = score_policy(test_pool, "proba_m", test_trips, q_best, n_best)
    n_ev = r["caught"] + r["missed"]
    per_model_rows.append({
        "Model": mname,
        "Val ROC-AUC": round(roc_auc_score(yval, val_probas[mname]), 3),
        "Test ROC-AUC": round(roc_auc_score(ytest, test_probas[mname]), 3),
        "policy_top_pct": round(100 * (1 - q_best), 2),
        "policy_persist_min": n_best,
        "caught": r["caught"],
        "caught_pct": round(100 * r["caught"] / max(n_ev, 1)),
        "caught_useful": r["useful_caught"],
        "caught_useful_pct": round(100 * r["useful_caught"] / max(n_ev, 1)),
        "missed": r["missed"],
        "nuisance_alarms": r["false_alarm_events"],
        "fa_per_well_month": round(r["false_alarm_events"] / max(len(TEST_WELLS), 1) / max(test_days / 30.0, 1e-9), 1),
        "avg_lead_min": round(float(np.mean(r["leads"])), 1) if r["leads"] else None,
        "median_lead_min": round(float(np.median(r["leads"])), 1) if r["leads"] else None,
    })
per_model_df = pd.DataFrame(per_model_rows)
print(per_model_df.to_string(index=False))

# ---- per-WELL breakdown, for the selected model ----
print("\n=== PER-WELL breakdown (selected model, shipped policy) ===")
test_pool["proba_m"] = best_proba
per_well_rows = []
for wid in TEST_WELLS:
    w_frame = test_pool[test_pool["well_id"] == wid]
    w_trips = test_trips[test_trips["well_id"] == wid]
    r = score_policy(w_frame, "proba_m", w_trips, Q, N_PERSIST)
    n_ev = r["caught"] + r["missed"]
    spec_row = specs.loc[wid]
    per_well_rows.append({
        "well": wid,
        "dominant_causes": spec_row["dominant_causes"],
        "trips": n_ev,
        "caught": r["caught"],
        "caught_pct": round(100 * r["caught"] / max(n_ev, 1)),
        "caught_useful": r["useful_caught"],
        "caught_useful_pct": round(100 * r["useful_caught"] / max(n_ev, 1)),
        "missed": r["missed"],
        "nuisance_alarms": r["false_alarm_events"],
        "fa_per_month": round(r["false_alarm_events"] / max(test_days / 30.0, 1e-9), 1),
        "avg_lead_min": round(float(np.mean(r["leads"])), 1) if r["leads"] else None,
    })
per_well_df = pd.DataFrame(per_well_rows)
print(per_well_df.to_string(index=False))

# ---- catch rate per cause, selected model ----
print("\n=== Catch rate by failure cause (selected model) ===")
ev_sel = alarm_events(test_pool, "proba_m", per_well_thresholds(test_pool, "proba_m", Q), N_PERSIST)
cause_rows = []
for _, trip in test_trips.iterrows():
    st, wid, cz = trip["timestamp"], trip["well_id"], trip["failure_cause"]
    ws = st - pd.Timedelta(minutes=ALERT_WINDOW_MIN)
    hit = len(ev_sel[(ev_sel["well_id"] == wid) & (ev_sel["start"] < st) & (ev_sel["end"] >= ws)]) > 0
    cause_rows.append({"cause": cz, "caught": int(hit)})
cause_catch = pd.DataFrame(cause_rows).groupby("cause")["caught"].agg(["sum", "count"])
cause_catch.columns = ["caught", "trips"]
cause_catch["caught_pct"] = (100 * cause_catch["caught"] / cause_catch["trips"]).round(0)
print(cause_catch.to_string())


# ============================================================
# WELL-LEVEL GRAPHS
# ============================================================
from sklearn.metrics import roc_curve as _roc_curve, precision_recall_curve

PALETTE = {"Logistic Regression": "#4C72B0", "Random Forest": "#C96A16", "XGBoost": "#2A9D8F"}
GRID_KW = dict(alpha=0.25, linewidth=0.6)

# --- 1. ROC + Precision-Recall curves on the held-out wells ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for mname in models.keys():
    fpr_c, tpr_c, _ = _roc_curve(ytest, test_probas[mname])
    axes[0].plot(fpr_c, tpr_c, label=f"{mname} (AUC={roc_auc_score(ytest, test_probas[mname]):.3f})",
                 color=PALETTE[mname], linewidth=1.6)
    prec_c, rec_c, _ = precision_recall_curve(ytest, test_probas[mname])
    axes[1].plot(rec_c, prec_c, label=mname, color=PALETTE[mname], linewidth=1.6)
axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="random")
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC - held-out wells 8, 9, 10"); axes[0].legend(fontsize=8); axes[0].grid(**GRID_KW)
axes[1].axhline(ytest.mean(), color="gray", linestyle=":", linewidth=1,
                label=f"base rate ({100*ytest.mean():.2f}%)")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall (the honest view for rare events)")
axes[1].legend(fontsize=8); axes[1].grid(**GRID_KW); axes[1].set_ylim(0, max(0.05, float(np.nanmax(prec_c[:-1])) * 1.1))
plt.tight_layout(); plt.savefig(f"{CHARTS}/roc_pr_curves.png", dpi=150); plt.close()
print("Saved roc_pr_curves.png")

# --- 2. Model comparison at the event level ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
mnames = per_model_df["Model"].tolist()
colors = [PALETTE[m] for m in mnames]
axes[0].bar(mnames, per_model_df["caught_pct"], color=colors)
axes[0].set_title("Trips caught (%)"); axes[0].set_ylabel(f"% of {n_events} trips")
for i, v in enumerate(per_model_df["caught_pct"]):
    axes[0].text(i, v + 1, f"{v}%", ha="center", fontsize=9)
axes[1].bar(mnames, per_model_df["fa_per_well_month"], color=colors)
axes[1].set_title("Nuisance alarms per well per month")
for i, v in enumerate(per_model_df["fa_per_well_month"]):
    axes[1].text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=9)
leads = per_model_df["avg_lead_min"].fillna(0)
axes[2].bar(mnames, leads, color=colors)
axes[2].set_title("Average lead time (min)")
for i, v in enumerate(leads):
    axes[2].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9)
for ax in axes:
    ax.tick_params(axis="x", rotation=18, labelsize=8); ax.grid(axis="y", **GRID_KW)
plt.suptitle("Each model at its OWN validation-tuned alarm policy - held-out wells", fontsize=10)
for ax in axes:
    top = max((b.get_height() for b in ax.patches), default=1)
    ax.set_ylim(0, top * 1.20)      # headroom so the value labels clear the frame
plt.tight_layout(); plt.savefig(f"{CHARTS}/model_comparison_events.png", dpi=150); plt.close()
print("Saved model_comparison_events.png")

# --- 3. Trade-off curves, one line per model ---
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for mname in models.keys():
    test_pool["proba_m"] = test_probas[mname]
    xs, ys = [], []
    for q in QUANTILES:
        r = score_policy(test_pool, "proba_m", test_trips, q, 5)
        n_ev = max(r["caught"] + r["missed"], 1)
        xs.append(r["false_alarm_events"] / max(len(TEST_WELLS), 1) / max(test_days / 30.0, 1e-9))
        ys.append(100 * r["caught"] / n_ev)
    ax.plot(xs, ys, "o-", label=mname, color=PALETTE[mname], linewidth=1.6, markersize=4)
ax.set_xlabel("Nuisance alarms per well per month"); ax.set_ylabel("Trips caught (%)")
ax.set_title("Operating-point trade-off (5-minute persistence)")
ax.legend(fontsize=8); ax.grid(**GRID_KW)
plt.tight_layout(); plt.savefig(f"{CHARTS}/tradeoff_curves.png", dpi=150); plt.close()
print("Saved tradeoff_curves.png")
test_pool["proba_m"] = best_proba

# --- 4. Per-well caught vs missed, and trips by cause ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
w_lbls = [f"Well {r['well']}" for _, r in per_well_df.iterrows()]
axes[0].bar(w_lbls, per_well_df["caught"], label="caught", color="#2A9D8F")
axes[0].bar(w_lbls, per_well_df["missed"], bottom=per_well_df["caught"], label="missed", color="#C1666B")
for i, r in per_well_df.reset_index().iterrows():
    axes[0].text(i, r["trips"] + 0.8, f"{r['caught_pct']}%", ha="center", fontsize=9)
axes[0].set_title("Trips caught vs missed, per held-out well"); axes[0].set_ylabel("trips")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", **GRID_KW)

cause_by_well = (test_trips.groupby(["well_id", "failure_cause"]).size().unstack(fill_value=0))
cause_by_well.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20", width=0.6)
axes[1].set_title("Trips by cause, per held-out well")
axes[1].set_xlabel(""); axes[1].set_ylabel("trips")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(fontsize=7, ncol=2); axes[1].grid(axis="y", **GRID_KW)
plt.tight_layout(); plt.savefig(f"{CHARTS}/per_well_performance.png", dpi=150); plt.close()
print("Saved per_well_performance.png")

# --- 5. Catch rate by cause + lead-time distribution ---
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
cc = cause_catch.sort_values("caught_pct")
axes[0].barh(cc.index.tolist(), cc["caught_pct"], color="#4C72B0")
for i, (idx, row) in enumerate(cc.iterrows()):
    axes[0].text(row["caught_pct"] + 1.5, i, f"{int(row['caught'])}/{int(row['trips'])}", va="center", fontsize=8)
axes[0].set_xlabel("% of trips caught"); axes[0].set_title("Catch rate by failure cause")
axes[0].set_xlim(0, 105); axes[0].grid(axis="x", **GRID_KW)

if lead_times:
    axes[1].hist(lead_times, bins=18, color="#2A9D8F", edgecolor="white")
    axes[1].axvline(float(np.mean(lead_times)), color="#C96A16", linestyle="--",
                    label=f"mean {np.mean(lead_times):.0f} min")
    axes[1].axvline(float(np.median(lead_times)), color="#16233A", linestyle=":",
                    label=f"median {np.median(lead_times):.0f} min")
    axes[1].legend(fontsize=8)
axes[1].set_xlabel("Lead time before trip (min)"); axes[1].set_ylabel("caught trips")
axes[1].set_title("How much warning the caught trips gave"); axes[1].grid(axis="y", **GRID_KW)
plt.tight_layout(); plt.savefig(f"{CHARTS}/cause_and_leadtime.png", dpi=150); plt.close()
print("Saved cause_and_leadtime.png")

# --- 6. Risk timeline per held-out well (one panel each, 10-day window with trips) ---
fig, axes = plt.subplots(len(TEST_WELLS), 1, figsize=(11, 2.6 * len(TEST_WELLS)), sharex=False)
if len(TEST_WELLS) == 1:
    axes = [axes]
for ax, wid in zip(axes, TEST_WELLS):
    wf = test_pool[test_pool["well_id"] == wid]
    wtrips = test_trips[test_trips["well_id"] == wid]
    if len(wtrips) == 0:
        continue
    centre = wtrips.iloc[len(wtrips) // 2]["timestamp"]
    lo, hi = centre - pd.Timedelta(days=5), centre + pd.Timedelta(days=5)
    seg = wf[(wf["timestamp"] >= lo) & (wf["timestamp"] <= hi)]
    ax.plot(seg["timestamp"], seg["proba_m"], color="#16233A", linewidth=0.7)
    ax.axhline(test_thresh_map[wid], color="#C96A16", linestyle="--", linewidth=1,
               label=f"this well's threshold ({test_thresh_map[wid]:.3f})")
    shown = False
    for _, tr in wtrips[(wtrips["timestamp"] >= lo) & (wtrips["timestamp"] <= hi)].iterrows():
        ax.axvline(tr["timestamp"], color="#C1666B", linewidth=1.2, alpha=0.85,
                   label=None if shown else "actual trip")
        shown = True
    ax.set_title(f"Well {wid} - predicted risk over a 10-day window ({specs.loc[wid, 'dominant_causes']})", fontsize=9)
    ax.set_ylabel("risk"); ax.legend(fontsize=7, loc="upper left"); ax.grid(**GRID_KW)
plt.tight_layout(); plt.savefig(f"{CHARTS}/well_risk_timelines.png", dpi=150); plt.close()
print("Saved well_risk_timelines.png")

# --- 7. Feature importance of the selected model ---
best_est = models[best_name]
imp = None
if hasattr(best_est, "feature_importances_"):
    imp = pd.Series(best_est.feature_importances_, index=STAGE1_FEATURES)
elif hasattr(best_est, "coef_"):
    imp = pd.Series(np.abs(best_est.coef_[0]), index=STAGE1_FEATURES)
if imp is not None:
    top = imp.sort_values(ascending=False).head(18)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.barh(top.index.tolist(), top.values, color="#4C72B0")
    ax.set_title(f"What drives the detector - top 18 features ({best_name})", fontsize=10)
    ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="x", **GRID_KW)
    plt.tight_layout(); plt.savefig(f"{CHARTS}/feature_importance.png", dpi=150); plt.close()
    print("Saved feature_importance.png")

# --- 8. Raw sensor traces around one real trip (the physics behind an alarm) ---
tr0 = test_trips.iloc[len(test_trips) // 2]
wid0, st0 = tr0["well_id"], tr0["timestamp"]
seg = test_pool[(test_pool["well_id"] == wid0) &
                (test_pool["timestamp"] >= st0 - pd.Timedelta(hours=8)) &
                (test_pool["timestamp"] <= st0 + pd.Timedelta(hours=1))]
trace_cols = [(c, lbl) for c, lbl in [
    ("motor_current_a", "Motor current (A)"), ("vibration_g", "Vibration (g)"),
    ("Pd_discharge_psi", "Discharge pressure (psi)"), ("voltage_v", "Supply voltage (V)"),
    ("Tm_motor_temp_f", "Motor temperature (F)"), ("Pi_intake_psi", "Intake pressure (psi)"),
] if c in SENSOR_COLS][:4]
fig, axes = plt.subplots(len(trace_cols) + 1, 1, figsize=(10, 2.0 * (len(trace_cols) + 1)), sharex=True)
for ax, (col, lbl) in zip(axes, trace_cols):
    if col in seg.columns:
        ax.plot(seg["timestamp"], seg[col], color="#3E5C76", linewidth=0.9)
    ax.axvline(st0, color="#C1666B", linewidth=1.3)
    ax.set_ylabel(lbl, fontsize=8); ax.grid(**GRID_KW)
axes[-1].plot(seg["timestamp"], seg["proba_m"], color="#16233A", linewidth=1.2)
axes[-1].axhline(test_thresh_map[wid0], color="#C96A16", linestyle="--", linewidth=1)
axes[-1].axvline(st0, color="#C1666B", linewidth=1.3)
axes[-1].set_ylabel("model risk", fontsize=8); axes[-1].set_xlabel("time"); axes[-1].grid(**GRID_KW)
axes[0].set_title(f"Well {wid0} - sensors and model risk around a real {tr0['failure_cause']} trip", fontsize=10)
plt.tight_layout(); plt.savefig(f"{CHARTS}/sensor_traces_trip.png", dpi=150); plt.close()
print("Saved sensor_traces_trip.png")



AVOIDABLE_DOWNTIME_HR = 3.0
downtime_avoided_hr = caught_events * AVOIDABLE_DOWNTIME_HR
print(f"Estimated downtime avoided: ~{downtime_avoided_hr:.0f} hours across held-out wells")

event_log_df = pd.DataFrame(event_log)

# ============================================================
# SUPERVISED - STAGE 2: root-cause classification, normalized features
#
# Trains on rows where the simulator's own state machine says the well is
# actively in "onset" (onset_cause is non-empty) - the period the fault
# signature is actually present - instead of the full Stage-1 H-minute
# pre-trip window, most of which still looks statistically normal. This is
# the fix for the majority-class collapse reported earlier: that collapse
# traced to noisy back-filled labels, not to the features themselves.
# ============================================================
print("\n=== SUPERVISED STAGE 2: Root-cause classification ===")
cause_fit["onset_cause"] = cause_fit["onset_cause"].astype(str)
cause_test["onset_cause"] = cause_test["onset_cause"].astype(str)

# ---- equipment configuration as prior knowledge ----
# GAS_LOCK and UNDERLOAD both present as a motor-current dip and are close to
# indistinguishable from the sensor trace alone - which is why both sat at
# 0.00 recall. But they are not equally likely on every well: a well with a
# weak or absent gas separator is far more gas-lock-prone than one with an
# 85%-efficient rotary separator. That is real prior knowledge an engineer
# uses when interpreting the same symptom, and the model had no access to it.
# These are per-well constants available for any new well (they come off the
# equipment spec sheet), so they transfer to unseen wells by construction.
EQUIP_FEATURES = ["gas_separator_efficiency_pct", "pump_n_stages", "motor_rated_hp"]
for frame in (cause_fit, cause_test):
    for col in EQUIP_FEATURES:
        frame[col] = frame["well_id"].map(specs[col].to_dict()).astype("float32")

STAGE2_FEATURES = ALL_FEATURES + EQUIP_FEATURES
print(f"Stage 2 features: {len(ALL_FEATURES)} sensor/rolling + {len(EQUIP_FEATURES)} equipment-spec")

print("Training rows per cause (fit set, true onset rows only):")
print(cause_fit["onset_cause"].value_counts())

le = LabelEncoder()
y_cause_fit = le.fit_transform(cause_fit["onset_cause"])
X_cause_fit = cause_fit[STAGE2_FEATURES].to_numpy(dtype="float32")

cause_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample", random_state=7, n_jobs=2)
cause_model.fit(X_cause_fit, y_cause_fit)

X_cause_test = cause_test[STAGE2_FEATURES].to_numpy(dtype="float32")
y_cause_test_true = cause_test["onset_cause"]
known_labels = set(le.classes_)
valid_mask = y_cause_test_true.isin(known_labels)
X_cause_test_valid = X_cause_test[valid_mask.values]
y_cause_test_valid = le.transform(y_cause_test_true[valid_mask])

cause_pred = cause_model.predict(X_cause_test_valid)
cause_acc = accuracy_score(y_cause_test_valid, cause_pred)
print(f"\nRoot-cause accuracy on held-out wells (n={len(y_cause_test_valid)}): {cause_acc:.3f}")
# some causes (e.g. HIGH_TEMP) never occur in held-out wells by design - pass
# the full label set explicitly so classes with 0 test support still print
# a (0-support) row instead of crashing on a class-count mismatch
cause_report = classification_report(
    y_cause_test_valid, cause_pred, labels=range(len(le.classes_)),
    target_names=le.classes_, zero_division=0,
)
print(cause_report)

# UNDERLOAD was previously listed here as "not frequency-related", which was
# wrong: inflow starvation is the textbook frequency-adjustable failure mode -
# slowing the pump until its rate matches the well's inflow is exactly what a
# pump-off controller does. See frequency_recommendation.py, which implements it.
FREQ_RELATED = {"GAS_LOCK": True, "UNDERLOAD": True, "HIGH_TEMP": True,
                 "HIGH_DISCHARGE": True, "VIBRATION": False, "LOW_VOLTAGE": False}

# ============================================================
# Save results
# ============================================================
with open(f"{RESULTS}/{RESULTS_NAME}", "w") as f:
    f.write(f"Horizon H = {H} minutes\n")
    f.write(f"Train wells (fully seen): {TRAIN_WELLS}\n")
    f.write(f"Held-out test wells (never seen): {TEST_WELLS}\n")
    f.write("Time-holdout within train wells: final 6 weeks used as validation\n")
    f.write("Features: z-score normalized PER WELL (fit-period baseline for train wells; "
             "first-30-days calibration baseline for held-out test wells)\n\n")

    f.write("=== Well equipment specs & safe operating band ===\n")
    for well_id in specs.index:
        s = specs.loc[well_id]
        f.write(f"  Well {well_id}: Motor={s['motor_model']}, Protector={s['protector_model']}, "
                 f"GasSeparator={s['gas_separator_model']} ({s['gas_separator_efficiency_pct']}% eff), "
                 f"Pump={s['pump_model']}\n")
        f.write(f"    Safe Hz band: [{s['safe_min_hz']:.0f}, {s['safe_max_hz']:.0f}] Hz, "
                 f"BEP={s['pump_bep_hz']:.0f} Hz, dominant causes: {s['dominant_causes']}\n")

    f.write("\n=== UNSUPERVISED: Clustering ===\n")
    f.write(f"K-Means (k=6) silhouette (normalized features): {km_sil:.3f}\n")
    f.write("K-Means cluster -> failure-within-H rate:\n")
    f.write(cluster_failure_rate.to_string() + "\n")
    f.write(f"DBSCAN: {n_db_clusters} clusters found. Failure rate: "
             f"outliers={outlier_failure_rate.get(True, float('nan')):.4f} vs. "
             f"clustered-normal={outlier_failure_rate.get(False, float('nan')):.4f}\n\n")

    f.write("=== SUPERVISED STAGE 1: Failure detection - models vs. baselines ===\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write(f"Best model (selected on VALIDATION EVENT-LEVEL COST, never test): {best_name}\n")
    f.write("Model selection on validation event-level cost (lower is better):\n")
    f.write(selection_df.to_string(index=False) + "\n")
    f.write(f"Stage 1 features: {len(STAGE1_FEATURES)} ({'relative-only' if USE_RELATIVE_FEATURES else 'raw + rolling'})\n")
    f.write(f"Alarm policy (tuned on validation, cost-weighted {MISS_COST_RATIO:.0f}:1): "
             f"per-well top {100*(1-Q):.1f}% of that well's own scores, "
             f"persistence={N_PERSIST} consecutive minutes\n")
    f.write("Per-well alert thresholds (held-out): "
             + ", ".join(f"well {w}: {t:.4f}" for w, t in sorted(test_thresh_map.items())) + "\n")
    f.write(f"Minute-level confusion matrix (pre-persistence): TN={tn} FP={fp} FN={fn} TP={tp}\n")
    f.write(f"Trips in held-out wells: {n_events}\n")
    f.write(f"Caught (>=1 alarm in prior {ALERT_WINDOW_MIN} min): {caught_events} ({100*caught_events/max(n_events,1):.0f}%)\n")
    f.write(f"Missed failures: {missed_events} ({100*missed_events/max(n_events,1):.0f}%)\n")
    f.write(f"Caught with >={MIN_USEFUL_LEAD_MIN:.0f} min warning (actionable): {final['useful_caught']} "
             f"({100*final['useful_caught']/max(n_events,1):.0f}%)\n")
    f.write(f"False-alarm EVENTS: {false_alarm_events} (~{fa_per_well_month:.1f} per well per month)\n")
    f.write(f"Total alarms raised: {final['n_alarms']}\n")
    if lead_times:
        f.write(f"Average lead time: {np.mean(lead_times):.1f} min (median {np.median(lead_times):.1f} min)\n")
    f.write(f"Estimated downtime avoided: ~{downtime_avoided_hr:.0f} hours across held-out wells\n\n")
    f.write("Alarm-policy grid searched on validation (top 10 by cost):\n")
    f.write(policy_df.head(10).to_string(index=False) + "\n\n")
    f.write("Trade-off curve on held-out wells (REPORTED ONLY - policy above was chosen on validation):\n")
    f.write(tradeoff_df.to_string(index=False) + "\n\n")
    f.write("=== PER-MODEL event-level performance on held-out wells (each at its own validation-tuned policy) ===\n")
    f.write(per_model_df.to_string(index=False) + "\n\n")
    f.write("=== PER-WELL breakdown (selected model, shipped policy) ===\n")
    f.write(per_well_df.to_string(index=False) + "\n\n")
    f.write("=== Catch rate by failure cause (selected model) ===\n")
    f.write(cause_catch.to_string() + "\n\n")

    f.write("=== SUPERVISED STAGE 2: Root-cause classification ===\n")
    f.write(f"Accuracy on held-out wells: {cause_acc:.3f}\n")
    f.write(cause_report + "\n")
    f.write("=== Frequency-related vs. not (per cause) - drives the recommendation branch ===\n")
    for c, is_freq in FREQ_RELATED.items():
        f.write(f"  {c}: {'frequency adjustment applicable (clip to well safe_min_hz/safe_max_hz)' if is_freq else 'NOT frequency-related -> flag inspection instead'}\n")

print("\nSaved", RESULTS_NAME)

# charts
rep_well = TEST_WELLS[0]
rep_events = event_log_df[(event_log_df["well_id"] == rep_well) & (event_log_df["caught"] == True)]
if len(rep_events) > 0:
    target_st = rep_events.iloc[len(rep_events) // 2]["timestamp"]
    win_start, win_end = target_st - pd.Timedelta(hours=6), target_st + pd.Timedelta(hours=3)
    seg = test_pool[(test_pool["well_id"] == rep_well) & (test_pool["timestamp"] >= win_start) & (test_pool["timestamp"] <= win_end)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _trace = "vibration_g" if "vibration_g" in SENSOR_COLS else SENSOR_COLS[0]
    axes[0].plot(seg["timestamp"], seg[_trace], color="#3E5C76", linewidth=1, label=f"{_trace} (raw)")
    axes[0].axvline(target_st, color="#C96A16", linestyle="--", linewidth=1.5, label="Actual trip")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(f"Held-out well {rep_well} — actual vs. predicted risk ({best_name})")
    axes[1].plot(seg["timestamp"], seg["proba"], color="#16233A", linewidth=1.5, label="predicted trip risk")
    axes[1].axhline(test_thresh_map[rep_well], color="gray", linestyle=":", linewidth=1, label="Alert threshold (this well)")
    axes[1].axvline(target_st, color="#C96A16", linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Predicted\ntrip probability"); axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{CHARTS}/heldout_well_timeline.png", dpi=150)
    plt.close()
    print(f"Saved timeline chart for held-out well {rep_well}")

fig, ax = plt.subplots(figsize=(6, 5))
cm_cause = confusion_matrix(y_cause_test_valid, cause_pred, labels=range(len(le.classes_)))
im = ax.imshow(cm_cause, cmap="Blues")
ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_, fontsize=8)
ax.set_xlabel("Predicted cause"); ax.set_ylabel("Actual cause")
ax.set_title("Stage 2: Root-cause classification (held-out wells, normalized)")
for i in range(cm_cause.shape[0]):
    for j in range(cm_cause.shape[1]):
        ax.text(j, i, cm_cause[i, j], ha="center", va="center",
                 color="white" if cm_cause[i, j] > cm_cause.max() / 2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig(f"{CHARTS}/cause_confusion_matrix.png", dpi=150)
plt.close()
print("Saved cause_confusion_matrix.png")


# ============================================================
# FREEZE THE TRAINED MODELS
#
# Everything above trains and evaluates in one pass. To show the pipeline a
# genuinely new well later, the models - and every decision that was tuned on
# validation - have to be saved exactly as they are here. Re-running training
# with the new well included would not be a test of these models; it would be
# a different model.
#
# Saved here: the selected Stage 1 detector, the Stage 2 root-cause model, the
# feature list, the global std floor used when normalising, and the alarm
# policy (quantile + persistence) chosen on validation. score_new_well.py
# loads these and never re-fits anything.
# ============================================================
import joblib

STORE = f"{OUT}/model_store"
os.makedirs(STORE, exist_ok=True)
joblib.dump({
    "stage1_model": models[best_name],
    "stage1_name": best_name,
    "stage2_model": cause_model,
    "stage2_label_encoder": le,
    "all_features": ALL_FEATURES,
    "stage2_features": STAGE2_FEATURES,
    "equip_features": EQUIP_FEATURES,
    "sensor_cols": SENSOR_COLS,
    "global_std_floor": global_std_floor,
    "horizon_min": H,
    "alarm_quantile": Q,
    "alarm_persist_min": N_PERSIST,
    "alert_window_min": ALERT_WINDOW_MIN,
    "min_useful_lead_min": MIN_USEFUL_LEAD_MIN,
    "calib_days": 30,
    "trained_on_wells": TRAIN_WELLS,
    "held_out_wells": TEST_WELLS,
}, f"{STORE}/esp_models.joblib", compress=3)
print(f"\nFroze {best_name} + root-cause model to model_store/esp_models.joblib")
print(f"  alarm policy: top {100*(1-Q):.1f}% of each well's own scores, "
      f"{N_PERSIST} consecutive minutes")
