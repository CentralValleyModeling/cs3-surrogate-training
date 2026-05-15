"""
test_ann_dss.py
---------------
Reads input features from a CalSim HEC-DSS file, processes monthly → daily
the same way the training pipeline does (forward-fill via resample('D').ffill()),
builds 118-day sliding windows, and runs the saved EC + X2 inference models.

Run:
    python test_ann_dss.py

Or open as a Jupyter notebook via VS Code (each # %% block is a cell).
"""

# %% ── Configuration ──────────────────────────────────────────────────────────

import os, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import pyhecdss
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths (relative to repository root) ──
DSS_FILE    = r"Inputs/dcr_base_1.dss"
MODEL_DIR   = r"Export"

# Window size must match training
WINDOW = 118

# EC stations to predict (all 12)
EC_STATIONS = [
    "RSAN007_EC", "ROLD024_EC", "RSAC054_EC", "RSAC081_EC",
    "RSAC092_EC", "SLMZU011_EC", "SLMZU003_EC", "MIDR_INTAKE_EC",
    "VICT_INTAKE_EC", "CVP_INTAKE_EC", "OLDR_CCF_EC", "RSAN018_EC",
]

# Predictor order used when building tensors (must match model input names)
PREDICTORS = ["dcc", "exports", "sac", "sjr", "tide", "net_dcd", "smscg"]

# Mapping: predictor name → model input tensor name
INPUT_NAMES = {
    "dcc":     "dcc_input",
    "exports": "exports_input",
    "sac":     "sac_input",
    "sjr":     "sjr_input",
    "tide":    "tide_input",
    "net_dcd": "net_dcd_input",
    "smscg":   "smscg_input",
}

# %% ── Step 0: Catalog the DSS file (so you can see what paths exist) ─────────

print("=" * 70)
print(f"DSS file: {DSS_FILE}")
print("=" * 70)

with pyhecdss.DSSFile(DSS_FILE) as _d:
    catalog = _d.read_catalog()
print(f"\nTotal records in DSS: {len(catalog)}")
print("\nFirst 30 paths:")
for p in catalog["pathname"][:30] if "pathname" in catalog.columns else catalog.iloc[:30, 0]:
    print(" ", p)


# %% ── Step 1: Read features using the same approach as training ──────────────
#
# add_pathnames() mirrors dssioutils_dcr_1_30cm.py exactly:
#   - reads each path from the DSS file
#   - resamples to daily (resample('D').ffill()) — no-op for already-daily data
#   - converts PeriodIndex to DatetimeIndex
#   - sums all paths together

def add_pathnames(dssfile, paths):
    total = 0
    for path in paths:
        df = list(pyhecdss.get_ts(dssfile, path))[0][0]
        df = df.resample('D').ffill()
        if isinstance(df.index, pd.PeriodIndex):
            df.index = df.index.to_timestamp()
        total = total + df.iloc[:, 0]
    return total


def read_features_from_dss(dssfile):
    """
    Read all 7 EC input features from the DSS file.

    Paths match the training data sources exactly (L2020A_DCP_EX scenario,
    C_SAC048 node, MONTEZUMA SMSCG). Mismatched paths are the primary cause
    of prediction errors when using a wrong scenario or node name.
    """
    sac = add_pathnames(dssfile, [
        '/CALSIM-SMOOTH/C_SAC048/FLOW//1DAY/L2020A_DCP_EX/',
        '/CALSIM/C_CSL004A/CHANNEL//1MON/L2020A/',
        '/CALSIM/C_CLV004/FLOW//1MON/L2020A_1EX_DEC_CLOSED_DCP_EX/',
        '/CALSIM/C_MOK019/CHANNEL//1MON/L2020A/',
    ])
    exports = add_pathnames(dssfile, [
        '/CALSIM/C_CAA003_TD/FLOW//1MON/L2020A_DCP_EX/',
        '/CALSIM/C_DMC000_TD/FLOW//1MON/L2020A_DCP_EX/',
        '/CALSIM/D408/FLOW//1MON/L2020A_DCP_EX/',
        '/CALSIM/D_SJR028_WTPDWS/FLOW//1MON/L2020A_DCP_EX/',
    ])
    dcc     = add_pathnames(dssfile, ['/CALSIM/DXC/GATE-DAYS-OPEN//1MON/L2020A/'])
    net_dcd = add_pathnames(dssfile, ['/CALSIM/NET_DICU/DICU_FLOW//1MON/L2020A/'])
    sjr     = add_pathnames(dssfile, ['/CALSIM-SMOOTH/C_SJR070/FLOW//1DAY/L2020A_DCP_EX/'])
    tide    = add_pathnames(dssfile, ['/DWR/SAN_FRANCISCO/STAGE-MAX-MIN//1DAY/ASTRO_NAVD_20170607/'])
    smscg   = add_pathnames(dssfile, ['/MONTEZUMA/SMSCG/GATE-OPERATE//1DAY/DCP_EX/'])

    df = pd.concat([dcc, exports, sac, sjr, tide, net_dcd, smscg],
                   axis=1, join='inner')
    df.columns = PREDICTORS
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna()
    return df


print("\nReading features from DSS...")
df = read_features_from_dss(DSS_FILE)
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"Total daily rows: {len(df)}")
print(f"\nFeature summary:")
print(df.describe().round(2))


# %% ── Step 1b: DIAGNOSTIC — export DSS inputs and compare against training data

TRAINING_REF_CSV  = os.path.join(os.path.dirname(MODEL_DIR), "Inputs", "data_dcr_base_1.csv")
DIAG_FEATURES_CSV = os.path.join(MODEL_DIR, "diag_dss_input_features.csv")

# Export the raw feature DataFrame so it can be opened in Excel for inspection
df.to_csv(DIAG_FEATURES_CSV)
print(f"\n[DIAG] Raw DSS input features saved → {DIAG_FEATURES_CSV}")
print(f"       Open this file to verify that predictor values look sensible.")

# Compare feature statistics against training reference
if os.path.exists(TRAINING_REF_CSV):
    ref_df = pd.read_csv(TRAINING_REF_CSV, index_col=0, parse_dates=True)
    ref_stats  = ref_df[PREDICTORS].describe().loc[["mean", "std", "min", "max"]]
    dss_stats  = df[PREDICTORS].describe().loc[["mean", "std", "min", "max"]]

    print("\n[DIAG] Feature statistics — DSS input vs training reference (data_dcr_base_1)")
    print(f"{'':12s} {'Feature':<12s} {'Train mean':>12s} {'DSS mean':>12s} "
          f"{'Train min':>12s} {'DSS min':>12s} {'Train max':>12s} {'DSS max':>12s}  STATUS")
    print("-" * 100)
    for feat in PREDICTORS:
        t_mean, d_mean = ref_stats.loc["mean", feat], dss_stats.loc["mean", feat]
        t_min,  d_min  = ref_stats.loc["min",  feat], dss_stats.loc["min",  feat]
        t_max,  d_max  = ref_stats.loc["max",  feat], dss_stats.loc["max",  feat]
        t_std          = ref_stats.loc["std",  feat]
        # Flag if DSS mean deviates more than 2 std from training mean
        flag = "*** OUT OF RANGE ***" if abs(d_mean - t_mean) > 2 * t_std else "OK"
        print(f"{'':12s} {feat:<12s} {t_mean:>12.2f} {d_mean:>12.2f} "
              f"{t_min:>12.2f} {d_min:>12.2f} {t_max:>12.2f} {d_max:>12.2f}  {flag}")

    # Export side-by-side comparison CSV
    comp = pd.concat(
        [ref_stats.add_suffix("_train"), dss_stats.add_suffix("_dss")],
        axis=1
    )[sorted([c for c in ref_stats.columns for _ in range(2)],
             key=lambda c: PREDICTORS.index(c))]
    comp_csv = os.path.join(MODEL_DIR, "diag_feature_comparison.csv")
    comp.to_csv(comp_csv)
    print(f"\n[DIAG] Side-by-side stats saved → {comp_csv}")
else:
    print(f"\n[DIAG] Training reference CSV not found at {TRAINING_REF_CSV} — skipping comparison.")

# Export a sample of windowed inputs for spot-checking (first 5 windows, all 7 predictors)
DIAG_WINDOWS_CSV = os.path.join(MODEL_DIR, "diag_sample_windows.csv")
_sample_rows = []
_temp_X, _temp_dates = [], []
_arr = df[PREDICTORS].to_numpy()
for t in range(WINDOW - 1, min(WINDOW - 1 + 5, len(df))):
    win = _arr[t - WINDOW + 1 : t + 1, :]   # (118, 7)
    row = {"window_end_date": df.index[t].date()}
    for fi, feat in enumerate(PREDICTORS):
        row[f"{feat}_mean"] = win[:, fi].mean().round(3)
        row[f"{feat}_min"]  = win[:, fi].min().round(3)
        row[f"{feat}_max"]  = win[:, fi].max().round(3)
    _sample_rows.append(row)
pd.DataFrame(_sample_rows).to_csv(DIAG_WINDOWS_CSV, index=False)
print(f"[DIAG] Sample window stats (first 5 windows) saved → {DIAG_WINDOWS_CSV}")
print(f"       Compare these per-window statistics against the training data range above.\n")


# %% ── Step 2: Build 118-day sliding windows ──────────────────────────────────

def make_windows(df, window=118):
    """
    Identical to EC_estimator.make_windows():
    For each day t from index (window-1) onwards, take rows [t-117 .. t].
    Returns X of shape (N, window, n_features) and the corresponding dates.
    """
    X_arr = df[PREDICTORS].to_numpy()   # (N_days, 7)
    windows, dates = [], []
    for t in range(window - 1, len(df)):
        windows.append(X_arr[t - window + 1 : t + 1, :])  # (118, 7)
        dates.append(df.index[t])
    X = np.stack(windows, axis=0)   # (N_windows, 118, 7)
    return X, dates


print("\nBuilding 118-day windows...")
X, dates = make_windows(df, WINDOW)
print(f"Windows shape: {X.shape}   ({X.shape[0]} predictions)")


# %% ── Step 3: Build model inputs dict ────────────────────────────────────────

def build_inputs(X_windows):
    """
    Split (N, 118, 7) array into 7 named input arrays of shape (N, 118),
    matching the model's expected input tensor names.
    """
    return {
        INPUT_NAMES[feat]: X_windows[:, :, i].astype(np.float32)
        for i, feat in enumerate(PREDICTORS)
    }


model_inputs = build_inputs(X)
print("\nModel input tensors:")
for k, v in model_inputs.items():
    print(f"  {k}: shape={v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")


# %% ── Step 4: Run EC inference models ────────────────────────────────────────

results = {"date": [d.strftime("%Y-%m-%d") for d in dates]}

print(f"\nRunning EC inference models...")
for station in EC_STATIONS:
    model_path = os.path.join(MODEL_DIR, station, "inference_model")
    if not os.path.isdir(model_path):
        print(f"  SKIP {station} — model not found at {model_path}")
        continue

    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]

    # Run in batches to avoid OOM on large datasets
    batch_size = 512
    preds = []
    for i in range(0, len(dates), batch_size):
        batch = {k: tf.constant(v[i:i+batch_size]) for k, v in model_inputs.items()}
        out = infer(**batch)
        # The output tensor is named "output_inverse_scale"
        pred = out["output_inverse_scale"].numpy().flatten()
        preds.append(pred)

    preds = np.concatenate(preds)
    results[station] = preds
    print(f"  {station}: [{preds.min():.1f}, {preds.max():.1f}] µS/cm  "
          f"(mean={preds.mean():.1f})")

results_df = pd.DataFrame(results)
results_df["date"] = pd.to_datetime(results_df["date"])
results_df = results_df.set_index("date")
print(f"\nPredictions shape: {results_df.shape}")


# %% ── Step 5: Run X2 inference model ─────────────────────────────────────────

x2_model_path = os.path.join(MODEL_DIR, "X2_DIS", "inference_model")
if os.path.isdir(x2_model_path):
    print("\nRunning X2 inference model...")
    # X2 uses only NDOI (≈ SAC + SJR + lateral - exports + net_dcd), smscg, tide
    # Approximate NDOI from available features: sac + sjr - exports - net_dcd
    # Adjust this formula if your DSS has a dedicated NDOI path
    ndoi = (df["sac"] + df["sjr"] - df["exports"] - df["net_dcd"]).clip(lower=0)
    ndoi_df = pd.DataFrame({"NDOI": ndoi, "smscg": df["smscg"], "tide": df["tide"]})
    ndoi_df = ndoi_df.dropna()

    X_x2_arr = ndoi_df.to_numpy()
    X_x2_windows, dates_x2 = [], []
    for t in range(WINDOW - 1, len(ndoi_df)):
        X_x2_windows.append(X_x2_arr[t - WINDOW + 1 : t + 1, :])
        dates_x2.append(ndoi_df.index[t])
    X_x2 = np.stack(X_x2_windows, axis=0)  # (N, 118, 3)

    x2_inputs = {
        "NDOI_input":  X_x2[:, :, 0].astype(np.float32),
        "smscg_input": X_x2[:, :, 1].astype(np.float32),
        "tide_input":  X_x2[:, :, 2].astype(np.float32),
    }

    x2_model = tf.saved_model.load(x2_model_path)
    x2_infer = x2_model.signatures["serving_default"]

    x2_preds = []
    for i in range(0, len(dates_x2), batch_size):
        batch = {k: tf.constant(v[i:i+batch_size]) for k, v in x2_inputs.items()}
        out = x2_infer(**batch)
        x2_preds.append(out["output_inverse_scale"].numpy().flatten())
    x2_preds = np.concatenate(x2_preds)

    x2_df = pd.Series(x2_preds, index=dates_x2, name="X2_DIS")
    print(f"  X2_DIS: [{x2_preds.min():.1f}, {x2_preds.max():.1f}] km  "
          f"(mean={x2_preds.mean():.1f})")
else:
    x2_df = None
    print("\nX2 model not found — skipping")


# %% ── Step 6: Save predictions to CSV ───────────────────────────────────────

out_csv = os.path.join(MODEL_DIR, "dss_inference_results.csv")
if x2_df is not None:
    out_df = results_df.join(x2_df, how="outer")
else:
    out_df = results_df
out_df.to_csv(out_csv)
print(f"\nPredictions saved → {out_csv}")
print(out_df.head(10).round(1).to_string())


# %% ── Step 7: Quick verification plots ───────────────────────────────────────

# Compare first few stations against each other to check plausibility
stations_to_plot = [s for s in EC_STATIONS if s in results_df.columns][:4]

fig, axes = plt.subplots(len(stations_to_plot), 1,
                         figsize=(14, 3 * len(stations_to_plot)), sharex=True)
if len(stations_to_plot) == 1:
    axes = [axes]

for ax, station in zip(axes, stations_to_plot):
    ax.plot(results_df.index, results_df[station], lw=0.8, label=station)
    ax.set_ylabel("EC (µS/cm)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Date")
fig.suptitle("EC Predictions from DSS Inputs", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "dss_inference_results.png"), dpi=120)
plt.show()
print("\nPlot saved → dss_inference_results.png")


# %% ── Step 8: Spot-check one window against expected range ───────────────────

# Show the last window's raw inputs so you can verify they look sensible
print("\n── Last window (118 days) raw inputs ──")
last_window = df.iloc[-WINDOW:]
print(last_window[PREDICTORS].describe().round(3).to_string())

print("\n── Prediction for last window ──")
for station in stations_to_plot:
    if station in results_df.columns:
        print(f"  {station}: {results_df[station].iloc[-1]:.1f} µS/cm")
