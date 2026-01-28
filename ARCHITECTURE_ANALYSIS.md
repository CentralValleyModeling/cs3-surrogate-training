# Architecture Analysis: EC_estimator.py and train_EC.ipynb

## Summary
The current implementation has **full embedding of preprocessing within the TensorFlow model**:
- (Done!) **Input preprocessing**: Fully embedded (antecedent extraction + z-score normalization)
- (Done!) **Output scaling parameters**: Stored in model for inference
- (Done!) **Inference model**: Automatic inverse-scaling built-in

---

## 1. Current Training Configuration

### Target Stations (12 EC stations)
```
CVP_INTAKE_EC, MIDR_INTAKE_EC, OLDR_CCF_EC, ROLD024_EC,
RSAC054_EC, RSAC081_EC, RSAC092_EC, RSAN007_EC,
RSAN018_EC, SLMZU003_EC, SLMZU011_EC, VICT_INTAKE_EC
```

### Input Datasets (3 combined)
```
Inputs/data_base.csv
Inputs/data_base_ns.csv
Inputs/data_6k_tunnel.csv
```

### Training Parameters
- **WINDOW**: 118 days
- **calib_prop**: 0.8 (80% train, 20% validation)
- **Test period**: 1923-01-01 to 1939-12-31
- **Train period**: 1940-01-01 to 2015-12-31
- **epochs**: 2000
- **patience**: 1000
- **batch_size**: 128

### Input Features (7 predictors)
```
dcc, exports, sac, sjr, tide, net_dcd, smscg
```

---

## 2. INPUT PREPROCESSING (Embedded in Model)

### Location: `EC_estimator.py` - `preprocessing_layers()` function

**What happens:**
```
Raw 118-day input → Antecedent extraction (18 features) → Z-score Normalization
```

### Pipeline Details:

1. **Antecedent Extraction** (part of model graph):
   - Raw 118-day window → extracted features (18 total):
     - Current day value (1)
     - Daily values for days 1-7 (7)
     - 10 block averages of 11-day periods (10)
   - Done using `tf.keras.layers.Lambda` with `antecedent_from_raw()` function
   - **Embedded in the TensorFlow model** (Implemented!)

2. **Z-score Normalization** (part of model graph):
   - Uses `Normalization` layer from `tensorflow.keras.layers.experimental.preprocessing`
   - Adapts to training antecedents statistics (mean and std)
   - **Embedded in the TensorFlow model** (Implemented!)

---

## 3. MODEL ARCHITECTURE

### Dense Network Structure
```
Input: 7 × 118 raw values
    ↓
Antecedent extraction: 7 × 18 = 126 features
    ↓
Z-score normalization (per feature)
    ↓
Concatenate: 126 normalized features
    ↓
Dense(8, sigmoid)
    ↓
Dense(2, sigmoid, name="hidden")
    ↓
Dense(1, linear, name="emm_ec")
    ↓
Output: Scaled predictions [0.1, 0.9]
```

### Custom Layers in `EC_estimator.py`

#### MinMaxScaleLayer
- Scales inputs to [0.1, 0.9] range based on computed min/max
- Full serialization support (get_config + from_config)
- Stored on model as `model.output_scaler` for inference model creation

#### InverseMinMaxScaleLayer
- Inverse transform from [0.1, 0.9] back to original range
- Used by inference model to output original EC units
- Automatically created by `create_inference_model()`

---

## 4. TARGET SCALING

### Training Phase (External sklearn)
```python
from sklearn.preprocessing import MinMaxScaler

y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
y_train_scaled = y_scaler.fit_transform(y_train.values)
y_val_scaled = y_scaler.transform(y_val.values)
```

- sklearn `MinMaxScaler` used during training to scale targets
- Model learns to output in [0.1, 0.9] range
- Scaling parameters stored in model via `build_model(y_train=y_train)`

### Inference Phase (Embedded in Model)
```python
inference_model = annec.create_inference_model(model)
y_pred = inference_model.predict(X_test)  # Returns original EC units
```

- `create_inference_model()` creates model with `InverseMinMaxScaleLayer`
- Inverse scaling uses parameters stored during `build_model()`
- Single model call produces predictions in original EC units

---

## 5. METRICS COMPUTED

### Daily Metrics
- Train/Val/Test RMSE (daily)
- Train/Val/Test R² (daily)

### Monthly Aggregated Metrics
- Train/Val/Test RMSE (monthly)
- Train/Val/Test R² (monthly)

Dates are tracked during dataset combination to enable monthly aggregation:
```python
train_dates_combined = []
val_dates_combined = []
test_dates_combined = []
```

---

## 6. Data Flow Diagram

```
Training Phase:
┌─────────────────────────────────────────────┐
│  3 Datasets × 12 Stations                   │
│  (data_base, data_base_ns, data_6k_tunnel)  │
└───────────────────┬─────────────────────────┘
                    │
                    ├─→ Split: Test 1923-1939, Train 1940-2015
                    │
                    ├─→ Create windows (118 days)
                    │
                    ├─→ Apply calib_prop split (80/20)
                    │
                    ├─→ Combine 3 datasets
                    │
        ┌───────────┴───────────┐
        │                       │
        ↓                       ↓
   X_train, X_val          y_train, y_val
        │                       │
        │                       ├─→ sklearn MinMaxScaler [0.1, 0.9]
        │                       │
        ↓                       ↓
   [MODEL] preprocessing   y_train_scaled
   (antecedents + z-score)
        │                       │
        └───────────┬───────────┘
                    ↓
             Train with MSE loss

Inference Phase:
┌────────────────────────┐
│  Raw Input (7 × 118)   │
└───────────┬────────────┘
            │
            ├─→ [MODEL] Antecedent extraction
            │
            ├─→ [MODEL] Z-score normalization
            │
            ├─→ [MODEL] Dense layers
            │
            ├─→ [MODEL] Scaled output [0.1, 0.9]
            │
            ├─→ [INFERENCE MODEL] InverseMinMaxScaleLayer
            │
            ↓
   Predictions in original EC units
```

---

## 7. Model Serialization

### Training Model
- Saved to: `./Export/{station_name}/`
- Contains: Input normalization + Dense layers
- Outputs: Scaled [0.1, 0.9]
- Includes: `output_scaler` parameters stored during build

### Inference Model
- Saved to: `./Export/{station_name}/inference_model/`
- Contains: Everything in training model + Inverse scaling layer
- Outputs: Original EC units
- Load with custom_objects:

```python
custom_objects = {
    'MinMaxScaleLayer': annec.MinMaxScaleLayer,
    'InverseMinMaxScaleLayer': annec.InverseMinMaxScaleLayer
}
model = tf.keras.models.load_model(path, custom_objects=custom_objects)
```

---

## 8. Output Files

### Models (per station)
```
Export/{station_name}/
├── saved_model.pb          # Training model
├── keras_metadata.pb
├── variables/
│   ├── variables.data-00000-of-00001
│   └── variables.index
└── inference_model/
    ├── saved_model.pb      # Inference model
    └── variables/
```

### Results
```
Export/all_stations_results.csv
```

Columns:
- Station, Train_Samples, Val_Samples, Test_Samples
- Daily: Train_RMSE_Daily, Train_R2_Daily, Val_RMSE_Daily, Val_R2_Daily, Test_RMSE_Daily, Test_R2_Daily
- Monthly: Train_Months, Val_Months, Test_Months, Train_RMSE_Monthly, Train_R2_Monthly, Val_RMSE_Monthly, Val_R2_Monthly, Test_RMSE_Monthly, Test_R2_Monthly
- Ranges: y_train_min, y_train_max, y_test_min, y_test_max

---

## 9. Key Implementation Notes

1. **Random seed**: `np.random.seed(42)` reset before each station for reproducibility

2. **Memory management**: `tf.keras.backend.clear_session()` called after each station

3. **Combined training data**: All 3 datasets combined for normalization statistics

4. **Date tracking**: Dates tracked separately for train/val/test to enable monthly aggregation

5. **Validation split**: Applied per-dataset BEFORE combining (ensures balanced representation)
