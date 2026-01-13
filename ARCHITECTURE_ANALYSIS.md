# Architecture Analysis: EC_estimator.py and train_EC.ipynb

## Summary
The current implementation has **partial embedding of preprocessing within the TensorFlow model**:
- ✅ **Input normalization**: Fully embedded in preprocessing layers (part of model)
- ❌ **Target scaling**: NOT embedded in the model (external sklearn preprocessing)
- ❌ **Output inverse-transform**: NOT embedded in the model (external sklearn inverse-transform)

---

## 1. INPUT PREPROCESSING (Embedded in Model)

### Location: `EC_estimator.py` - `preprocessing_layers()` function (lines 133-165)

**What happens:**
```
Raw 118-day input → Antecedents extraction → Z-score Normalization
```

### Pipeline Details:

1. **Antecedent Extraction** (part of model graph):
   - Raw 118-day window → extracted features (18 total):
     - Current day value (1)
     - Daily values for days 1-7 (7)
     - 10 block averages of 11-day periods (10)
   - This is done using `tf.keras.layers.Lambda` with `antecedent_from_raw()` function
   - **Embedded in the TensorFlow model** ✅

2. **Z-score Normalization** (part of model graph):
   - Uses `Normalization` layer from `tensorflow.keras.layers.experimental.preprocessing`
   - Adapts to training antecedents statistics (mean and std)
   - **Embedded in the TensorFlow model** ✅
   - Computed from: `antecedent_from_raw_np(station_raw)` on training data

### Code:
```python
def preprocessing_layers(df_var, inputs, X_train):
    layers = []
    for fndx, feature in enumerate(feature_names()):
        station_raw = np.asarray(X_train[fndx])  # Shape: (N, 118)
        
        # Step 1: Extract antecedents (done in TF graph)
        antecedents_tf = tf.keras.layers.Lambda(
            antecedent_from_raw,
            name=f"{feature}_antecedents"
        )(inputs[fndx])
        
        # Step 2: Compute antecedents for z-score adapt
        station_ant = antecedent_from_raw_np(station_raw)  # Shape: (N, 18)
        
        # Step 3: Z-score normalization (embedded in TF graph)
        norm = Normalization(name=f"{feature}_norm")
        norm.adapt(station_ant)
        
        layers.append(norm(antecedents_tf))  # <-- Added to model graph
    
    return layers
```

---

## 2. TARGET SCALING (NOT Embedded in Model)

### Location: `train_EC.ipynb` - Main Training Cell (lines 151-172)

**What happens:**
```
Raw target values → sklearn MinMaxScaler [0.1, 0.9] (EXTERNAL to model)
```

### Target Scaling Details:

1. **Scaling is EXTERNAL** (sklearn preprocessing):
   ```python
   from sklearn.preprocessing import MinMaxScaler
   
   y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
   y_scaler.fit(y_train)  # Fit on training data only
   
   y_train_scaled = y_scaler.transform(y_train)
   y_test_scaled = y_scaler.transform(y_test)
   ```
   - **NOT part of the TensorFlow model** ❌
   - Scaling parameters stored in `y_scaler` object

2. **Scaled targets passed to training**:
   ```python
   history, model = annec.train_model(
       model, tensorboard_cb, 
       X_train, y_train_scaled,      # <-- Already scaled [0.1, 0.9]
       X_test, y_test_scaled,        # <-- Already scaled [0.1, 0.9]
       epochs=1000,
       patience=1000,
       batch_size=128
   )
   ```

3. **Model output layer** (in `build_model()` - line 169):
   ```python
   output = Dense(units=1, name="emm_ec", activation='linear')(x)
   ```
   - Simple linear output, no scaling layers
   - Outputs predictions in [0.1, 0.9] space because targets are in that space
   - **The model doesn't apply scaling, just learns to match the scaled targets** ❌

---

## 3. OUTPUT INVERSE-TRANSFORM (NOT Embedded in Model)

### Location: `train_EC.ipynb` - Evaluation Cell (lines 194-199)

**What happens:**
```
Scaled predictions [0.1, 0.9] → sklearn inverse-transform → Original EC units
```

### Inverse-Transform Details:

1. **External inverse-transform** (sklearn):
   ```python
   y_train_pred_scaled = model.predict(X_train, verbose=0)
   y_test_pred_scaled  = model.predict(X_test, verbose=0)
   
   # Inverse-transform using EXTERNAL scaler
   y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
   y_test_pred  = y_scaler.inverse_transform(y_test_pred_scaled)
   ```
   - **NOT part of the TensorFlow model** ❌
   - Depends on `y_scaler` object being available

2. **Predictions then converted to original units**:
   ```python
   print(f"y_train_pred_scaled: [{y_train_pred_scaled.min():.4f}, ...]")  # [0.1, 0.9]
   print(f"y_train_pred: [{y_train_pred.min():.1f}, ...]")  # Original EC units
   ```

---

## 4. Comparison Table

| Component | Embedded in Model | Where | Technology |
|-----------|------------------|-------|------------|
| **Input MinMax Scaling** | ❌ NOT DONE | - | - |
| **Input Antecedent Extraction** | ✅ YES | `preprocessing_layers()` | `tf.keras.layers.Lambda` |
| **Input Z-score Normalization** | ✅ YES | `preprocessing_layers()` | `Normalization` layer |
| **Target Scaling** | ❌ NO | Notebook (external) | sklearn `MinMaxScaler` |
| **Output Scaling** | ❌ NO | Model output is raw | Linear activation |
| **Output Inverse-Transform** | ❌ NO | Notebook (external) | sklearn `inverse_transform` |

---

## 5. Current Data Flow

```
Training Phase:
┌─────────────────────────────────┐
│  Raw Data (118-day windows)     │
│  + Raw Targets                  │
└────────────┬────────────────────┘
             │
             ├─→ [EXTERNAL] sklearn scale targets [0.1, 0.9]
             │                              ↓
             │                    y_train_scaled
             │
             ├─→ [MODEL] Antecedent extraction (TF Lambda)
             │                              ↓
             ├─→ [MODEL] Z-score normalization (TF Norm layer)
             │                              ↓
             ├─→ [MODEL] Dense(8), Dense(2), Dense(1, linear)
             │                              ↓
             └─→ [MODEL] Output [0.1, 0.9] (matches y_train_scaled)
                                           ↓
                                    Train with MSE loss

Inference Phase:
┌────────────────────────┐
│  Raw Input (118-day)   │
└────────────┬───────────┘
             │
             ├─→ [MODEL] Antecedent extraction
             │                       ↓
             ├─→ [MODEL] Z-score normalization
             │                       ↓
             ├─→ [MODEL] Dense layers
             │                       ↓
             ├─→ [MODEL] Output in [0.1, 0.9]
             │                       ↓
             ├─→ [EXTERNAL] sklearn inverse-transform
             │                       ↓
             └─→ Prediction in original EC units
```

---

## 6. Issues with Current Approach

1. **Model Serialization Problem**:
   - Model can be saved/loaded independently
   - But `y_scaler` must be saved/loaded separately
   - Requires manual management of two separate objects

2. **Deployment Complexity**:
   - Inference requires both model AND scaler object
   - Harder to package as a single self-contained unit

3. **Asymmetry**:
   - Inputs have some processing embedded (antecedents, normalization)
   - But outputs don't have processing embedded
   - Inconsistent architecture

4. **No Input Scaling**:
   - Currently NO MinMax scaling [0.1, 0.9] for raw input values
   - Only normalization of antecedents
   - Different from the commit message goal

---

## 7. What the Commit Message Said

From commit a485ca4:
> "Implement end-to-end scaling within TensorFlow model: MinMax [0.1, 0.9] for inputs and outputs"

**Reality Check**:
- ✅ Claims: End-to-end scaling → Actually: Partial scaling
- ✅ Claims: Inputs and outputs → Actually: Only some inputs (antecedents normalized, but NOT MinMax scaled), NO outputs
- ✅ Claims: Within TensorFlow model → Actually: Targets/inverse-transform are external

The implementation doesn't match the commit message goals.

---

## 8. Recommended Fixes to Match Intent

### Option A: Full Embedding (Matches commit message)
Implement custom TensorFlow layers:
1. Add `MinMaxScaleLayer` for input scaling [0.1, 0.9]
2. Add `MinMaxScaleLayer` for output scaling [0.1, 0.9]
3. Add `InverseMinMaxScaleLayer` for output inverse-transform
4. Remove external sklearn preprocessing from notebook
5. Create inference model with automatic inverse-scaling built-in

### Option B: Keep External (Current state)
Accept the current architecture and update documentation to clarify that:
- Input normalization is embedded
- Target scaling is external
- This is acceptable for the project's needs

---

## Conclusion

The current implementation is a **hybrid approach**:
- ✅ Input preprocessing (antecedents + normalization) is embedded in the model
- ❌ Target preprocessing and output post-processing are external

This works fine but doesn't match the commit message's stated goal of "end-to-end scaling within TensorFlow model."
