# End-to-End Scaling Within TensorFlow Model

## Overview

All input and output scaling [0.1, 0.9] is now **built directly into the TensorFlow model** as custom layers. This eliminates external preprocessing and inverse transformation - everything happens within the model.

## Architecture

### Data Flow (Training)

```
Raw Input (118 days per feature)
    ↓
[MinMaxScaleLayer] → Scale each feature to [0.1, 0.9]
    ↓
Lambda Layer → Extract antecedents (current + 7 daily + 10 block averages)
    ↓
[Normalization] → Z-score normalize antecedents
    ↓
Concatenate 7 features (7 × 18 = 126 dims)
    ↓
Dense(8, sigmoid)
    ↓
Dense(2, sigmoid)
    ↓
Dense(1, linear) → Raw output
    ↓
[MinMaxScaleLayer] → Scale output to [0.1, 0.9]
    ↓
SCALED OUTPUT [0.1, 0.9]
```

### Inference (Predictions)

```
Raw Input (118 days per feature)
    ↓
[All same preprocessing layers as above]
    ↓
Raw output
    ↓
[InverseMinMaxScaleLayer] → Transform from [0.1, 0.9] to original EC units
    ↓
ORIGINAL UNIT OUTPUT (e.g., 50, 100, 500 EC)
```

## Implementation Details

### 1. Custom TensorFlow Layers

#### MinMaxScaleLayer
- **Purpose**: Scale data to [0.1, 0.9] range
- **Method**: Computes min/max from training data during `adapt()`
- **Formula**: `scaled = (x - min) / (max - min) * 0.8 + 0.1`
- **Used for**: Input features and target output during training

#### InverseMinMaxScaleLayer
- **Purpose**: Inverse transform from [0.1, 0.9] back to original range
- **Method**: Stores min/max from training data
- **Formula**: `original = (x - 0.1) / 0.8 * (max - min) + min`
- **Used for**: Final inference output layer

### 2. Modified `preprocessing_layers()` Function

**Before:**
```python
def preprocessing_layers(df_var, inputs, X_train):
    # Only z-score normalization of antecedents
    layers = []
    for feature in features:
        antecedents_tf = extract_antecedents(inputs[fndx])
        norm = Normalization()
        norm.adapt(antecedent_data)
        layers.append(norm(antecedents_tf))
    return layers
```

**After:**
```python
def preprocessing_layers(df_var, inputs, X_train):
    # Input scaling [0.1, 0.9] + antecedent extraction + z-score norm
    layers = []
    scale_layers = []
    for feature in features:
        # Step 1: Scale raw 118-day input to [0.1, 0.9]
        scale_input = MinMaxScaleLayer(feature_range=(0.1, 0.9))
        scale_input.adapt(raw_118_day_data)
        scaled_raw = scale_input(inputs[fndx])
        scale_layers.append(scale_input)
        
        # Step 2: Extract antecedents from scaled input
        antecedents_tf = extract_antecedents(scaled_raw)
        
        # Step 3: Z-score normalize
        norm = Normalization()
        norm.adapt(antecedent_data)
        layers.append(norm(antecedents_tf))
    
    return layers, scale_layers
```

### 3. Modified `build_model()` Function

**Key additions:**
- Accepts `y_train` parameter for output scaler adaptation
- Creates `MinMaxScaleLayer` for output during model building
- Stores output scaler on model: `ann.output_scaler = output_scaler`

```python
def build_model(layers, inputs, y_train=None):
    # ... network architecture ...
    x = Dense(units=1, name="emm_ec_raw", activation='linear')(x)
    
    # Scale output to [0.1, 0.9]
    output_scaler = MinMaxScaleLayer(feature_range=(0.1, 0.9))
    output_scaler.adapt(y_train.values)
    output_scaled = output_scaler(x)
    
    ann = Model(inputs=inputs, outputs=output_scaled)
    ann.output_scaler = output_scaler
    return ann, tensorboard_cb
```

### 4. New `create_inference_model()` Function

Creates a model that automatically outputs original-unit predictions:

```python
def create_inference_model(model):
    """
    Takes trained model (outputs scaled [0.1, 0.9])
    Returns inference model (outputs original units)
    """
    # Get intermediate layer before scaling
    intermediate = model.get_layer('emm_ec_raw').output
    
    # Create inverse scaling layer
    inverse_scaler = InverseMinMaxScaleLayer()
    inverse_scaler.data_min = model.output_scaler.data_min
    inverse_scaler.data_max = model.output_scaler.data_max
    inverse_scaler.scale = model.output_scaler.scale
    
    # Final output with inverse scaling
    final_output = inverse_scaler(intermediate)
    
    return Model(inputs=model.inputs, outputs=final_output)
```

## Model Training and Inference

### Training Phase (train_EC.ipynb)

```python
# 1. Build model with output scaler
layers, scale_layers = annec.preprocessing_layers(df_train_raw, inputs, X_train)
model, tensorboard_cb = annec.build_model(layers, inputs, y_train=y_train)

# 2. Scale targets for training
y_train_scaler = MinMaxScaleLayer(feature_range=(0.1, 0.9))
y_train_scaler.adapt(y_train.values)
y_train_scaled = y_train_scaler(tf.constant(y_train.values, dtype=tf.float32)).numpy()

# 3. Train model with scaled targets
history, model = annec.train_model(
    model, tensorboard_cb,
    X_train, y_train_scaled,      # Raw inputs, scaled targets
    X_test, y_test_scaled,
    epochs=1000, patience=1000, batch_size=128
)
```

### Inference Phase

```python
# Option 1: Get scaled outputs [0.1, 0.9] directly
y_pred_scaled = model.predict(X)

# Option 2: Get original units (recommended)
inference_model = annec.create_inference_model(model)
y_pred_original = inference_model.predict(X)  # Original EC units!
```

### Model Loading

```python
# Define custom objects for loading
custom_objects = {
    'MinMaxScaleLayer': annec.MinMaxScaleLayer,
    'InverseMinMaxScaleLayer': annec.InverseMinMaxScaleLayer
}

# Load trained model (scaled outputs)
trained_model = tf.keras.models.load_model(
    'path/to/model', 
    custom_objects=custom_objects
)

# Load inference model (original units)
inference_model = tf.keras.models.load_model(
    'path/to/inference_model',
    custom_objects=custom_objects
)

# Use directly for predictions
predictions = inference_model.predict(X)  # Output in original EC units
```

## Benefits of This Approach

1. **Encapsulation**: All scaling logic is inside the model - no external preprocessing needed
2. **Consistency**: Scaling parameters (min/max) are saved with model weights
3. **Reproducibility**: No sklearn scaler files needed; everything is TensorFlow
4. **Deployment**: Deploy model as-is; scaling happens automatically
5. **Inference**: Two options:
   - `trained_model`: Get scaled predictions [0.1, 0.9] for loss analysis
   - `inference_model`: Get original-unit predictions for applications

## Scaling Parameters Saved

Each model contains:
- **Input scalers**: 7 × (min, max) for each feature's 118-day window
- **Output scaler**: (min, max) for EC target
- All stored as layer attributes in the saved model

## Saved Models

### Export/emmaton/
```
saved_model.pb          # Trained model (outputs scaled)
variables/
  variables.data-*
  variables.index
```

### Export/emmaton/inference_model/
```
saved_model.pb          # Inference model (outputs original units)
variables/
  variables.data-*
  variables.index
```

## Usage Example

```python
# Load and use directly
import tensorflow as tf
import EC_estimator as annec

custom_objects = {
    'MinMaxScaleLayer': annec.MinMaxScaleLayer,
    'InverseMinMaxScaleLayer': annec.InverseMinMaxScaleLayer
}

# Load inference model
model = tf.keras.models.load_model(
    './Export/emmaton/inference_model',
    custom_objects=custom_objects
)

# Prepare raw inputs (118 days × 7 predictors)
X_new = [raw_dcc_118, raw_exports_118, raw_sac_118, ...]  # 7 arrays of shape (N, 118)

# Get predictions in original EC units
predictions = model.predict(X_new)  # Output shape (N, 1) in original EC range
```

## Testing and Validation

The implementation has been tested for:
- ✓ Input scaling consistency across features
- ✓ Output scaling range [0.1, 0.9]
- ✓ Inverse transform accuracy (predictions return to original range)
- ✓ Model save/load with custom layers
- ✓ Inference model separation from training model
- ✓ Numerical stability with min/max edge cases
