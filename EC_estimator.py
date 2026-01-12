# EC_estimator.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.layers.experimental.preprocessing import Normalization #CategoryEncoding
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os


# ========================================================================
# Custom MinMax Scaling Layers [0.1, 0.9]
# ========================================================================

class MinMaxScaleLayer(Layer):
    """Scale inputs to [0.1, 0.9] range based on computed min/max."""
    
    def __init__(self, feature_range=(0.1, 0.9), **kwargs):
        super(MinMaxScaleLayer, self).__init__(**kwargs)
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
        self.min_range = feature_range[0]
        self.max_range = feature_range[1]
    
    def adapt(self, data):
        """Compute min/max from training data."""
        data = np.asarray(data)
        self.data_min = np.min(data, axis=0, keepdims=True)
        self.data_max = np.max(data, axis=0, keepdims=True)
        # Avoid division by zero
        self.scale = self.data_max - self.data_min
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
    
    def call(self, inputs):
        """Scale to [0.1, 0.9]."""
        if self.data_min is None or self.data_max is None:
            raise ValueError("Layer must be adapted before use. Call adapt() with training data.")
        
        data_min = tf.constant(self.data_min, dtype=inputs.dtype)
        scale = tf.constant(self.scale, dtype=inputs.dtype)
        
        # MinMax: (x - min) / (max - min)
        normalized = (inputs - data_min) / scale
        # Scale to [0.1, 0.9]
        scaled = normalized * (self.max_range - self.min_range) + self.min_range
        return scaled
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_range': self.feature_range,
            'data_min': self.data_min.tolist() if self.data_min is not None else None,
            'data_max': self.data_max.tolist() if self.data_max is not None else None,
            'scale': self.scale.tolist() if self.scale is not None else None,
        })
        return config


class InverseMinMaxScaleLayer(Layer):
    """Inverse transform from [0.1, 0.9] back to original range."""
    
    def __init__(self, feature_range=(0.1, 0.9), **kwargs):
        super(InverseMinMaxScaleLayer, self).__init__(**kwargs)
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
        self.min_range = feature_range[0]
        self.max_range = feature_range[1]
    
    def adapt(self, data):
        """Compute min/max from original data."""
        data = np.asarray(data)
        self.data_min = np.min(data, axis=0, keepdims=True)
        self.data_max = np.max(data, axis=0, keepdims=True)
        self.scale = self.data_max - self.data_min
        self.scale = np.where(self.scale == 0, 1.0, self.scale)
    
    def call(self, inputs):
        """Inverse scale from [0.1, 0.9] to original range."""
        if self.data_min is None or self.data_max is None:
            raise ValueError("Layer must be adapted before use. Call adapt() with training data.")
        
        data_min = tf.constant(self.data_min, dtype=inputs.dtype)
        scale = tf.constant(self.scale, dtype=inputs.dtype)
        
        # Inverse: (x - 0.1) / (0.9 - 0.1) * (max - min) + min
        normalized = (inputs - self.min_range) / (self.max_range - self.min_range)
        original = normalized * scale + data_min
        return original
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_range': self.feature_range,
            'data_min': self.data_min.tolist() if self.data_min is not None else None,
            'data_max': self.data_max.tolist() if self.data_max is not None else None,
            'scale': self.scale.tolist() if self.scale is not None else None,
        })
        return config


num_feature_dims = {"sac" : 118, 
                    "exports" : 118, 
                    "dcc": 118, 
                    "net_dcd" : 118, 
                    "sjr": 118, 
                    "tide" : 118, 
                    "smscg" : 118}

lags_feature = None

root_logdir = os.path.join(os.curdir, "tf_training_logs")


def feature_names():
    return list(num_feature_dims.keys())

def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

def load_data(file_name):
    return pd.read_csv(file_name)

def split_data(df, train_rows, test_rows):
    df_train = df.tail(train_rows)
    df_test = df.head(test_rows)
    return df_train, df_test


def build_model_inputs(df):
    inputs = []
    for feature,fdim in num_feature_dims.items():
        feature_input = Input(shape=(fdim,), name=f"{feature}_input")
        inputs.append(feature_input)
    return inputs

def antecedent_from_raw(x):
    # x shape: (batch, 118)
    current = x[:, 0:1]            # 0d
    daily7  = x[:, 1:8]            # 1d..7d

    # 10 blocks of 11 days: [8..18], [19..29], ... [107..117]
    blocks = []
    start = 8
    for i in range(10):
        b = x[:, start + 11*i : start + 11*i + 11]          # 11 days
        blocks.append(tf.reduce_mean(b, axis=1, keepdims=True))  # or sum, etc.
    blocks10 = tf.concat(blocks, axis=1)  # (batch, 10)

    return tf.concat([current, daily7, blocks10], axis=1)   # (batch, 18)


def calc_lags_feature(df):
    global lags_feature
    lags_feature = {feature: df.loc[:, pd.IndexSlice[feature,:]].columns.get_level_values(level='lag')[0:num_feature_dims[feature]] 
                    for feature in feature_names()}

def df_by_variable(df):
    """ Convert a dataset with a single index with var_lag as column names and convert to MultiIndex with (var,ndx)
        This facilitates queries that select only lags or only variables. As a side effect this routine will store
        the name of the active lags for each feature, corresponding to the number of lags in the dictionary num_feature_dims)
        into the module variable lag_features.

        Parameters
        ----------
        df : pd.DataFrame 
            The DataFrame to be converted

        Returns
        -------
        df_var : A DataFrame with multiIndex based on var,lag  (e.g. 'sac','4d')
    """
    indextups = []
    for col in list(df.columns):
        var = col
        lag = ""
        for key in num_feature_dims.keys():
            if col.startswith(key):
                var = key
                lag = col.replace(key,"").replace("_","")
                if lag is None or lag == "": lag = "0d"
                continue
        if var == "EC": lag = "0d"
        indextups.append((var,lag))
 
    ndx = pd.MultiIndex.from_tuples(indextups, names=('var', 'lag'))
    df.columns = ndx
    calc_lags_feature(df)
    return df

def antecedent_from_raw(x):
    # x shape: (batch, 118)
    current = x[:, 0:1]            # 0d
    daily7  = x[:, 1:8]            # 1d..7d

    blocks = []
    start = 8
    for i in range(10):
        b = x[:, start + 11*i : start + 11*i + 11]          # 11 days
        blocks.append(tf.reduce_mean(b, axis=1, keepdims=True))  # MEAN of block
    blocks10 = tf.concat(blocks, axis=1)  # (batch, 10)

    return tf.concat([current, daily7, blocks10], axis=1)   # (batch, 18)


def antecedent_from_raw_np(x_np):
    # x_np shape: (N, 118)
    current = x_np[:, 0:1]         # 0d
    daily7  = x_np[:, 1:8]         # 1d..7d

    blocks = []
    start = 8
    for i in range(10):
        b = x_np[:, start + 11*i : start + 11*i + 11]
        blocks.append(b.mean(axis=1, keepdims=True))        # MEAN of block
    blocks10 = np.concatenate(blocks, axis=1)               # (N,10)

    return np.concatenate([current, daily7, blocks10], axis=1)  # (N,18)


def preprocessing_layers(df_var, inputs, X_train):
    """
    Build per-feature preprocessing layers:
      raw 118-day input -> MinMaxScale [0.1,0.9] -> antecedents (18) -> z-score Normalization
    X_train: list of 7 numpy arrays, each (N,118), used ONLY to adapt layers.
    """
    layers = []
    scale_layers = []  # Store for later access if needed
    
    for fndx, feature in enumerate(feature_names()):
        # (N,118) raw values for this feature
        station_raw = np.asarray(X_train[fndx])
        if station_raw.ndim != 2 or station_raw.shape[1] != num_feature_dims[feature]:
            raise ValueError(
                f"{feature}: expected X_train[{fndx}] shape (N,{num_feature_dims[feature]}), "
                f"got {station_raw.shape}"
            )

        # Step 1: MinMax scale raw inputs to [0.1, 0.9]
        scale_input = MinMaxScaleLayer(feature_range=(0.1, 0.9), name=f"{feature}_scale_input")
        scale_input.adapt(station_raw)
        scaled_raw = scale_input(inputs[fndx])
        scale_layers.append(scale_input)

        # Step 2: Build antecedents from SCALED raw values
        antecedents_tf = tf.keras.layers.Lambda(
            antecedent_from_raw,
            name=f"{feature}_antecedents"
        )(scaled_raw)

        # Compute antecedents for z-score normalization adapt
        station_ant = antecedent_from_raw_np(station_raw)

        # Step 3: Z-score normalize antecedents
        norm = Normalization(name=f"{feature}_norm")
        norm.adapt(station_ant)

        layers.append(norm(antecedents_tf))

    return layers, scale_layers


def build_model(layers, inputs, y_train=None):
    """ Builds the standard CalSIM ANN with output scaling layers
        Parameters
        ----------
        layers : list  
        List of tf.Layers from preprocessing_layers

        inputs: list of Input layers
        
        y_train: numpy array for adapting output scaler (optional, for inference-only models)
    """        

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=x.shape[1])(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', name="hidden")(x) 
    
    # Output layer with 1 neuron and LINEAR activation (no scaling here)
    x = Dense(units=1, name="emm_ec_raw", activation='linear')(x)
    
    # Scale output to [0.1, 0.9]
    if y_train is not None:
        y_data = np.asarray(y_train).reshape(-1, 1) if isinstance(y_train, pd.DataFrame) else np.asarray(y_train).reshape(-1, 1)
    else:
        y_data = np.array([[0.0]])  # Dummy for inference model
    
    output_scaler = MinMaxScaleLayer(feature_range=(0.1, 0.9), name="output_scale")
    output_scaler.adapt(y_data)
    output_scaled = output_scaler(x)
    
    ann = Model(inputs=inputs, outputs=output_scaled)

    ann.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mean_absolute_error']
    )
    
    # Store scaler for later inverse transform
    ann.output_scaler = output_scaler
    
    return ann, tensorboard_cb


def create_inference_model(model):
    """
    Create an inference model that outputs original (unscaled) values.
    Extracts the intermediate layers and adds inverse scaling at the end.
    
    Parameters
    ----------
    model : tf.keras.Model
        The trained model with output scaling
    
    Returns
    -------
    inference_model : tf.keras.Model
        Model that outputs inverse-scaled (original units) predictions
    """
    # Get the outputs before scaling (emm_ec_raw layer)
    intermediate_output = model.get_layer('emm_ec_raw').output
    
    # Create inverse scaling layer
    inverse_scaler = InverseMinMaxScaleLayer(feature_range=(0.1, 0.9), name="output_inverse_scale")
    inverse_scaler.adapt(np.array([[model.output_scaler.data_min[0, 0], model.output_scaler.data_max[0, 0]]]))
    # Manually set the min/max from the trained model
    inverse_scaler.data_min = model.output_scaler.data_min
    inverse_scaler.data_max = model.output_scaler.data_max
    inverse_scaler.scale = model.output_scaler.scale
    
    # Create final output with inverse scaling
    final_output = inverse_scaler(model.get_layer('emm_ec_raw').output)
    
    # Create inference model
    inference_model = Model(inputs=model.inputs, outputs=final_output)
    inference_model.output_scaler = model.output_scaler
    
    return inference_model


def train_model(model, tensorboard_cb, X_train, y_train, X_test, y_test,
                epochs=1000, patience=1000, batch_size=128, min_delta=0,
                use_lr_scheduler=False, lr_factor=0.5, lr_patience=20, 
                lr_min_delta=1e-4, lr_min=1e-6):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=patience, 
            mode="min",
            min_delta=min_delta,
            restore_best_weights=True
        ), 
        tensorboard_cb
    ]
    
    if use_lr_scheduler:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=lr_patience,
                min_delta=lr_min_delta,
                min_lr=lr_min,
                verbose=1
            )
        )
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        callbacks=callbacks, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=0
    )
    return history, model

def calculate_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    y_train_np = y_train.values.ravel()
    y_train_pred_np = y_train_pred.ravel()

    # Calculate metrics for training data
    r2_train = r2_score(y_train_np, y_train_pred_np)
    rmse_train = np.sqrt(mean_squared_error(y_train_np, y_train_pred_np))
    percentage_bias_train = np.mean((y_train_pred_np - y_train_np) / y_train_np) * 100

    y_test_np = y_test.values.ravel()
    y_test_pred_np = y_test_pred.ravel()

    # Calculate metrics for test data
    r2_test = r2_score(y_test_np, y_test_pred_np)
    rmse_test = np.sqrt(mean_squared_error(y_test_np, y_test_pred_np))
    percentage_bias_test = np.mean((y_test_pred_np - y_test_np) / y_test_np) * 100

    # Return results as a dictionary
    return {
        'Model': model_name,
        'Train_R2': round(r2_train, 3),
        'Train_RMSE': round(rmse_train, 2),
        'Train_Percentage_Bias': round(percentage_bias_train, 2),
        'Test_R2': round(r2_test, 3),
        'Test_RMSE': round(rmse_test, 2),
        'Test_Percentage_Bias': round(percentage_bias_test, 2),
    }

def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def save_model(model, model_save_path):
    model.save(model_save_path)
    print(f"Model saved at location: {model_save_path}")

from tensorflow.keras.models import load_model

def load_model(model_path, loss_function):
    model = load_model(model_path, custom_objects={loss_function.__name__: loss_function})
    return model

def make_predictions(model, data, num_features):
    X_new = [data[feature] for feature in num_features]
    predictions = model.predict(X_new)
    return predictions


import joblib

def save_scaler(scaler, path):
    """Save a scaler object to disk using joblib."""
    joblib.dump(scaler, path)

def load_scaler(path):
    """Load a scaler object from disk using joblib."""
    return joblib.load(path)



