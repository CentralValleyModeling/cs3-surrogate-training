# EC_estimator.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
try:
    from tensorflow.keras.layers import Normalization
except ImportError:
    from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


# Custom MinMaxScaleLayer for storing scaling parameters
class MinMaxScaleLayer(tf.keras.layers.Layer):
    """Layer to apply MinMax scaling and store parameters for inverse transform"""
    def __init__(self, feature_range=(0.1, 0.9), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
    
    def adapt(self, data):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        feature_min, feature_max = self.feature_range
        self.scale = (feature_max - feature_min) / (self.data_max - self.data_min + 1e-10)
    
    def call(self, x):
        feature_min, feature_max = self.feature_range
        x_std = (x - self.data_min) / (self.data_max - self.data_min + 1e-10)
        return x_std * (feature_max - feature_min) + feature_min


class InverseMinMaxScaleLayer(tf.keras.layers.Layer):
    """Layer to apply inverse MinMax scaling"""
    def __init__(self, feature_range=(0.1, 0.9), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
    
    def call(self, x):
        feature_min, feature_max = self.feature_range
        x_std = (x - feature_min) / (feature_max - feature_min)
        return x_std * (self.data_max - self.data_min) + self.data_min


# Use 118-day window like EC, then extract 18 antecedents
num_feature_dims = {"NDOI" : 118, 
                    "smscg" : 118, 
                    "tide": 118, 
                    }

lags_feature = None

root_logdir = os.path.join(os.curdir, "tf_training_logs")


def feature_names():
    return list(num_feature_dims.keys())


def antecedent_from_raw(x):
    """
    Convert 118 raw daily values to 18 antecedent features (TensorFlow version).
    
    Structure:
    - Current day (1 value): index 0
    - Daily for 7 days (7 values): indices 1-7
    - 10 averaged blocks of 11 days each (10 values): indices 8-117
    
    Parameters
    ----------
    x : tf.Tensor
        Shape (batch, 118) - raw daily values
    
    Returns
    -------
    tf.Tensor
        Shape (batch, 18) - antecedent features
    """
    # x shape: (batch, 118)
    current = x[:, 0:1]            # 0d
    daily7  = x[:, 1:8]            # 1d..7d

    # 10 blocks of 11 days: [8..18], [19..29], ... [107..117]
    blocks = []
    start = 8
    for i in range(10):
        b = x[:, start + 11*i : start + 11*i + 11]          # 11 days
        blocks.append(tf.reduce_mean(b, axis=1, keepdims=True))  # MEAN of block
    blocks10 = tf.concat(blocks, axis=1)  # (batch, 10)

    return tf.concat([current, daily7, blocks10], axis=1)   # (batch, 18)


def antecedent_from_raw_np(x_np):
    """
    Convert 118 raw daily values to 18 antecedent features (NumPy version for adapt()).
    
    Structure:
    - Current day (1 value): index 0
    - Daily for 7 days (7 values): indices 1-7
    - 10 averaged blocks of 11 days each (10 values): indices 8-117
    
    Parameters
    ----------
    x_np : np.ndarray
        Shape (N, 118) - raw daily values
    
    Returns
    -------
    np.ndarray
        Shape (N, 18) - antecedent features
    """
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

def preprocessing_layers(df_var, inputs, X_train):
    """
    Build per-feature preprocessing layers:
      raw 118-day input -> antecedents (18) -> z-score normalization
    
    This matches the EC_estimator preprocessing approach.
    
    Parameters
    ----------
    df_var : pd.DataFrame
        Combined training data for statistics (not used directly, kept for API compatibility)
    inputs : list
        List of Input layers, one per feature
    X_train : list
        List of numpy arrays, each (N, 118), used to adapt Normalization layers
    
    Returns
    -------
    layers : list
        List of normalized antecedent tensors
    """
    layers = []
    for fndx, feature in enumerate(feature_names()):
        # (N,118) raw values for this feature
        station_raw = np.asarray(X_train[fndx])
        if station_raw.ndim != 2 or station_raw.shape[1] != num_feature_dims[feature]:
            raise ValueError(
                f"{feature}: expected X_train[{fndx}] shape (N,{num_feature_dims[feature]}), "
                f"got {station_raw.shape}"
            )

        # Build antecedents for adapt: (N,18)
        station_ant = antecedent_from_raw_np(station_raw)

        # TF graph: raw -> antecedents -> z-score normalization
        antecedents_tf = tf.keras.layers.Lambda(
            antecedent_from_raw,
            name=f"{feature}_antecedents"
        )(inputs[fndx])

        norm = Normalization(name=f"{feature}_norm")
        norm.adapt(station_ant)

        layers.append(norm(antecedents_tf))

    return layers





def build_model(layers, inputs, y_train=None):
    """ Builds the standard CalSIM ANN for X2
        The model outputs predictions in scaled space [0.1, 0.9].
        For inference in original units, use create_inference_model().
        
        Parameters
        ----------
        layers : list  
        List of tf.Layers

        inputs: list of Input layers
        
        y_train: numpy array (original units) for storing scaling parameters
    """        

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    x = Dense(units=8, activation='sigmoid', input_dim=x.shape[1])(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', name="hidden")(x) 
    
    # Output layer outputs directly in scaled space [0.1, 0.9]
    # No additional scaling layer - the model learns to output in this range
    output = Dense(units=1, name="x2_output", activation='linear')(x)
    
    ann = Model(inputs=inputs, outputs=output)

    ann.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mean_absolute_error']
    )
    
    # Store scaling parameters for creating inference model later
    if y_train is not None:
        y_data = np.asarray(y_train).reshape(-1, 1) if isinstance(y_train, pd.DataFrame) else np.asarray(y_train).reshape(-1, 1)
        output_scaler = MinMaxScaleLayer(feature_range=(0.1, 0.9), name="output_scale_params")
        output_scaler.build(input_shape=(None, 1))
        output_scaler.adapt(y_data)
        ann.output_scaler = output_scaler
    
    return ann, tensorboard_cb


def create_inference_model(model):
    """
    Create an inference model that outputs original (unscaled) values.
    Takes the scaled output [0.1, 0.9] from the trained model and applies inverse scaling.
    
    Parameters
    ----------
    model : tf.keras.Model
        The trained model that outputs scaled predictions [0.1, 0.9]
    
    Returns
    -------
    inference_model : tf.keras.Model
        Model that outputs inverse-scaled (original units) predictions
    """
    # Get the scaled output from the trained model [0.1, 0.9]
    scaled_output = model.output
    
    # Create inverse scaling layer using the stored scaler parameters
    inverse_scaler = InverseMinMaxScaleLayer(feature_range=(0.1, 0.9), name="output_inverse_scale")
    inverse_scaler.data_min = model.output_scaler.data_min.copy()
    inverse_scaler.data_max = model.output_scaler.data_max.copy()
    inverse_scaler.scale = model.output_scaler.scale.copy()
    
    # Apply inverse scaling: [0.1, 0.9] → original X2 units
    final_output = inverse_scaler(scaled_output)
    
    # Create inference model
    inference_model = Model(inputs=model.inputs, outputs=final_output)
    inference_model.output_scaler = model.output_scaler
    
    return inference_model


def train_model(model, tensorboard_cb, X_train, y_train, X_val, y_val,
                epochs=2000, patience=1000, batch_size=128, min_delta=0):
    """
    Train the model with early stopping and validation monitoring.
    
    Parameters match EC_estimator for consistency.
    """
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
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        callbacks=callbacks, 
        batch_size=batch_size, 
        epochs=epochs, 
        verbose=1
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
        'Train_R2': round(r2_train, 2),
        'Train_RMSE': round(rmse_train, 2),
        'Train_Percentage_Bias': round(percentage_bias_train, 2),
        'Test_R2': round(r2_test, 2),
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






