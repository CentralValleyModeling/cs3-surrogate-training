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


class MinMaxScaler091(Layer):
    """Custom min-max normalization layer that scales each feature to [0.1, 0.9].
    
    Uses per-feature (axis=0) min/max to preserve individual feature scales.
    Stores stats as tf.Variable for proper graph execution.
    """
    def __init__(self, **kwargs):
        super(MinMaxScaler091, self).__init__(**kwargs)
        self.min_val = None  # shape (F,) where F = number of antecedent features (18)
        self.max_val = None
        self.range_val = None  # renamed to avoid shadowing built-in
    
    def adapt(self, data):
        """Compute per-feature min and max from training data.
        
        Args:
            data: numpy array or tensor of shape (N, F) where F is features (18 antecedents)
        """
        data = tf.cast(data, tf.float32)
        # Per-feature min/max (axis=0) instead of global scalar
        mins = tf.reduce_min(data, axis=0)  # shape (F,)
        maxs = tf.reduce_max(data, axis=0)  # shape (F,)
        rng = maxs - mins
        # Prevent division by zero per feature
        rng = tf.maximum(rng, 1e-7)
        
        # Store as tf.Variable so they persist in graph execution
        self.min_val = tf.Variable(mins, trainable=False, name="min_val")
        self.max_val = tf.Variable(maxs, trainable=False, name="max_val")
        self.range_val = tf.Variable(rng, trainable=False, name="range_val")
    
    def call(self, x):
        """Scale each feature to [0.1, 0.9] range."""
        x = tf.cast(x, tf.float32)
        # Broadcasting: x is (batch, F), min_val/range_val are (F,)
        normalized = (x - self.min_val) / self.range_val
        # Clip to [0, 1] to handle values outside training range
        normalized = tf.clip_by_value(normalized, 0.0, 1.0)
        # Scale to [0.1, 0.9]
        return 0.1 + normalized * 0.8
    
    def get_config(self):
        config = super().get_config()
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
      raw 118-day input -> antecedents (18) -> Normalization
    X_train: list of 7 numpy arrays, each (N,118), used ONLY to adapt Normalization.
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

        # TF graph: raw -> antecedents -> normalize
        antecedents_tf = tf.keras.layers.Lambda(
            antecedent_from_raw,
            name=f"{feature}_antecedents"
        )(inputs[fndx])

        norm = MinMaxScaler091(name=f"{feature}_norm")
        norm.adapt(station_ant)

        layers.append(norm(antecedents_tf))

    return layers


def build_model(layers, inputs):
    """ Builds the standard CalSIM ANN
        Parameters
        ----------
        layers : list  
        List of tf.Layers

        inputs: dataframe
    """        

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=root_logdir)
    x = tf.keras.layers.concatenate(layers)
    
    # First hidden layer with 8 neurons and sigmoid activation function
    # Use glorot_uniform (Xavier) for sigmoid instead of he_normal (which is for ReLU)
    x = Dense(units=8, activation='sigmoid', input_dim=x.shape[1], kernel_initializer="glorot_uniform")(x)
    
    # Second hidden layer with 2 neurons and sigmoid activation function
    x = Dense(units=2, activation='sigmoid', kernel_initializer="glorot_uniform", name="hidden")(x) 
    
    # Output layer with 1 neuron
    output = Dense(units=1,name="emm_ec",activation="relu")(x)
    ann = Model(inputs = inputs, outputs = output)

    ann.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
        loss=root_mean_squared_error, 
        metrics=['mean_absolute_error']
    )
    
    return ann, tensorboard_cb



def train_model(model, tensorboard_cb, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=50, 
            mode="min", 
            restore_best_weights=True,
            verbose=1), 
            tensorboard_cb
        ], 
        batch_size=128, 
        epochs=200, 
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






