import os
import pandas as pd
import datetime as dt
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from time import time
import os

##############################
# 1. READ DATASET
##############################

# Read data (for instance only the 12 csv files that correspond to 2020)
data_path = "data"
# files = [f for f in sorted(os.listdir(data_path)) if '2020' in f]   # List all files
files = sorted(os.listdir(data_path))[12*5-3:]
print(files)
df = pd.concat([pd.read_csv(f"{data_path}/{filename}") for filename in files]) # Read into pandas dataframe

##############################
# 2. PREPROCESSING
##############################

# Format the pandas dataframe to put start_time at index (casted to datetime) and cast value to float
df = df[["start_time", "value"]]
df.columns = ["Datetime", "Value"]
df["Value"] = df["Value"].astype("float")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.index = df.pop("Datetime")

# Transform the data to evenly spaced series (i.e. fill minutes that do not have a value with next seen value)
df = df.asfreq("1min", method="backfill")


# Min-Max normalization (We found that the time series 
# historical minimum was 130 and the maximum 260)
SCALER = MinMaxScaler((-1,1)).fit([[130.0], [260.0]])
df[['Value']] = SCALER.transform(df[['Value']])

# Transform the dataset into a supervised machine learning problem (time series forecasting)
# In which for each past history window of size x, we have to predict the next y values (forecasting horizon)
# We define for instance a past history window of size 10 minutes, to predict the next 5 minutes
PAST_HISTORY = 1440
FORECASTING_HORIZON = 240

# Build past history features (Now each row (each timestamp) has PAST_HISTORY columns (x_t) with the previous values
for i in range(PAST_HISTORY, 0, -1):
        df[f"X_{i}"] = df["Value"].shift(i)
        
# Build forecasting horizon (Now each row also has FORECASTING_HORIZON columns (y_t) with future values to be predicted)
for i in range(FORECASTING_HORIZON):
        df[f"y_{i}"] = df["Value"].shift(-i)

# Now we need to drop rows with NaNs, because at the beginning and end of the dataset there are missing values when doing the windowing
# We also drop the Value column, since it is the same as X_1
df = df.drop("Value", axis=1)
df = df.dropna()


##############################
# 3. TRAIN/TEST SPLIT
##############################

# Now we have the dataset ready to be used within a supervised ML algorithm
# We just need to split the dataset into train/test
# For instance we take the first 9 months for trainin, and the last 3 months (90 days) for testing
end_date = df.index.max() # Get the last day of the dataset
start_test = end_date - dt.timedelta(days=90) # Get the day in which the test set should start
df_train = df[df.index <= start_test].copy()
df_test = df[df.index > start_test].copy()


# Split each dataframe into X (past history features) and y(values forecasting horizon)
def split_input_output(df_):
    df = df_.copy()
    y_cols = [c for c in df.columns if c.startswith("y_")]
    X_cols = [c for c in df.columns if c.startswith("X_")]
    y = df[y_cols].values
    X = df[X_cols].values
    return X, y

X_train, y_train = split_input_output(df_train)
X_test, y_test = split_input_output(df_test)


# The data preprocessing finishes at this point, and we are ready to train a ML model with the training set, and evaluate over the test set
################################################
# 5. MODEL TRAINING
################################################
use_lr = True # Flag to use LinearRegression or LSTM model
train = True # Flag to train a new model or load a model that was already trained

################################################
# 5.1 LINEAR REGRESSION
################################################
if use_lr:
    model_name = 'saved_models/lr_PH{}_FH{}_minmax.joblib'.format(PAST_HISTORY, FORECASTING_HORIZON)
    if train:
        model = LinearRegression()
        t = time()
        # Train Model
        model.fit(X_train, y_train)
        print("Train time: ", time() -t)
        # Save model using joblib
        dump(model,model_name)
    
    # Load trained model from file
    model = load(model_name)

################################################
# 5.2 LSTM
################################################
else:
    ## Change input shape for LSTM input
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    model_name = "saved_models/lstm_PH{}_FH{}_minmax.h5".format(PAST_HISTORY, FORECASTING_HORIZON)

    if train:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=X_train.shape[1:]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(FORECASTING_HORIZON),
        ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) 
        model.compile(optimizer=optimizer, loss='mae')
        t = time()
        history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=512, epochs=2)
        print("Train time: ", time() -t)
        model.save(model_name)
        
    model = tf.keras.models.load_model(model_name)

################################################
# 6. PREDICTION & EVALUATION
################################################

# Predict over test set
y_pred = model.predict(X_test)

# De-normalize predictions and labels to evaluate
y_pred = SCALER.inverse_transform(y_pred)
y_test = SCALER.inverse_transform(y_test)
# y_pred = np.round(SCALER.inverse_transform(y_pred))
# y_test = np.round(SCALER.inverse_transform(y_test))
# Calculate MAE TEST
mae = np.mean(np.abs(y_pred-y_test))
print("MAE Test: {}".format(mae))
print()

y_pred_train = model.predict(X_train)

y_pred_train = SCALER.inverse_transform(y_pred_train)
y_train = SCALER.inverse_transform(y_train)
mae = np.mean(np.abs(y_pred_train-y_train))
print("MAE Train: {}".format(mae))
print()

################################################
# 7. PLOTS
################################################

def plot_prediction(i, past_history, forecasting_horizon, X_test, y_test, y_pred, name):
    plt.figure()
    plt.plot(list(range(-past_history,0)),X_test[i], color='black', label="x")
    plt.scatter(list(range(forecasting_horizon)),y_test[i], color='orange', label="y")
    plt.scatter(list(range(forecasting_horizon)),y_pred[i], label="o")
    plt.legend()
    plt.title("MAE:{}".format(abs(y_test[i]- y_pred[i]).mean()))
    plt.savefig("figures/pred{}_FH{}_PH{}_I{}.png".format(name,forecasting_horizon,past_history,i))

def plot_history(history, past_history, forecasting_horizon, name):
    plt.figure()
    plt.plot(np.array(history.history['loss'])[2:] )#* (260.0-130.0))
    plt.plot(np.array(history.history['val_loss'])[2:]) #* (260.0-130.0))
    plt.title('Model error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("figures/loss_{}_FH{}_PH{}.png".format(name,forecasting_horizon,past_history))

if not use_lr:
    X_train, X_test = np.squeeze(X_train, -1), np.squeeze(X_test, -1)
    plot_history(history, PAST_HISTORY, FORECASTING_HORIZON, "LSTM")
    
plot_prediction(102, PAST_HISTORY, FORECASTING_HORIZON, SCALER.inverse_transform(X_test),  y_test,  y_pred, name="LR" if use_lr else "LSTM")
plot_prediction(100, PAST_HISTORY, FORECASTING_HORIZON, SCALER.inverse_transform(X_train),  y_train,  y_pred_train, name="LR" if use_lr else "LSTM")


