import datetime as dt
from typing import Union, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator

RAW_COLUMNS = ["start_time", "value"]
INDEX_COL = "Datetime"
VALUE_COL = "Value"
COLUMNS = [INDEX_COL, VALUE_COL]

FEATURES_METHODS = {"mean", "sum"}


def read_dataframe(filename: Union[str, List[str]]):
    if type(filename) is list:
        df = pd.concatenate([pd.read_csv(fn) for fn in filename])
    else:
        df = pd.read_csv(filename)
    df = df[RAW_COLUMNS]
    df.columns = COLUMNS
    df[VALUE_COL] = df[VALUE_COL].astype("float")
    df[INDEX_COL] = pd.to_datetime(df[INDEX_COL])
    df.index = df.pop(INDEX_COL)
    df[df[VALUE_COL] == 0.0] = np.nan
    df = df.dropna()
    return df


def transform_to_evenly_spaced(df_, freq="1min", method="backfill"):
    df = df_.copy()
    df = df.asfreq(freq, method=method)
    return df


def preprocess_time_series(df_, rolling_method="sum", rolling_window=1):
    assert (
        rolling_method in FEATURES_METHODS
    ), f"Rolling method not valid, it should be one of the following option {FEATURES_METHODS}"
    df = df_.copy()
    if rolling_method == "sum":
        df = df.rolling(rolling_window).sum()
    elif rolling_method == "mean":
        df = df.rolling(rolling_window).mean()
    return df


def _build_past_history_features(df_, past_history, value_column, train_lag=1):
    df = df_.copy()
    for i in range(past_history, 0, -1):
        df[f"X_{i}"] = df[value_column].shift(i * train_lag)
    return df


def _build_forecasting_horizon_features(df_, forecasting_horizon, value_column, test_lag=1):
    df = df_.copy()
    for i in range(forecasting_horizon):
        df[f"y_{i}"] = df[value_column].shift(-i * test_lag)
    return df


def build_features(df_, past_history, forecasting_horizon, value_column=VALUE_COL, train_lag=1, test_lag=1):
    df = df_.copy()
    df = _build_past_history_features(df, past_history, value_column, train_lag)
    df = _build_forecasting_horizon_features(df, forecasting_horizon, value_column, test_lag)
    df = df.drop(value_column, axis=1)
    df = df.dropna()
    return df


def split_train_test(df, end_date=None, n_days_test=10):
    if end_date == None:
        end_date = df.index.max()
    start_test = end_date - dt.timedelta(days=n_days_test)
    df_train = df[df.index <= start_test].copy()
    df_test = df[df.index > start_test].copy()
    return df_train, df_test


def scale_features(
    df_,
    scaler="MinMaxScaler",
    end_date=None,
    n_days_test=10,
    fit_scaler=True,
    value_column=VALUE_COL,
):
    df = df_.copy()

    sc = None
    if isinstance(scaler, BaseEstimator):
        sc = scaler
    if isinstance(scaler, str):
        if scaler.upper() == "MINMAXSCALER":
            sc = MinMaxScaler()
        elif scaler.upper() == "STANDARDSCALER":
            sc = StandardScaler()
    if sc == None:
        raise ValueError(
            "scaler must be instance of sklearn.base.BaseEstimator or one of the following string: 'MinMax', 'Standard'"
        )

    if end_date == None:
        end_date = df.index.max()

    if fit_scaler:
        start_test = end_date - dt.timedelta(days=n_days_test)
        sc = sc.fit(df[df.index <= start_test][[value_column]])

    df[[value_column]] = sc.transform(df[[value_column]])
    return df, sc


def split_input_output(df_):
    df = df_.copy()
    y_cols = [c for c in df.columns if c.startswith("y_")]
    X_cols = [c for c in df.columns if c.startswith("X_")]
    y = df[y_cols].values
    X = df[X_cols].values
    return X, y
