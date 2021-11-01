import os
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from utils import data, metrics
from utils.models import CNN, LSTM, MLP, XGBoost, RandomForest, LinearRegression


SCALER = MinMaxScaler().fit([[130.0], [260.0]])
FIT_SCALER = False

# Model parameters
N_JOBS = -1
BATCH_SIZE = [64, 256]
EPOCHS = [1, 10, 50]
MODELS = {
    MLP: {
        "hidden_layers": [
                [8],
                [8, 16],
                [16, 8],
                [8, 16, 32],
                [32, 16, 8],
                [8, 16, 32, 16, 8],
                [32],
                [32, 64],
                [64, 32],
                [32, 64, 128],
                [128, 64, 32],
                [32, 64, 128, 64, 32]
        ],
        'input_shape':[None],
        'output_size':[None],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    CNN: {
        "conv_blocks": [
                [[32, 3, 2]],
                [[32, 5, 2], [32, 3, 2]],
                [[32, 7, 2], [32, 5, 2], [32, 2, 2]],
                [[32, 3, 0]],
                [[32, 5, 0], [32, 3, 0]],
                [[32, 7, 0], [32, 5, 0], [32, 2, 0]],
                [[64, 3, 2]],
                [[64, 5, 2], [64, 3, 2]],
                [[64, 7, 2], [64, 5, 2], [64, 2, 2]],
                [[64, 3, 0]],
                [[64, 5, 0], [64, 3, 0]],
                [[64, 7, 0], [64, 5, 0], [64, 2, 0]],
                [[128, 3, 2]],
                [[128, 5, 2], [128, 3, 2]],
                [[128, 7, 2], [128, 5, 2], [128, 2, 2]],
                [[128, 3, 0]],
                [[128, 5, 0], [128, 3, 0]],
                [[128, 7, 0], [128, 5, 0], [128, 2, 0]]
        ],

        'input_shape':[None],
        'output_size':[None],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    LSTM: {
        "layers": [1, 2, 4],
        "units": [32, 64, 128],
        "return_sequence": [True, False],
        'input_shape':[None],
        'output_size':[None],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    LinearRegression: {"n_jobs": [N_JOBS]},
    XGBoost: {
        "estimator__learning_rate": [0.05, 0.10, 0.25],
        "estimator__max_depth": [3, 10],
        "estimator__min_child_weight": [1, 5],
        "estimator__gamma": [0.1, 0.4],
        "estimator__colsample_bytree": [0.3, 0.7],
        "estimator__n_jobs": [N_JOBS],
    },
    RandomForest: {
        "n_estimators": [100, 500],
        "max_depth": [10, 100, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "bootstrap": [True, False],
        "n_jobs": [N_JOBS],
    }
}


def preprocess_data(df_, freq, past_history, forecasting_horizon, n_days_test):
    df = df_.copy()
    end_date = df.index.max()
    df = data.transform_to_evenly_spaced(df, freq=freq)
    df, scaler = data.scale_features(df, scaler=SCALER, fit_scaler=FIT_SCALER, n_days_test=n_days_test)
    df = data.build_features(df, past_history=past_history, forecasting_horizon=forecasting_horizon)
    df_train, df_test = data.split_train_test(df, end_date=end_date, n_days_test=n_days_test)
    X_train, y_train = data.split_input_output(df_train)
    X_test, y_test = data.split_input_output(df_test)
    return X_train, y_train, X_test, y_test, scaler, df_test.index


def evaluate(actual, predicted):
    ans = dict()
    for m in metrics.METRICS:
        ans[m] = metrics.METRICS[m](actual, predicted)
    return ans


def save_results(
    y_test_unscaled,
    y_pred_unscaled,
    errors,
    test_index,
    dataset_name,
    model_name,
    model_index,
    model_params,
    train_time,
    test_time,
    train_parameters,
    output_path,
    forecasting_horizon,
    past_history,
):
    out_path = f"{output_path}/FH{forecasting_horizon}-PH{past_history}/{dataset_name}"
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Save predictions
    y_pred_df = pd.DataFrame(y_pred_unscaled)
    y_pred_df.index = test_index
    y_pred_df.to_csv(f"{out_path}/o_{model_name}_{model_index}.csv", float_format="%.8f")

    # Save Y values
    y_test_df = pd.DataFrame(y_test_unscaled)
    y_test_df.index = test_index
    y_test_df.to_csv(f"{out_path}/y.csv", float_format="%.1f")

    # Save metrics
    errors_df = pd.DataFrame(errors)
    errors_df.columns = [c.upper() for c in errors_df.columns]
    errors_df = errors_df[[c for c in sorted(errors_df.columns)]]
    errors_df.index = test_index
    errors_df.to_csv(f"{out_path}/metrics_{model_name}_{model_index}.csv", float_format="%.8f")

    # Save model parameters
    models = {
        "Dataset": [dataset_name],
        "Model": [f"{model_name}_{model_index}"],
        "Model architecture": [model_name],
        "Model index": [model_index],
        "Model params": [json.dumps(model_params)],
        "Other params": [json.dumps(train_parameters)],
        "Training time": [train_time],
        "Inference time": [test_time],
    }
    models_df = pd.DataFrame(models)
    if Path(f"{out_path}/models.csv").exists():
        models_df_tmp = pd.read_csv(f"{out_path}/models.csv")
        models_df = models_df_tmp.append(models_df, ignore_index=True)
    models_df.to_csv(f"{out_path}/models.csv", index=False, float_format="%.8f")

    # Save summary of results
    results = metrics.evaluate_all(y_test_unscaled, y_pred_unscaled)
    results = {k.upper(): [results[k]] for k in sorted(results.keys())}
    results = {**models, **results}
    results = pd.DataFrame(results)
    if Path(f"{output_path}/results.csv").exists():
        results_tmp = pd.read_csv(f"{output_path}/results.csv")
        results = results_tmp.append(results, ignore_index=True)
    results.to_csv(f"{output_path}/results.csv", index=False, float_format="%.8f")


def main(data_path, output_path, past_history, forecasting_horizon, freq, n_days_test):
    files = sorted(os.listdir(data_path))

    for filename in files:
        dataset_name = filename.replace(".csv", "")
        print(dataset_name)

        df = data.read_dataframe(f"{data_path}/{filename}")
        X_train, y_train, X_test, y_test, scaler, test_index = preprocess_data(
            df, freq, past_history, forecasting_horizon, n_days_test
        )

        for model_func in MODELS:
            grid = ParameterGrid(MODELS[model_func])
            for idx, params in tqdm(list(enumerate(grid)), desc=model_func.__name__):
                if "input_shape" in params:
                    params["input_shape"] = (*X_train.shape, 1)
                if "output_size" in params:
                    params["output_size"] = y_train.shape[1]

                model = model_func()
                model.set_params(**params)

                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0

                t0 = time.time()
                y_pred = model.predict(X_test)
                inference_time = time.time() - t0

                y_pred_unscaled = scaler.inverse_transform(y_pred)
                y_test_unscaled = scaler.inverse_transform(y_test)

                errors = [evaluate(y, o) for y, o in zip(y_test_unscaled, y_pred_unscaled)]

                train_parameters = {
                    "Forecasting horizon": forecasting_horizon,
                    "Past history": past_history,
                    "Scaler": SCALER if type(SCALER) is str else type(SCALER).__name__,
                    "Data frequency": freq,
                    "Number days test": n_days_test,
                }
                save_results(
                    y_test_unscaled,
                    y_pred_unscaled,
                    errors,
                    test_index,
                    dataset_name,
                    model_func.__name__,
                    idx,
                    params,
                    train_time,
                    inference_time,
                    train_parameters,
                    output_path,
                    forecasting_horizon,
                    past_history,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--data-path", dest="data_path", type=str, default="./data")
    parser.add_argument("-o", "--output-path", dest="output_path", type=str, default="./results")
    parser.add_argument("-f", "--forecasting-horizon", dest="forecasting_horizon", type=int, default=60)
    parser.add_argument("-p", "--past-history", dest="past_history", type=int, default=240)
    parser.add_argument("-t", "--days-test", dest="n_days_test", type=int, default=10)
    parser.add_argument("-q", "--frequency", dest="freq", type=str, default="1min")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        output_path=args.output_path,
        past_history=args.past_history,
        forecasting_horizon=args.forecasting_horizon,
        freq=args.freq,
        n_days_test=args.n_days_test,
    )
