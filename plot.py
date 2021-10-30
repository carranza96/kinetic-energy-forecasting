from typing import Tuple, List
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json


def error_plots(results_path: str, model_architectures: List[str], metrics: List[str] = ["MAE"], hue_key: str = "key"):
    res = pd.read_csv(f"{results_path}/results.csv")

    res = res[res["Model architecture"].isin(model_architectures)]
    res["key"] = res.apply(
        lambda x: x["Model architecture"]
        + " - PH:"
        + str(json.loads(x["Other params"])["Past history"])
        + " - FH:"
        + str(json.loads(x["Other params"])["Forecasting horizon"]),
        axis=1,
    )

    for metric in metrics:
        plt.figure()
        fig = sns.boxplot(x="Dataset", hue=hue_key, y=metric, data=res)
        fig.figure.savefig(f"{metric}_by_{hue_key.lower().replace(' ','_')}.png")


def plot_pred_single_instance(
    results_path: str,
    forecasting_horizon: int,
    past_history: int,
    dataset: str,
    model: str,
    idx: int = None,
    metric: str = "MAE",
):
    dataset_path = f"{results_path}/FH{forecasting_horizon}-PH{past_history}/{dataset}"
    errors = pd.read_csv(f"{dataset_path}/metrics_{model}.csv")
    if not idx:
        idx = errors[metric].idxmin()
    y = pd.read_csv(f"{dataset_path}/y.csv")
    o = pd.read_csv(f"{dataset_path}/o_{model}.csv")
    fig = plt.figure()
    plt.plot(y.loc[idx].values[1:], label="GT")
    plt.plot(o.loc[idx].values[1:], label="Pred")
    plt.legend()
    fig.suptitle(f"Predicción ({metric}: {errors[metric].loc[idx]})")
    fig.savefig("pred.png")


def plot_complete_test_window(
    results_path: str,
    forecasting_horizon: int,
    past_history: int,
    dataset: str,
    models: List[str],
    zoom: Tuple[int, int],
):
    dataset_path = f"{results_path}/FH{forecasting_horizon}-PH{past_history}/{dataset}"

    fig = plt.figure()
    fig_zoom = plt.figure()

    y = pd.read_csv(f"{dataset_path}/y.csv")
    all_y = np.concatenate([y.iloc[i].values[1:] for i in range(0, len(y), forecasting_horizon)])
    fig.gca().plot(all_y, label="GT")
    fig_zoom.gca().plot(all_y[6000:8000], label="GT")

    for model in models:
        architecture_name = model.split("_")[0]

        o = pd.read_csv(f"{dataset_path}/o_{model}.csv")
        all_o = np.concatenate([o.iloc[i].values[1:] for i in range(0, len(y), forecasting_horizon)])

        fig.gca().plot(all_o, label=f"Pred {architecture_name}")
        fig_zoom.gca().plot(all_o[zoom[0] : zoom[1]], label=f"Pred {architecture_name}")

    fig.legend()
    fig.suptitle(f"{dataset.replace('-KE','')} test last 10 days")
    fig.savefig("pred_all.png")

    fig_zoom.legend()
    fig_zoom.suptitle(f"November test last 10 days (Zoom {zoom[0]}:{zoom[1]})")
    fig_zoom.savefig("pred_all_zoom.png")


if __name__ == "__main__":
    error_plots(
        results_path="./results",
        model_architectures=["LinearRegression", "XGBoost"],
        metrics=["MAE", "WAPE", "MAPE", "MSE", "RMSE"],
        hue_key="key",
    )
    error_plots(
        results_path="./results",
        model_architectures=["LinearRegression", "XGBoost"],
        metrics=["MAE", "WAPE", "MAPE", "MSE", "RMSE"],
        hue_key="Model architecture",
    )
    plot_pred_single_instance(
        results_path="./results",
        forecasting_horizon=60,
        past_history=240,
        dataset="2020-01-KE",
        model="XGBoost_0",
        idx=1000,
        metric="WAPE",
    )
    plot_complete_test_window(
        results_path="./results",
        forecasting_horizon=60,
        past_history=240,
        dataset="2020-01-KE",
        models=["LinearRegression_0", "XGBoost_0"],
        zoom=(6000, 8000),
    )
