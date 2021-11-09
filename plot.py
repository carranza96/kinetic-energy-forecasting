from collections import defaultdict
import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, List

import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bokeh.io import output_file, save, show
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Select, Legend, MultiSelect, Div, HoverTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import Colorblind8
import iqplot

from experiments.utils.metrics import METRICS

PALETTE = Colorblind8


def expand_df(df_, json_column="Other params"):
    df = df_.copy()
    json_df = pd.json_normalize(df["Other params"].apply(json.loads))
    df = df.join(json_df)
    return df


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
    ls = []
    for metric in metrics:
        plt.figure()
        fig = sns.boxplot(x="Dataset", hue=hue_key, y=metric, data=res)
        fig.figure.savefig(f"{metric}_by_{hue_key.lower().replace(' ','_')}.png")
        ls.append(fig)
    return ls


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
    return fig


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
    return fig, fig_zoom


def _plot_best_predictions_by_forecasting_horizon(
    results_path: str, fh: int, ph: int, metric: str, models=None, show_all_horizons=True
):
    if show_all_horizons:
        horizons = list(range(fh))
    else:
        horizons = [0, fh - 1]

    # Read result for the specific forecasting horizon and past history
    problem_conf = f"FH{fh}-PH{ph}"
    results = pd.read_csv(f"{results_path}/results.csv")
    if models:
        results = results[results["Model architecture"].isin(models)]
    results = expand_df(results, "Other params")
    results = results[(results["Forecasting horizon"] == fh) & (results["Past history"] == ph)]

    # get best model for each architecture-dataset
    results = results.sort_values(metric, ascending=True).drop_duplicates(["Dataset", "Model architecture"])

    datasets = sorted(list(results["Dataset"].unique()))
    model_architetures = sorted(list(results["Model architecture"].unique()))

    # Prepare data
    data = defaultdict(dict)
    for ds in datasets:
        y = pd.read_csv(f"{results_path}/{problem_conf}/{ds}/y.csv").drop("Datetime", axis=1).values
        for model_arch in model_architetures:
            filtered_result = results[
                (results["Dataset"] == ds) & (results["Model architecture"] == model_arch)
            ].copy()
            if not len(filtered_result):
                data[ds][f"{model_arch} mean"] = ["NaN"] * fh
                data[ds][model_arch] = ["NaN"] * fh
            else:
                model = filtered_result["Model"].values[0]
                preds = (
                    pd.read_csv(f"{results_path}/{problem_conf}/{ds}/o_{model}.csv").drop("Datetime", axis=1).values
                )
                # Save mean error value
                data[ds][f"{model_arch} mean"] = [METRICS[metric.lower()](y, preds)] * fh
                # save error for each forecasting horizon
                data[ds][model_arch] = [METRICS[metric.lower()](y[:, i], preds[:, i]) for i in range(fh)]

    # Prepare data for current view of the plot.
    default_dataset = sorted(list(data.keys()))[0]

    live_data = data[default_dataset].copy()
    live_data["x"] = list(range(fh))
    live_source = ColumnDataSource(live_data)

    # Create figure
    p = figure(
        title="Error for each prediction horizon of "
        + default_dataset
        + " \t(Best model of each type of architecture)",
        width=1200,
        height=600,
    )
    p.yaxis.axis_label = metric
    p.xaxis.axis_label = "Forecasting horizon"
    # Plot lines
    line_ls = []
    for i, model in enumerate(model_architetures):
        l_mean = p.line(
            x="x",
            y=f"{model} mean",
            source=live_source,
            line_width=2,
            line_color=PALETTE[i + 1],
            line_dash="dotted",
            line_alpha=0.8,
        )
        l = p.line(x="x", y=model, source=live_source, line_width=2, line_color=PALETTE[i + 1], legend_label=model)
        c = p.circle(x="x", y=model, source=live_source, color=PALETTE[i + 1])
        p.add_tools(
            HoverTool(
                description=f"{model} hover",
                renderers=[c],
                tooltips=[
                    ("Model", model),
                    ("FH", "@x"),
                    (metric, f"@{model}"),
                    (f"Mean {metric}", f"@{{{model} mean}}"),
                ],
            )
        )
        line_ls.append([l, l_mean, c])

    l = p.line(x=list(range(fh)), y=[0] * fh, line_color="black", line_dash="dotted", legend_label="Mean avg.")
    l.visible = False

    p.add_layout(p.legend[0], "right")

    # Create dropdown menu and add javascript logic
    dataset_select = Select(title="Dataset", value=default_dataset, options=datasets)
    model_select = MultiSelect(title="Models", value=model_architetures, options=model_architetures)

    callback = CustomJS(
        args={
            "dataset_select": dataset_select,
            "model_select": model_select,
            "data": data,
            "live_source": live_source,
            "title": p.title,
            "models": model_architetures,
            "model_lines": line_ls,
        },
        code="""
        console.log(data)
        console.log(live_source.data)
        console.log(dataset_select.value)
        console.log(model_select.value)
        const live_data = live_source.data;

        for (let i = 0; i < models.length; i++) {
            live_data[models[i]] = data[dataset_select.value][models[i]];
            live_data[models[i] + ' mean'] = data[dataset_select.value][models[i] + ' mean'];
            
            if (model_select.value.includes(models[i])){
                model_lines[i][0].visible = true;
                model_lines[i][1].visible = true;
                model_lines[i][2].visible = true;
            } else{ 
                model_lines[i][0].visible = false;
                model_lines[i][1].visible = false;
                model_lines[i][2].visible = false;
            }

        }

        title.text = "Error for each prediction horizon of " + dataset_select.value + " \t(Best model of each type of architecture)";
        live_source.change.emit();
        """,
    )

    dataset_select.js_on_change("value", callback)
    model_select.js_on_change("value", callback)

    return row(column(dataset_select, model_select), p)


def _plot_best_predictions(results_path: str, fh: int, ph: int, metric: str, models=None, show_all_horizons=True):
    if show_all_horizons:
        horizons = list(range(fh))
    else:
        horizons = [0, fh - 1]

    # Read result for the specific forecasting horizon and past history
    problem_conf = f"FH{fh}-PH{ph}"
    results = pd.read_csv(f"{results_path}/results.csv")
    if models:
        results = results[results["Model architecture"].isin(models)]
    results = expand_df(results, "Other params")
    results = results[(results["Forecasting horizon"] == fh) & (results["Past history"] == ph)]

    # get best model for each architecture-dataset
    results = results.sort_values(metric, ascending=True).drop_duplicates(["Dataset", "Model architecture"])

    datasets = sorted(list(results["Dataset"].unique()))
    model_architetures = sorted(list(results["Model architecture"].unique()))

    # Prepare data
    data = defaultdict(lambda: defaultdict(dict))
    for ds in datasets:

        y = pd.read_csv(f"{results_path}/{problem_conf}/{ds}/y.csv")
        x = pd.to_datetime(y.pop("Datetime")).values
        data[ds]["x"] = np.concatenate([x, [x.max() + i * (x[1] - x[0]) for i in range(1, fh)]])
        data[ds]["y"] = np.concatenate([y["0"].values, y.iloc[-1].values[1:]])

        for model_arch in model_architetures:
            filtered_result = results[
                (results["Dataset"] == ds) & (results["Model architecture"] == model_arch)
            ].copy()
            if not len(filtered_result):
                # Model not present for this dataset - Fill with NaNs
                data[ds][model_arch]["Metric Mean"] = "None"
                nan_pred = ["NaN" for _ in range(len(data[ds]["y"]))]
                for i in horizons:
                    data[ds][model_arch][f"t{i+1}"] = nan_pred
                    data[ds][model_arch][f"Metric t{i+1}"] = "None"
                data[ds][model_arch]["Mean"] = nan_pred
            else:
                model = filtered_result["Model"].values[0]
                # Save error value
                data[ds][model_arch]["Metric Mean"] = filtered_result[metric].values[0]

                # save prediction for each timestep by forecasting horizon
                preds = pd.read_csv(f"{results_path}/{problem_conf}/{ds}/o_{model}.csv")
                for i in horizons:
                    fh_preds = ["NaN"] * i + preds[str(i)].tolist() + ["NaN"] * (fh - i - 1)
                    data[ds][model_arch][f"t{i+1}"] = fh_preds
                    # Save metric by forecasting horizon
                    j = None if -fh + i + 1 == 0 else -fh + i + 1
                    fh_error = METRICS[metric.lower()](data[ds]["y"][i:j], preds[str(i)])
                    data[ds][model_arch][f"Metric t{i+1}"] = "{:.3f}".format(fh_error)
                # get average prediction for each timestep
                list_preds = [list(data[ds][model_arch][f"t{i+1}"]) for i in horizons]
                data[ds][model_arch]["Mean"] = [
                    np.mean([v[i] for v in list_preds if v[i] != "NaN"]) for i in range(len(data[ds]["x"]))
                ]

    # Prepare data for current view of the plot.
    default_dataset = sorted(list(data.keys()))[0]
    default_pred_horizon = "Mean"

    live_data = {m: data[default_dataset][m][f"{default_pred_horizon}"] for m in model_architetures}
    live_data["x"] = data[default_dataset]["x"]
    live_data["y"] = data[default_dataset]["y"]

    live_source = ColumnDataSource(live_data)

    # Create figure
    p = figure(
        title="Prediction on test (last 10 days) of "
        + default_dataset
        + " \t(Best model of each type of architecture)",
        width=1200,
        height=600,
        x_axis_type="datetime",
    )

    # Plot lines
    line_y = p.line(x="x", y="y", source=live_source, line_width=3, line_alpha=0.7, line_color=PALETTE[0])
    line_ls = [line_y]
    for i, model in enumerate(model_architetures):
        l = p.line(x=f"x", y=model, source=live_source, line_width=2, line_color=PALETTE[i + 1])
        line_ls.append(l)
    p.add_tools(
        HoverTool(
            renderers=[line_y],
            tooltips=[("date", "@x{%F %T}"), ("GT", "@y")] + [(m, f"@{m}") for m in model_architetures],
            formatters={"@x": "datetime"},
            mode="vline",
        )
    )

    # Create legend
    legend_labels = ["GT"] + [
        f"{model} ({metric}: {data[default_dataset][model][f'Metric {default_pred_horizon}']})"
        for model in model_architetures
    ]
    legend_items = list(zip(legend_labels, [[l] for l in line_ls]))
    legend = Legend(items=legend_items, location="top")
    p.add_layout(legend, "right")

    # Create dropdown menu and add javascript logic
    dataset_select = Select(title="Dataset", value=default_dataset, options=datasets)
    pred_horizon_select = Select(value="Mean", options=["Mean"] + [f"t{i+1}" for i in horizons])
    model_select = MultiSelect(title="Models", value=model_architetures, options=model_architetures)

    callback = CustomJS(
        args={
            "dataset_select": dataset_select,
            "pred_horizon_select": pred_horizon_select,
            "model_select": model_select,
            "data": data,
            "live_source": live_source,
            "title": p.title,
            "models": model_architetures,
            "model_lines": line_ls[1:],
            "model_legends": legend.items[1:],
            "metric": metric,
        },
        code="""
        console.log(model_select.value)
        const live_data = live_source.data;

        live_data['y'] = data[dataset_select.value]['y'];
        live_data['x'] = data[dataset_select.value]['x'];

        for (let i = 0; i < models.length; i++) {
            live_data[models[i]] = data[dataset_select.value][models[i]][pred_horizon_select.value];
            model_legends[i].label.value = model_legends[i].label.value.split('(')[0] + '(' + metric + ': ' + data[dataset_select.value][models[i]]['Metric '+pred_horizon_select.value] + ')';
            
            if (model_select.value.includes(models[i])){
                model_lines[i].visible = true;
            } else{ 
                model_lines[i].visible = false;
            }

        }

        title.text = "Prediction on test (last 10 days) of " + dataset_select.value + " \t(Best model of each type of architecture)";
        live_source.change.emit();
        """,
    )

    dataset_select.js_on_change("value", callback)
    pred_horizon_select.js_on_change("value", callback)
    model_select.js_on_change("value", callback)

    return row(column(dataset_select, pred_horizon_select, model_select), p)


def _plot_error_dist(results_path: str, fh: int, ph: int, metric: str, models=None):
    # Read result for the specific forecasting horizon and past history
    problem_conf = f"FH{fh}-PH{ph}"
    results = pd.read_csv(f"{results_path}/results.csv")
    if models:
        results = results[results["Model architecture"].isin(models)]
    results = expand_df(results, "Other params")
    results = results[(results["Forecasting horizon"] == fh) & (results["Past history"] == ph)]

    p = iqplot.stripbox(
        data=results,
        q="MAE",
        cats=["Dataset", "Model architecture"],
        title="Mean error distribution of each type of architecture",
        jitter=True,
        q_axis="y",
        width=1200,
        height=600,
        tooltips=[(metric, "@{MAE}"), ("Model", "@{Model}"), ("Model params", "@{Model params}")],
        box_kwargs=dict(fill_color=None, line_color="gray"),
        median_kwargs=dict(line_color="gray"),
        whisker_kwargs=dict(line_color="gray"),
        whisker_caps=True,
        toolbar_location="right",
    )

    return row(p)


def plot_best_predictions(
    results_path: str,
    out_file: str = "predictions.html",
    metric: str = "MAE",
    fh_ph_list: List[str] = None,
    models: List[str] = None,
    show_all_horizons: bool = True,
):
    output_file(out_file)
    if not fh_ph_list:
        fh_ph_list = sorted(os.listdir(results_path))
    tabs = []
    for fh_ph in fh_ph_list:
        if fh_ph == "results.csv":
            continue
        fh = int(fh_ph.split("-")[0].replace("FH", ""))
        ph = int(fh_ph.split("-")[1].replace("PH", ""))
        predictions = _plot_best_predictions(results_path, fh, ph, metric, models, show_all_horizons)
        boxplot = _plot_error_dist(results_path, fh, ph, metric, models)
        error_fh = _plot_best_predictions_by_forecasting_horizon(
            results_path, fh, ph, metric, models, show_all_horizons
        )

        page = Panel(child=column(predictions, error_fh, boxplot), title=f"FH {fh} | PH {ph}")
        tabs.append(page)

    tabs = Tabs(tabs=tabs)

    title = Div(text="<h1>Kinetic Energy Forecasting</h1>", sizing_mode="stretch_width")

    save(column(title, tabs))
    return tabs


if __name__ == "__main__":
    # error_plots(
    #     results_path="./results",
    #     model_architectures=["LinearRegression", "XGBoost"],
    #     metrics=["MAE", "WAPE", "MAPE", "MSE", "RMSE"],
    #     hue_key="key",
    # )
    # error_plots(
    #     results_path="./results",
    #     model_architectures=["LinearRegression", "XGBoost"],
    #     metrics=["MAE", "WAPE", "MAPE", "MSE", "RMSE"],
    #     hue_key="Model architecture",
    # )
    # plot_pred_single_instance(
    #     results_path="./results",
    #     forecasting_horizon=60,
    #     past_history=240,
    #     dataset="2020-01-KE",
    #     model="XGBoost_0",
    #     idx=1000,
    #     metric="WAPE",
    # )
    # plot_complete_test_window(
    #     results_path="./results",
    #     forecasting_horizon=60,
    #     past_history=240,
    #     dataset="2020-01-KE",
    #     models=["LinearRegression_0", "XGBoost_0"],
    #     zoom=(6000, 8000),
    # )
    # plot_best_predictions(
    #     results_path="./results3",
    #     out_file="predictions_freq1min_20daystrain_10daystest.html",
    #     metric="MAE",
    #     fh_ph_list=["FH60-PH240", "FH240-PH960", "FH240-PH1440", "FH240-PH1920"],  # None or [] will take all possible options
    #     models=None,  # None or [] will take all models
    #     show_all_horizons=False,  # If False, plot only mean, t1 and t{fh}
    # )
    
    # plot_best_predictions(
    #     results_path="./results_moretrain",
    #     out_file="predictions_freq1min_2monthstrain_10daystest.html",
    #     metric="MAE",
    #     fh_ph_list=["FH240-PH1440"],  # None or [] will take all possible options
    #     models=None,  # None or [] will take all models
    #     show_all_horizons=False,  # If False, plot only mean, t1 and t{fh}
    # )
    
    plot_best_predictions(
        results_path="./results",
        out_file="predictions_freq15min_2monthstrain_10daystest.html",
        metric="MAE",
        fh_ph_list=["FH96-PH384"],  # None or [] will take all possible options
        models=None,  # None or [] will take all models
        show_all_horizons=False,  # If False, plot only mean, t1 and t{fh}
    )

