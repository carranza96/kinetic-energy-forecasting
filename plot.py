import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

# res = pd.read_csv("results/results.csv")
print()

def error_plots():
    res = pd.read_csv("results/results.csv")

    res = res[res["Model architecture"]=="LR"]
    res["key"] = res.apply(lambda x: x["Model architecture"] + " - PH:" + str(json.loads(x["Other params"])["Past history"]) 
                                           +  " - FH:" + str(json.loads(x["Other params"])["Forecasting horizon"]), axis=1)
    plt.figure()
    wape = sns.boxplot(x="Dataset", hue="Model architecture", y="WAPE", data=res)
    wape.figure.savefig("wape.png")

    plt.figure()
    mape = sns.boxplot(x="Dataset", hue="Model architecture", y="MAPE", data=res)
    mape.figure.savefig("mape.png")

    plt.figure()
    mse = sns.boxplot(x="Dataset", hue="Model architecture", y="MSE", data=res)
    mse.figure.savefig("mse.png")

    plt.figure()
    rmse = sns.boxplot(x="Dataset", hue="Model architecture", y="RMSE", data=res)
    rmse.figure.savefig("rmse.png")
    
    plt.figure(figsize=(15,5))
    mae = sns.boxplot(x="Dataset", hue="Model architecture", y="MAE", data=res)
    mae.figure.savefig("mae.png")
    
    plt.figure(figsize=(15,5))
    mae = sns.boxplot(x="Dataset", hue="Model architecture", y="MAE", data=res)
    mae.figure.savefig("mae.png")
    
    plt.figure(figsize=(15,5))
    # mae = sns.pointplot(x="Dataset", hue="key", y="MAE", data=res, join=False)
    mae = sns.barplot(x="Dataset", hue="key", y="MAE", data=res)
    mae.figure.savefig("mae2.png")
    
    
error_plots()

def plot_pred_single_instance():
    errors = pd.read_csv("results/FH60-PH240/2020-01-KE/metrics_XGBoost_0.csv")
    i = 1000#errors["WAPE"].idxmin()
    y = pd.read_csv("results/FH60-PH240/2020-01-KE/y.csv")
    o = pd.read_csv("results/FH60-PH240/2020-01-KE/o_XGBoost_0.csv")
    fig = plt.figure()
    plt.plot(y.loc[i].values[1:], label="GT")
    plt.plot(o.loc[i].values[1:], label="Pred")
    plt.legend()
    fig.suptitle("Predicción (WAPE {})".format(errors["WAPE"].loc[i]))
    fig.savefig("pred.png")

def plot_complete_test_window():
    y = pd.read_csv("results/FH60-PH240/2020-11-KE/y.csv")
    o_xgb = pd.read_csv("results/FH60-PH240/2020-11-KE/o_XGBoost_0.csv")
    o_lr = pd.read_csv("results/FH60-PH240/2020-11-KE/o_LR_0.csv")

    all_y = np.concatenate([y.iloc[i].values[1:] for i in range(0, len(y), 60)])
    all_o_xgb = np.concatenate([o_xgb.iloc[i].values[1:] for i in range(0, len(y), 60)])
    all_o_lr = np.concatenate([o_lr.iloc[i].values[1:] for i in range(0, len(y), 60)])

    fig = plt.figure()
    plt.plot(all_y, label="GT")
    plt.plot(all_o_xgb, label="Pred XGBoost")
    plt.plot(all_o_lr, label="Pred LR")
    
    plt.legend()
    fig.suptitle("November test last 10 days")
    fig.savefig("pred_all.png")
    
    fig = plt.figure()
    plt.plot(all_y[6000:8000], label="GT")
    plt.plot(all_o_xgb[6000:8000], label="Pred XGBoost")
    plt.plot(all_o_lr[6000:8000], label="Pred LR")
    
    plt.legend()
    fig.suptitle("November test last 10 days (Zoom 6000:8000)")
    fig.savefig("pred_all_zoom.png")
    
    


plot_complete_test_window()
