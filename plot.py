import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


res = pd.read_csv("results/results.csv")
print()

def error_plots():
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
    
# error_plots()

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

y = pd.read_csv("results/FH60-PH240/2020-02-KE/y.csv")
o = pd.read_csv("results/FH60-PH240/2020-02-KE/o_RandomForest_0.csv")
all_y = np.concatenate([y.iloc[i].values[1:] for i in range(0, len(y), 60)])
all_o = np.concatenate([o.iloc[i].values[1:] for i in range(0, len(y), 60)])
fig = plt.figure()
plt.plot(all_y, label="GT")
plt.plot(all_o, label="Pred")
plt.legend()
fig.suptitle("RF Feb test last 10 days")
fig.savefig("pred_all.png")


