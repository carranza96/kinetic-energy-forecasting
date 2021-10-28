import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = pd.read_csv("results/results.csv")
print()
# jan = res[res["Dataset"]=="2020-01-KE"]

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