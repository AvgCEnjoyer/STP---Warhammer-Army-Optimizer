import pandas as pd

df = pd.read_csv("Results/results_1775003889.csv")

# Mittelwerte pro Konfiguration
grouped = df.groupby([
    "algorithm",
    "pop_size",
    "n_gen"
]).mean()

print(grouped.sort_values("hv", ascending=False).head(10))