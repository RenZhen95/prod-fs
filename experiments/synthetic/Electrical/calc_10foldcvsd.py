import os, sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3 calc_10foldcvsdf.py <csvFile>")
else:
    csvFile = Path(sys.argv[1])

df = pd.read_csv(csvFile, index_col=0)

df30 = df[df["nObs"] == 30.0]
df50 = df[df["nObs"] == 50.0]
df70 = df[df["nObs"] == 70.0]

# Order of classifiers in tables
clf_order = ["kNN", "SVM", "NB", "LDA", "DT"]

print("Dataset with 30 observations")
print(round(df30[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

print("Dataset with 50 observations")
print(round(df50[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

print("Dataset with 70 observations")
print(round(df70[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

sys.exit(0)
