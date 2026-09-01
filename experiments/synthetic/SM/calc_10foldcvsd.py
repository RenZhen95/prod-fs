import os, sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3 calc_10foldcvsdf.py <csvFile>")
else:
    csvFile = Path(sys.argv[1])

df = pd.read_csv(csvFile, index_col=0)

df2 = df[df["nClass"] == 2.0]
df3 = df[df["nClass"] == 3.0]
df4 = df[df["nClass"] == 4.0]

# Order of classifiers in tables
clf_order = ["kNN", "SVM", "NB", "LDA", "DT"]

print("Dataset with 2 classes")
print(round(df2[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

print("Dataset with 3 classes")
print(round(df3[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

print("Dataset with 4 classes")
print(round(df4[["Bal.Acc", "Clf"]].groupby("Clf").std(ddof=1).reindex(clf_order), 2))

sys.exit(0)
