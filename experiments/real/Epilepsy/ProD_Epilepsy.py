import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time

from prod_fs import ProD
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", index_col=0)
Xdf = df.iloc[:,:-1]
ydf = df.iloc[:,-1]
X, Xtest, y, ytest = train_test_split(
    Xdf, ydf, test_size=0.2, random_state=0, stratify=ydf
)

# According to Canedo (2012), 3 % of 178 features = 6 features
nRetainedFeatures = 6

# Categories:
# 

sys.exit()
# === === === ===
# Carrying out feature selection for each dataset
elapsed_times = pd.Series(
    data=np.zeros(1),
    index=["ProD"]
)

scores_df = pd.DataFrame(
    data=np.zeros((X.shape[1], 2)),
    columns=[
        "feature",
        "ProD"
    ]
)
scores_df["feature"] = np.arange(0, X.shape[1], 1)

rank_df = pd.DataFrame(
    data=np.zeros((16, 2)),
    columns=[
        "rank",
        "ProD"
    ]
)
rank_df["rank"] = np.arange(0, 16, 1)

# Proposed algorithm
tProD_start = process_time()
prodRanker = ProD(
    integration_method="trapz", delta=500, bw_method="scott",
    k=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5,
    averaging_method="weighted"
)
prodRanker.fit(X, y)
tProD_stop = process_time()
tProD = tProD_stop - tProD_start

# === === === === === === ===
# GET ELAPSED TIME
elapsed_times.at["ProD"] = tProD

# === === === === === === ===
# GETTING TOP N FEATURES
rank_df.loc[:, "ProD"] = prodRanker.get_topnFeatures(nRetainedFeatures)

scores_df.loc[:, "ProD"] = prodRanker.feature_importances_

elapsed_times.to_csv("ProD_elapsed_times.csv")
rank_df.to_csv("ProD_rank.csv")
scores_df.to_csv("ProD_scores_df.csv")

sys.exit(0)
