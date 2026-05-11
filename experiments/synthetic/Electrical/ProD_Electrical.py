import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from prod_fs import ProD

if len(sys.argv) < 3:
    print(
        "Possible usage: python3 ProD_Electrical.py " +
        "<processedDatasets> <dataset_name>"
    )
    sys.exit(1)
else:
    synthetic_datasets_pkl = Path(sys.argv[1])
    dataset_name = sys.argv[2]

nRetainedFeatures = 10 # Top 10 % of 100 (Canedo, 2012)

with open(synthetic_datasets_pkl, "rb") as handle:
    synthetic_datasets = pickle.load(handle)

# === === === ===
# Carrying out feature selection for each dataset
# 50 iterations x 3 different number of samples
elapsed_times = pd.DataFrame(
    data=np.zeros((50*3, 3)),
    columns=["ProD", "iteration", "n_obs"]
)

scores_df = pd.DataFrame(
    data=np.zeros((50*3*100, 4)),
    columns=["feature", "ProD", "iteration", "n_obs"]
)
scores_df["feature"] = np.tile(np.arange(0, 100, 1), 150)

rank_df = pd.DataFrame(
    data=np.zeros((50*3*10, 4)),
    columns=["rank", "ProD", "iteration", "n_obs"]
)
rank_df["rank"] = np.tile(np.arange(0, 10, 1), 150)

count_time = 0
count = 0
count_r = 0

for n_obs in synthetic_datasets.keys():
    print(f"n_obs: {n_obs}")
    n_obs_datasets = synthetic_datasets[n_obs]

    for i in n_obs_datasets.keys():
        print(f"Iteration {i} ... ")
        X = n_obs_datasets[i]['X']
        y = n_obs_datasets[i]['y']

        # Proposed algorithm
        tProD_start = process_time()
        prodRanker = ProD(
            integration_method="trapz", delta=500, bw_method="scott",
            k=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5
        )
        prodRanker.fit(X, y)
        tProD_stop = process_time()
        tProD = tProD_stop - tProD_start

        # === === === === === === ===
        # GETTING TOP N FEATURES
        rank_df.loc[count_r:count_r+9, "ProD"] = prodRanker.get_topnFeatures(nRetainedFeatures)
        rank_df.loc[count_r:count_r+9, "iteration"] = np.repeat([i], 10)
        rank_df.loc[count_r:count_r+9, "n_obs"] = np.repeat([n_obs], 10)
        count_r += 10
        
        scores_df.loc[count:count+99, "ProD"] = prodRanker.feature_importances_
        scores_df.loc[count:count+99, "iteration"] = np.repeat([i], 100)
        scores_df.loc[count:count+99, "n_obs"] = np.repeat([n_obs], 100)
        count += 100

        # === === === === === === ===
        # GET ELAPSED TIME
        elapsed_times.at[count_time, "ProD"] = tProD
        elapsed_times.at[count_time, "iteration"] = i
        elapsed_times.at[count_time, "n_obs"] = n_obs
        count_time += 1

with open(f"{dataset_name}ProD_ranks.pkl", "wb") as handle:
    pickle.dump(rank_df, handle)

with open(f"{dataset_name}ProD_feature_scores.pkl", "wb") as handle:
    pickle.dump(scores_df, handle)

elapsed_times.to_csv(f"{dataset_name}ProD_elapsed_times.csv", sep=',')

sys.exit(0)
