import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from skrebate import ReliefF, MultiSURF
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

# Taking the top nRetainedFeatures
def get_indsTopnFeatures(_importances, _n):
    return sorted(
        range(len(_importances)),
        key=lambda i: _importances[i], reverse=True
    )[:_n]

if len(sys.argv) < 3:
    print(
        "Possible usage: python3 featureSelection_Electrical.py " +
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
    data=np.zeros((50*3, 7)),
    columns=[
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "n_obs"
    ]
)

scores_df = pd.DataFrame(
    data=np.zeros((50*3*100, 8)),
    columns=[
        "feature",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "n_obs"
    ]
)
scores_df["feature"] = np.tile(np.arange(0, 100, 1), 150)

rank_df = pd.DataFrame(
    data=np.zeros((50*3*10, 8)),
    columns=[
        "rank",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "n_obs"
    ]
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

        # === === === === === === ===
        # FEATURE RANKING METHODS
        # From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
        # ReliefF
        tRlfF_start = process_time()
        RlfF = ReliefF(n_neighbors=7, n_jobs=-1) # From Cai, 2014
        RlfF.fit(X, y)
        tRlfF_stop = process_time()
        tRlfF = tRlfF_stop - tRlfF_start

        # MultiSURF
        tMSurf_start = process_time()
        MSurf = MultiSURF(n_jobs=-1)
        MSurf.fit(X, y)
        tMSurf_stop = process_time()
        tMSurf = tMSurf_stop - tMSurf_start

        # From scikit-learn
        # Mutual Information
        tMI_start = process_time()
        resMI = mutual_info_classif(X, y, n_neighbors=7, random_state=0, n_jobs=-1)
        tMI_stop = process_time()
        tMI = tMI_stop - tMI_start

        # ANOVA F-value
        tFT_start = process_time()
        resFT_stat, resFT_p = f_classif(X, y)
        tFT_stop = process_time()
        tFT = tFT_stop - tFT_start

        # Random forest ensemble data mining to increase information gain/reduce impurity
        tRF_start = process_time()
        rfGini = RandomForestClassifier(
            n_estimators=1000, criterion="gini", random_state=0, n_jobs=-1
        )
        rfGini.fit(X, y)
        tRF_stop = process_time()
        tRF = tRF_stop - tRF_start

        # === === === === === === ===
        # GETTING TOP N FEATURES
        rank_df.loc[count_r:count_r+9, "RlfF"] = get_indsTopnFeatures(
            RlfF.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+9, "MSurf"] = get_indsTopnFeatures(
            MSurf.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+9, "MI"] = get_indsTopnFeatures(
            resMI, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+9, "RFGini"] = get_indsTopnFeatures(
            rfGini.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+9, "FT"] = get_indsTopnFeatures(
            resFT_stat, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+9, "iteration"] = np.repeat([i], 10)
        rank_df.loc[count_r:count_r+9, "n_obs"] = np.repeat([n_obs], 10)
        count_r += 10
        
        scores_df.loc[count:count+99, "RlfF"] = RlfF.feature_importances_
        scores_df.loc[count:count+99, "MSurf"] = MSurf.feature_importances_
        scores_df.loc[count:count+99, "MI"] = resMI
        scores_df.loc[count:count+99, "RFGini"] = rfGini.feature_importances_
        scores_df.loc[count:count+99, "FT"] = resFT_stat

        scores_df.loc[count:count+99, "iteration"] = np.repeat([i], 100)
        scores_df.loc[count:count+99, "n_obs"] = np.repeat([n_obs], 100)
        count += 100

        # === === === === === === ===
        # GET ELAPSED TIME
        elapsed_times.at[count_time, "RlfF"] = tRlfF
        elapsed_times.at[count_time, "MSurf"] = tMSurf
        elapsed_times.at[count_time, "RFGini"] = tRF
        elapsed_times.at[count_time, "MI"] = tMI
        elapsed_times.at[count_time, "FT"] = tFT
        elapsed_times.at[count_time, "iteration"] = i
        elapsed_times.at[count_time, "n_obs"] = n_obs
        count_time += 1

with open(f"{dataset_name}ranks.pkl", "wb") as handle:
    pickle.dump(rank_df, handle)

with open(f"{dataset_name}feature_scores.pkl", "wb") as handle:
    pickle.dump(scores_df, handle)

elapsed_times.to_csv(f"{dataset_name}elapsed_times.csv", sep=',')

sys.exit(0)
