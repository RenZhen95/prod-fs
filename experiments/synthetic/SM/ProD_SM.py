import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from prod_fs import ProD

if len(sys.argv) < 2:
    print(
        "Possible usage: python3 ProD_SM.py <datasetFolder>"
    )
    sys.exit(1)
else:
    datasetFolder = Path(sys.argv[1])

nRetainedFeatures = 120 # Top 3 % of 4000 (Canedo, 2012)

nClass2 = pd.read_csv(datasetFolder.joinpath("SDI_nClass2.csv"), index_col=0)
nClass3 = pd.read_csv(datasetFolder.joinpath("SDI_nClass3.csv"), index_col=0)
nClass4 = pd.read_csv(datasetFolder.joinpath("SDI_nClass4.csv"), index_col=0)

# Let's just take the 1 dimension, 5 genes per dimension for now
def get_dataset_index(idxdf, ndim, ngenes):
    idxdf_ = idxdf[idxdf["nDimension"]==ndim]
    idxdf_ = idxdf_[idxdf_["nGenes_perDim"]==ngenes]
    return list(idxdf_.index)

# Canedo2012
# - 3-class, 25 samples per class
#    - 20 FCR from 2 groups with 10 genes each
#    - 10 FCR + 30 PCR from 4 groups with 10 genes each
#    - 60 PCR from 6 groups with 10 genes each
# Here:
# To make things simple, just take top 120 and use Canedo's success metric
#  - 3 dimensions, 20 genes
nClass2_sel_idx = get_dataset_index(nClass2, 3, 20)
nClass3_sel_idx = get_dataset_index(nClass3, 3, 20)
nClass4_sel_idx = get_dataset_index(nClass4, 3, 20)
print(nClass2_sel_idx)
print(nClass3_sel_idx)
print(nClass4_sel_idx)

Xfolder = datasetFolder.joinpath('X')
yfolder = datasetFolder.joinpath('y')


# === === === ===
# Carrying out feature selection for each dataset
# 4 iterations x 3 classes
elapsed_times = pd.DataFrame(
    data=np.zeros((4*3, 3)),
    columns=["ProD", "iteration", "nClass"]
)

scores_df = pd.DataFrame(
    data=np.zeros((4*3*4060, 4)),
    columns=["feature", "ProD", "iteration", "nClass"]
)
scores_df["feature"] = np.tile(np.arange(0, 4060, 1), 12)

rank_df = pd.DataFrame(
    data=np.zeros((4*3*120, 4)),
    columns=["rank", "ProD", "iteration", "nClass"]
)
rank_df["rank"] = np.tile(np.arange(0, 120, 1), 12)

count_time = 0
count = 0
count_r = 0

for sel_idxs in [nClass2_sel_idx, nClass3_sel_idx, nClass4_sel_idx]:
    for d_itr, d_idx in enumerate(sel_idxs):
        X = pd.read_csv(Xfolder.joinpath(f"{d_idx+1}_X.csv"), sep='\s+', header=None)
        X = X.values
        y = pd.read_csv(yfolder.joinpath(f"{d_idx+1}_y.csv"), header=None)
        y = y.to_numpy().reshape(-1)

        nClass = len(list(set(y)))
        print(f"nClass: {nClass} ... | iteration: {d_itr}")

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
        rank_df.loc[count_r:count_r+119, "ProD"] = pdeSegregate.top_features_[
            :nRetainedFeatures
        ]
        rank_df.loc[count_r:count_r+119, "iteration"] = np.repeat([d_itr], 120)
        rank_df.loc[count_r:count_r+119, "nClass"] = np.repeat([nClass], 120)
        count_r += 120

        scores_df.loc[count:count+4059, "ProD"] = pdeSegregate.feature_importances_
        scores_df.loc[count:count+4059, "iteration"] = np.repeat([d_itr], 4060)
        scores_df.loc[count:count+4059, "nClass"] = np.repeat([nClass], 4060)
        count += 4060

        # === === === === === === ===
        # GET ELAPSED TIME
        elapsed_times.at[count_time, "ProD"] = tProD
        elapsed_times.at[count_time, "iteration"] = d_itr
        elapsed_times.at[count_time, "nClass"] = nClass
        count_time += 1

with open("SMProD_ranks.pkl", "wb") as handle:
    pickle.dump(rank_df, handle)

with open("SMProD_feature_scores.pkl", "wb") as handle:
    pickle.dump(scores_df, handle)

elapsed_times.to_csv("SMProD_elapsed_times.csv", sep=',')

sys.exit(0)
