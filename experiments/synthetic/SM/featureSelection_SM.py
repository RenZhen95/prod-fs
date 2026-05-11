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

if len(sys.argv) < 2:
    print(
        "Possible usage: python3 featureSelection_SDI.py <datasetFolder>"
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
    data=np.zeros((4*3, 7)),
    columns=[
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "nClass"
    ]
)

scores_df = pd.DataFrame(
    data=np.zeros((4*3*4060, 8)),
    columns=[
        "feature",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "nClass"
    ]
)
scores_df["feature"] = np.tile(np.arange(0, 4060, 1), 12)

rank_df = pd.DataFrame(
    data=np.zeros((4*3*120, 8)),
    columns=[
        "rank",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT",
        "iteration",
        "nClass"
    ]
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
        resMI = mutual_info_classif(X, y, n_neighbors=7, random_state=0)
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
        rank_df.loc[count_r:count_r+119, "RlfF"] = get_indsTopnFeatures(
            RlfF.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+119, "MSurf"] = get_indsTopnFeatures(
            MSurf.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+119, "MI"] = get_indsTopnFeatures(
            resMI, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+119, "RFGini"] = get_indsTopnFeatures(
            rfGini.feature_importances_, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+119, "FT"] = get_indsTopnFeatures(
            resFT_stat, nRetainedFeatures
        )
        rank_df.loc[count_r:count_r+119, "iteration"] = np.repeat([d_itr], 120)
        rank_df.loc[count_r:count_r+119, "nClass"] = np.repeat([nClass], 120)
        count_r += 120

        scores_df.loc[count:count+4059, "RlfF"] = RlfF.feature_importances_
        scores_df.loc[count:count+4059, "MSurf"] = MSurf.feature_importances_
        scores_df.loc[count:count+4059, "MI"] = resMI
        scores_df.loc[count:count+4059, "RFGini"] = rfGini.feature_importances_
        scores_df.loc[count:count+4059, "FT"] = resFT_stat

        scores_df.loc[count:count+4059, "iteration"] = np.repeat([d_itr], 4060)
        scores_df.loc[count:count+4059, "nClass"] = np.repeat([nClass], 4060)
        count += 4060

        # === === === === === === ===
        # GET ELAPSED TIME
        elapsed_times.at[count_time, "RlfF"] = tRlfF
        elapsed_times.at[count_time, "MSurf"] = tMSurf
        elapsed_times.at[count_time, "RFGini"] = tRF
        elapsed_times.at[count_time, "MI"] = tMI
        elapsed_times.at[count_time, "FT"] = tFT
        elapsed_times.at[count_time, "iteration"] = d_itr
        elapsed_times.at[count_time, "nClass"] = nClass
        count_time += 1

with open("SDIranks.pkl", "wb") as handle:
    pickle.dump(rank_df, handle)

with open("SDIfeature_scores.pkl", "wb") as handle:
    pickle.dump(scores_df, handle)

elapsed_times.to_csv("SDIelapsed_times.csv", sep=',')

sys.exit(0)
