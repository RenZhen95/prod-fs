import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 combine_fss.py <folder> <dataset>"
    )
    sys.exit(1)
else:        
    folder = Path(sys.argv[1])
    dataset = sys.argv[2]

# === === === ===
# Feature Scores (other FS)
with open(folder.joinpath(f"OtherFS/{dataset}feature_scores.pkl"), "rb") as handle:
    pyFSS_featurescores = pickle.load(handle)

# Reading weights from IRelief
IRlf_folder = folder.joinpath("IRelief")
IRlf_featurescores_30 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_30.csv"), header=None
)
IRlf_featurescores_50 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_50.csv"), header=None
)
IRlf_featurescores_70 = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}WeightsI_70.csv"), header=None
)
IRlf = pd.Series(
    data=np.zeros(
        IRlf_featurescores_30.shape[0]*IRlf_featurescores_30.shape[1]*3
    ), name="IRlf"
)
i = 0
for IRlf_featurescores in [
        IRlf_featurescores_30, IRlf_featurescores_50, IRlf_featurescores_70
]:
    for iteration in IRlf_featurescores.columns:
        for feature in IRlf_featurescores.index:
            IRlf[i] = IRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["IRlf"] = IRlf

# Reading weights from LHRelief
LHRlf_folder = folder.joinpath("LHRelief")
LHRlf_featurescores_30 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_30.csv"), header=None
)
LHRlf_featurescores_50 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_50.csv"), header=None
)
LHRlf_featurescores_70 = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}WeightsLM_70.csv"), header=None
)
LHRlf = pd.Series(
    data=np.zeros(
        LHRlf_featurescores_30.shape[0]*LHRlf_featurescores_30.shape[1]*3
    ), name="LHRlf"
)
i = 0
for LHRlf_featurescores in [
        LHRlf_featurescores_30, LHRlf_featurescores_50, LHRlf_featurescores_70
]:
    for iteration in LHRlf_featurescores.columns:
        for feature in LHRlf_featurescores.index:
            LHRlf[i] = LHRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["LHRlf"] = LHRlf

# Reading scores from ProD
with open(folder.joinpath(f"ProD/{dataset}ProD_feature_scores.pkl"), "rb") as handle:
    pdes_featurescores = pickle.load(handle)

pyFSS_featurescores["ProD"] = pdes_featurescores["ProD"]


# === === === ===
# Ranks
with open(folder.joinpath(f"OtherFS/{dataset}ranks.pkl"), "rb") as handle:
    pyFSS_ranks = pickle.load(handle)

# Reading ranks from mRMR (Ding, 2005)
# Minus 1 because MATLAB indexing starts from 1 and not 0
# From MATLAB documentation:
#   If idx(3) is 5 :: The third most important featurey is the 10th column
mRMR_folder = folder.joinpath("mRMR")
mRMR_30 = pd.read_csv(
    mRMR_folder.joinpath(f"{dataset}mRMR_nObs30.csv"), header=None
)
mRMR_30 = mRMR_30 - 1
mRMR_50 = pd.read_csv(
    mRMR_folder.joinpath(f"{dataset}mRMR_nObs50.csv"), header=None
)
mRMR_50 = mRMR_50 - 1
mRMR_70 = pd.read_csv(
    mRMR_folder.joinpath(f"{dataset}mRMR_nObs70.csv"), header=None
)
mRMR_70 = mRMR_70 - 1

def get_rankings(_importances):
    n_features = len(_importances)

    # Sorted features from most important to least
    sortedfeatures = sorted(
        range(n_features), key=lambda i: _importances[i], reverse=True
    )

    # Rank of each feature
    feature_ranks = np.zeros(n_features)
    for r, f in enumerate(sortedfeatures):
        feature_ranks[f] = r # Index = Feature | Value = Rank

    return feature_ranks

# === === === ===
# Top 10 Features of IRelief
IRlf_ranks_30 = IRlf_featurescores_30.apply(get_rankings)
IRlf_ranks_50 = IRlf_featurescores_50.apply(get_rankings)
IRlf_ranks_70 = IRlf_featurescores_70.apply(get_rankings)

# Top 10 features x 50 iterations x 3 number of observations
IRlf_ranks = pd.Series(data=np.zeros(10*50*3), name="IRlf")
i = 0
for IRlf_ranks_df in [IRlf_ranks_30, IRlf_ranks_50, IRlf_ranks_70]:
    for itr in range(50):
        for rank in range(10):
            top_feature = IRlf_ranks_df.loc[:,itr][
                IRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            IRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["IRlf"] = IRlf_ranks

# === === === ===
# Top 10 Features of LHRelief
LHRlf_ranks_30 = LHRlf_featurescores_30.apply(get_rankings)
LHRlf_ranks_50 = LHRlf_featurescores_50.apply(get_rankings)
LHRlf_ranks_70 = LHRlf_featurescores_70.apply(get_rankings)

# Top 10 features x 50 iterations x 3 number of observations
LHRlf_ranks = pd.Series(data=np.zeros(10*50*3), name="LHRlf")
i = 0
for LHRlf_ranks_df in [LHRlf_ranks_30, LHRlf_ranks_50, LHRlf_ranks_70]:
    for itr in range(50):
        for rank in range(10):
            top_feature = LHRlf_ranks_df.loc[:,itr][
                LHRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            LHRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["LHRlf"] = LHRlf_ranks

# === === === ===
# Top 10 Features of mRMR

# Here as compared to the ranks df from IRlf and LHRlf,
#  - Index = Rank | Value = Feature

mRMR_30_10 = mRMR_30.iloc[:10,:]
mRMR_50_10 = mRMR_50.iloc[:10,:]
mRMR_70_10 = mRMR_70.iloc[:10,:]

# Top 10 features x 50 iterations x 3 nObs
mRMR_ranks = pd.Series(data=np.zeros(10*50*3), name="mRMR")
i = 0
for mRMR_ranks_df in [mRMR_30_10, mRMR_50_10, mRMR_70_10]:
    for itr in range(50):
        for rank in range(10):
            mRMR_ranks[i] = mRMR_ranks_df.loc[rank, itr]
            i += 1
pyFSS_ranks["mRMR"] = mRMR_ranks

# Reading ranks from ProD
with open(folder.joinpath(f"ProD/{dataset}ProD_ranks.pkl"), "rb") as handle:
    pdes_ranks = pickle.load(handle)

pyFSS_ranks["ProD"] = pdes_ranks["ProD"]


# === === === ===
# Elapsed Times
py_times = pd.read_csv(
    folder.joinpath(f"OtherFS/{dataset}elapsed_times.csv"), index_col=0
)
IRlf_times = pd.read_csv(
    IRlf_folder.joinpath(f"{dataset}_tI.csv"), header=None
)
LHRlf_times = pd.read_csv(
    LHRlf_folder.joinpath(f"{dataset}_tLM.csv"), header=None
)
mRMR_times = pd.read_csv(
    mRMR_folder.joinpath(f"{dataset}tmRMR.csv"), header=None
)
pdes_times = pd.read_csv(
    folder.joinpath(f"ProD/{dataset}ProD_elapsed_times.csv"), index_col=0
)
py_times["IRlf"]  = IRlf_times.stack().values
py_times["LHRlf"] = LHRlf_times.stack().values
py_times["mRMR"] = mRMR_times.stack().values
py_times["ProD"] = pdes_times["ProD"]

pyFSS_featurescores.to_csv(
    folder.joinpath(f"Combined/{dataset}_featurescores.csv")
)
pyFSS_ranks.to_csv(
    folder.joinpath(f"Combined/{dataset}_ranks.csv")
)
py_times.to_csv(
    folder.joinpath(f"Combined/{dataset}_elapsedtimes.csv")
)

sys.exit(0)
