import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print(
        "Possible usage: python3.11 combine_fss.py <folder>"
    )
    sys.exit(1)
else:
    folder = Path(sys.argv[1])

# === === === ===
# Feature Scores (other FS)
with open(folder.joinpath("OtherFS/SDIfeature_scores.pkl"), "rb") as handle:
    pyFSS_featurescores = pickle.load(handle)

# Reading weights from IRelief
IRlf_folder = folder.joinpath("IRelief")
IRlf_nClass2 = pd.read_csv(
    IRlf_folder.joinpath("WeightsI_nClass2.csv"), header=None
)
IRlf_nClass3 = pd.read_csv(
    IRlf_folder.joinpath("WeightsI_nClass3.csv"), header=None
)
IRlf_nClass4 = pd.read_csv(
    IRlf_folder.joinpath("WeightsI_nClass4.csv"), header=None
)
IRlf = pd.Series(
    data=np.zeros(
        IRlf_nClass2.shape[0]*IRlf_nClass2.shape[1]*3 # 3 different nClass
    ), name="IRlf"
)
i = 0
for IRlf_featurescores in [
        IRlf_nClass2, IRlf_nClass3, IRlf_nClass4
]:
    for iteration in IRlf_featurescores.columns:
        for feature in IRlf_featurescores.index:
            IRlf[i] = IRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["IRlf"] = IRlf

# Reading weights from LHRelief
LHRlf_folder = folder.joinpath("LHRelief")
LHRlf_nClass2 = pd.read_csv(
    LHRlf_folder.joinpath("WeightsLM_nClass2.csv"), header=None
)
LHRlf_nClass3 = pd.read_csv(
    LHRlf_folder.joinpath("WeightsLM_nClass3.csv"), header=None
)
LHRlf_nClass4 = pd.read_csv(
    LHRlf_folder.joinpath("WeightsLM_nClass4.csv"), header=None
)
LHRlf = pd.Series(
    data=np.zeros(
        LHRlf_nClass2.shape[0]*LHRlf_nClass2.shape[1]*3 # 3 different nClass
    ), name="LHRlf"
)
i = 0
for LHRlf_featurescores in [
        LHRlf_nClass2, LHRlf_nClass3, LHRlf_nClass4
]:
    for iteration in LHRlf_featurescores.columns:
        for feature in LHRlf_featurescores.index:
            LHRlf[i] = LHRlf_featurescores.at[feature, iteration]
            i += 1
pyFSS_featurescores["LHRlf"] = LHRlf

# Reading scores from PDE-S
with open(folder.joinpath(f"PDE-S/SDIPDE-S_feature_scores.pkl"), "rb") as handle:
    pdes_featurescores = pickle.load(handle)

pyFSS_featurescores["PDE-S"] = pdes_featurescores["PDE-S"]


# === === === ===
# Ranks
with open(folder.joinpath("OtherFS/SDIranks.pkl"), "rb") as handle:
    pyFSS_ranks = pickle.load(handle)

# Reading ranks from mRMR (Ding, 2005)
# Minus 1 because MATLAB indexing starts from 1 and not 0
# From MATLAB documentation:
#   If idx(3) is 5 :: The third most important featurey is the 10th column
mRMR_folder = folder.joinpath("mRMR")
mRMR_nClass2 = pd.read_csv(
    mRMR_folder.joinpath("mRMR_nClass2.csv"), header=None
)
mRMR_nClass2 = mRMR_nClass2 - 1
mRMR_nClass3 = pd.read_csv(
    mRMR_folder.joinpath("mRMR_nClass3.csv"), header=None
)
mRMR_nClass3 = mRMR_nClass3 - 1
mRMR_nClass4 = pd.read_csv(
    mRMR_folder.joinpath("mRMR_nClass4.csv"), header=None
)
mRMR_nClass4 = mRMR_nClass4 - 1

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
# Top 120 Features of IRelief
IRlf_nClass2 = IRlf_nClass2.apply(get_rankings)
IRlf_nClass3 = IRlf_nClass3.apply(get_rankings)
IRlf_nClass4 = IRlf_nClass4.apply(get_rankings)

# Top 120 features x 4 iterations x 3 nClasses
IRlf_ranks = pd.Series(data=np.zeros(120*4*3), name="IRlf")
i = 0
for IRlf_ranks_df in [IRlf_nClass2, IRlf_nClass3, IRlf_nClass4]:
    for itr in range(4):
        for rank in range(120):
            top_feature = IRlf_ranks_df.loc[:,itr][
                IRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            IRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["IRlf"] = IRlf_ranks

# === === === ===
# Top 120 Features of LHRelief
LHRlf_nClass2 = LHRlf_nClass2.apply(get_rankings)
LHRlf_nClass3 = LHRlf_nClass3.apply(get_rankings)
LHRlf_nClass4 = LHRlf_nClass4.apply(get_rankings)

# Top 120 features x 4 iterations x 3 nClasses
LHRlf_ranks = pd.Series(data=np.zeros(120*4*3), name="LHRlf")
i = 0
for LHRlf_ranks_df in [LHRlf_nClass2, LHRlf_nClass3, LHRlf_nClass4]:
    for itr in range(4):
        for rank in range(120):
            top_feature = LHRlf_ranks_df.loc[:,itr][
                LHRlf_ranks_df.loc[:,itr] == rank
            ].index[0]

            LHRlf_ranks[i] = top_feature
            i += 1
pyFSS_ranks["LHRlf"] = LHRlf_ranks

# === === === ===
# Top 120 Features of mRMR

# Here as compared to the ranks df from IRlf and LHRlf,
#  - Index = Rank | Value = Feature

mRMR_nClass2_120 = mRMR_nClass2.iloc[:120,:]
mRMR_nClass3_120 = mRMR_nClass3.iloc[:120,:]
mRMR_nClass4_120 = mRMR_nClass4.iloc[:120,:]

# Top 120 features x 4 iterations x 3 nClasses
mRMR_ranks = pd.Series(data=np.zeros(120*4*3), name="mRMR")
i = 0
for mRMR_ranks_df in [mRMR_nClass2_120, mRMR_nClass3_120, mRMR_nClass4_120]:
    for itr in range(4):
        for rank in range(120):
            mRMR_ranks[i] = mRMR_ranks_df.loc[rank, itr]
            i += 1
pyFSS_ranks["mRMR"] = mRMR_ranks


# Reading ranks from PDE-S
with open(folder.joinpath(f"PDE-S/SDIPDE-S_ranks.pkl"), "rb") as handle:
    pdes_ranks = pickle.load(handle)

pyFSS_ranks["PDE-S"] = pdes_ranks["PDE-S"]


# === === === ===
# Elapsed Times
py_times = pd.read_csv(
    folder.joinpath("OtherFS/SDIelapsed_times.csv"), index_col=0
)
IRlf_times = pd.read_csv(
    IRlf_folder.joinpath("tI.csv"), header=None
)
LHRlf_times = pd.read_csv(
    LHRlf_folder.joinpath("tLM.csv"), header=None
)
mRMR_times = pd.read_csv(
    mRMR_folder.joinpath("tmRMR.csv"), header=None
)
pdes_times = pd.read_csv(
    folder.joinpath(f"PDE-S/SDIPDE-S_elapsed_times.csv"), index_col=0
)
py_times["IRlf"]  = IRlf_times.stack().values
py_times["LHRlf"] = LHRlf_times.stack().values
py_times["mRMR"] = mRMR_times.stack().values
py_times["PDE-S"] = pdes_times["PDE-S"]

pyFSS_featurescores.to_csv(folder.joinpath("Combined/SDIfeaturescores.csv"))
pyFSS_ranks.to_csv(folder.joinpath("Combined/SDIranks.csv"))
py_times.to_csv(folder.joinpath("Combined/SDIelapsedtimes.csv"))

sys.exit(0)
