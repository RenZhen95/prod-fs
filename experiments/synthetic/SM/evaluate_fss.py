import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

"""
For the Electrical datasets, 3 % of 4000 would be roughly 120
"""

if len(sys.argv) < 4:
    print(
        "Possible usage: python3.11 evaluate_fss.py <resultsFolder> " +
        "<trueSignatures> <nTop>"
    )
    print(
        "<resultsFolder> should be the folder with the combined ranks from all the " +
        "different FS methods."
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    trueSignatures_folder = Path(sys.argv[2])
    nTop = int(sys.argv[3])

# Reading the top features
ranks_df = pd.read_csv(
    resultsFolder.joinpath("SDIranks.csv"), index_col=0
)

# Narrow threshold if desired
only_retain = np.arange(0, nTop, 1)
ranks_df = ranks_df[ranks_df["rank"].isin(only_retain)]

# Getting true signatures
nClass2_idxs = [16, 43, 70, 97]
nClass3_idxs = [17, 44, 71, 98]
nClass4_idxs = [18, 45, 72, 99]

# IMPORTANT NOTE: R indexing starts from 1 just like MATLAB !!
def get_trueSignatures(_idxs):
    _trueSignatures_dict = defaultdict()
    for _i, _it in enumerate(_idxs):
        _true_signatures = pd.read_csv(
            trueSignatures_folder.joinpath(f"{_it}_trueSignatures.csv"), header=None
        )
        _trueSignatures_dict[_i] = _true_signatures - 1
    return _trueSignatures_dict

nClass2_trueSignatures = get_trueSignatures(nClass2_idxs)
nClass3_trueSignatures = get_trueSignatures(nClass3_idxs)
nClass4_trueSignatures = get_trueSignatures(nClass4_idxs)
trueSignatures = {
    2.0: nClass2_trueSignatures,
    3.0: nClass3_trueSignatures,
    4.0: nClass4_trueSignatures,
}

# Features Summary
# Reading the feature scores
feature_scores_df = pd.read_csv(
    resultsFolder.joinpath("SDIfeaturescores.csv"), index_col=0
)
features = feature_scores_df["feature"].to_numpy()
nFeatures = len(list(set(features)))

# Parameters for success metric according to Canedo, 2012
# Total number of irrelevant features
It = 4060 - 3
# Total number of relevant features (one per dimension)
Rt = 3
# Alpha
_alpha = min(0.5, Rt/It)

print(f"Total irrelevant features : {It}")
print(f"Total relevant features   : {Rt}")

def count_instances_attop(_top):
    list_FSS = list(_top.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("nClass")

    nClass = list(set(_top["nClass"].to_numpy()))[0]
    itr = list(set(_top["iteration"].to_numpy()))[0]

    # Idea: Only one feature should be selected from each of the three
    # dimensions, meaning there should only be three relevant features

    # As soon as one feature is selected from a dimension, the rest are
    # redundant, and will be considered "irrelevant"
    _truegenes_dim1 = trueSignatures[nClass][itr].loc[0].values
    _truegenes_dim2 = trueSignatures[nClass][itr].loc[1].values
    _truegenes_dim3 = trueSignatures[nClass][itr].loc[2].values

    _flag_dim1 = pd.Series(
        data=[False for _ in range(len(list_FSS))], index=list_FSS
    )
    _flag_dim2 = pd.Series(
        data=[False for _ in range(len(list_FSS))], index=list_FSS
    )
    _flag_dim3 = pd.Series(
        data=[False for _ in range(len(list_FSS))], index=list_FSS
    )

    count = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)

    for i in _top.index:
        for fss in count.index:
            # Checking if a top feature is in the first dimension
            if (_top.at[i, fss] in _truegenes_dim1) and (not _flag_dim1[fss]):
                count.at[fss] += 1
                _flag_dim1[fss] = True

            # Checking if a top feature is in the second dimension
            elif (_top.at[i, fss] in _truegenes_dim2) and (not _flag_dim2[fss]):
                count.at[fss] += 1
                _flag_dim2[fss] = True

            # Checking if a top feature is in the third dimension
            elif (_top.at[i, fss] in _truegenes_dim3) and (not _flag_dim3[fss]):
                count.at[fss] += 1
                _flag_dim3[fss] = True

    return count

groupeddf_relvcount = ranks_df.groupby(["iteration", "nClass"]).apply(
    count_instances_attop
)

def check_ifrelevantistop(_top):
    list_FSS = list(_top.columns)
    list_FSS.remove("rank")
    list_FSS.remove("iteration")
    list_FSS.remove("nClass")

    nClass = list(set(_top["nClass"].to_numpy()))[0]
    itr = list(set(_top["iteration"].to_numpy()))[0]

    _truegenes_dim1 = trueSignatures[nClass][itr].loc[0].values
    _truegenes_dim2 = trueSignatures[nClass][itr].loc[1].values
    _truegenes_dim3 = trueSignatures[nClass][itr].loc[2].values

    _flag_dim1 = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)
    _flag_dim2 = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)
    _flag_dim3 = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)

    # 1 relevant gene per dimension of 3 dimensions
    _top_3first = _top.iloc[:3,:]

    relv_atthetop = pd.Series(
        data=[False for i in range(len(list_FSS))], index=list_FSS
    )

    for fss in list_FSS:
        nF_dim1 = len([f for f in _top_3first[fss] if f in _truegenes_dim1])
        nF_dim2 = len([f for f in _top_3first[fss] if f in _truegenes_dim2])
        nF_dim3 = len([f for f in _top_3first[fss] if f in _truegenes_dim3])

        if (nF_dim1 == 1) and (nF_dim2 == 1) and (nF_dim3 == 1):
            relv_atthetop[fss] = True

    return relv_atthetop

groupeddf_relvattop = ranks_df.groupby(["iteration", "nClass"]).apply(
    check_ifrelevantistop
)

# Computing the success rate as defined by Canedo, 2012
fss_list = list(groupeddf_relvattop.columns)
average_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[2.0, 3.0, 4.0], columns=fss_list
)
sd_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[2.0, 3.0, 4.0], columns=fss_list
)

def calculate_success(_x):
    """
    Calculating the success rate as defined by Canedo, 2012
    """
    success_col = _x / Rt
    IRatio = _alpha * ((nTop - _x) / It)

    return (success_col - IRatio) * 100

def calculate_success_per_nClass(nClass):
    relvcount = groupeddf_relvcount.xs(nClass, level=1)
    relvattop = groupeddf_relvattop.xs(nClass, level=1)

    groupeddf_success = relvcount.apply(calculate_success)

    for fss in groupeddf_success.columns:
        for it_ in groupeddf_success.index:
            if relvattop.at[it_, fss]:
                groupeddf_success.at[it_, fss] = 100.0

    return groupeddf_success.mean(), groupeddf_success.std()

avrsuccess_nClass2, sdsuccess_nClass2 = calculate_success_per_nClass(2.0)
avrsuccess_nClass3, sdsuccess_nClass3 = calculate_success_per_nClass(3.0)
avrsuccess_nClass4, sdsuccess_nClass4 = calculate_success_per_nClass(4.0)

average_successrate.loc[2.0] = avrsuccess_nClass2
average_successrate.loc[3.0] = avrsuccess_nClass3
average_successrate.loc[4.0] = avrsuccess_nClass4

sd_successrate.loc[2.0] = sdsuccess_nClass2
sd_successrate.loc[3.0] = sdsuccess_nClass3
sd_successrate.loc[4.0] = sdsuccess_nClass4

average_successrate.to_csv(f"{nTop}_SDIsuccessrates.csv")
sd_successrate.to_csv(f"{nTop}_SDIsuccessrates_sd.csv")

sys.exit(0)
