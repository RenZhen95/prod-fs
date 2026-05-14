import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 3:
    print(
        "Possible usage: python3.11 evaluate_fss.py <resultsFolder> <datasetName>"
    )
    print(
        "<resultsFolder> should be the folder with the combined ranks from all the " +
        "different FS methods."
    )
    sys.exit(1)
else:
    resultsFolder = Path(sys.argv[1])
    datasetName = sys.argv[2]

# Reading the feature scores
feature_scores_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_featurescores.csv"), index_col=0
)

# Reading the top 10 features
ranks_df = pd.read_csv(
    resultsFolder.joinpath(f"{datasetName}_ranks.csv"), index_col=0
)

# Features Summary
features = feature_scores_df["feature"].to_numpy()
nFeatures = len(list(set(features)))

# ANDOR
# - Relevant   : 0, 1, 2, 3
# - Redundant  : 4, 5, 6, 7
# - Correlated : 8, 9
if datasetName[0:5] == "ANDOR":
    relevant_list   = [0, 1, 2, 3]
    redundant_list  = [4, 5, 6, 7]
    correlated_list = [8, 9]
# ADDER
# - Relevant   : 0, 1, 2
# - Redundant  : 3, 4, 5
# - Correlated : 6, 7
elif datasetName[0:5] == "ADDER":
    relevant_list   = [0, 1, 2]
    redundant_list  = [3, 4, 5]
    correlated_list = [6, 7]

# Parameters for success metric according to Canedo, 2012
# Total number of irrelevant features
It = nFeatures - len(relevant_list)
# Total number of relevant features
Rt = len(relevant_list)
# Alpha
_alpha = min(0.5, Rt/It)

print(f"Total irrelevant features : {It}")
print(f"Total relevant features   : {Rt}")

# For the Electrical datasets, 10 % of 100 would be 10
nTop = 10

def count_instances_intop10(_top10, _list=[]):
    list_FSS = list(_top10.columns)
    list_FSS.remove("rank")

    count = pd.Series(data=np.zeros(len(list_FSS)), index=list_FSS)

    for i in _top10.index:
        for fss in count.index:
            if _top10.at[i, fss] in _list:
                count.at[fss] += 1

    return count

groupeddf_relvcount = ranks_df.groupby(["iteration", "n_obs"]).apply(
    count_instances_intop10, _list=relevant_list
)

def check_ifrelevantistop(_top10, _list=[]):
    list_FSS = list(_top10.columns)
    list_FSS.remove("rank")

    _top10_nfirst = _top10.iloc[:len(_list),:]
    relv_atthetop = pd.Series(
        data=[False for i in range(len(list_FSS))], index=list_FSS
    )

    for fss in list_FSS:
        if set(_top10_nfirst[fss].values) == set(_list):
            relv_atthetop[fss] = True

    return relv_atthetop

groupeddf_relvattop = ranks_df.groupby(["iteration", "n_obs"]).apply(
    check_ifrelevantistop, _list=relevant_list
)

# Computing the success rate as defined by Canedo, 2012
fss_list = list(groupeddf_relvattop.columns)
average_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[30, 50, 70], columns=fss_list
)
sd_successrate = pd.DataFrame(
    data=np.zeros((3, len(fss_list))), index=[30, 50, 70], columns=fss_list
)

def calculate_success(_x):
    """
    Calculating the success rate as defined by Canedo, 2012
    """
    success_col = _x / Rt
    IRatio = _alpha * ((nTop - _x) / It)

    return (success_col - IRatio) * 100

def calculate_success_per_nobs(nobs):
    relvcount = groupeddf_relvcount.xs(nobs, level=1)
    relvattop = groupeddf_relvattop.xs(nobs, level=1)

    groupeddf_success = relvcount.apply(calculate_success)

    for fss in groupeddf_success.columns:
        for it_ in groupeddf_success.index:
            if relvattop.at[it_, fss]:
                groupeddf_success.at[it_, fss] = 100.0

    return groupeddf_success.mean(), groupeddf_success.std()

avrsuccess_obs30, sdsuccess_obs30 = calculate_success_per_nobs(30)
avrsuccess_obs50, sdsuccess_obs50 = calculate_success_per_nobs(50)
avrsuccess_obs70, sdsuccess_obs70 = calculate_success_per_nobs(70)

average_successrate.loc[30] = avrsuccess_obs30
average_successrate.loc[50] = avrsuccess_obs50
average_successrate.loc[70] = avrsuccess_obs70

sd_successrate.loc[30] = sdsuccess_obs30
sd_successrate.loc[50] = sdsuccess_obs50
sd_successrate.loc[70] = sdsuccess_obs70

average_successrate.to_csv(f"{datasetName}_successrates.csv")
sd_successrate.to_csv(f"{datasetName}_successrates_sd.csv")

sys.exit(0)
