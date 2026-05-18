import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from skrebate import ReliefF, MultiSURF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

# Taking the top nRetainedFeatures
def get_indsTopnFeatures(featImportances, nCap):
    res_tmp = -featImportances
    res_tmpSorted = np.sort(res_tmp)

    inds_topFeatures = []
    counter = 1
    for i in res_tmpSorted:
        inds_topFeatures.append(np.where(res_tmp==i)[0][0])

        counter += 1
        if counter==nCap+1:
            break

    return inds_topFeatures

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 featureSelection.py <processedDatasets> " +
        "<fsResults_matlab> <savefolder>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    fsResults_matlab_folder = Path(sys.argv[2])
    savefolder = Path(sys.argv[3])
    nRetainedFeatures = 100

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)


# === === === ===
# Reading the feature rankings, output from the feature selection methods ran on MATLAB
ReliefI_dict = defaultdict()
ReliefLM_dict = defaultdict()
mRMR_dict = defaultdict()
for f in os.scandir(fsResults_matlab_folder):
    ds_name = f.name.split('_')[0]
    if "ReliefI" in f.name:
        tmpDF_RI = pd.read_csv(f, header=None).values
        ReliefI_dict[ds_name] = tmpDF_RI.reshape((tmpDF_RI.shape[0],))

    elif "ReliefLM" in f.name:
        # Taking only NN of 7 due to computational load
        # (suggested by author as good rule of thumb)
        tmpDF_RLM = pd.read_csv(f, header=None).values
        ReliefLM_dict[ds_name] = tmpDF_RLM.reshape((tmpDF_RLM.shape[0],))

    # Reading ranks from mRMR (Ding, 2005)
    # Minus 1 because MATLAB indexing starts from 1 and not 0
    # From MATLAB documentation:
    #   If idx(3) is 5 :: The third most important feature is the 10th column
    elif "_mRMR" in f.name:
        tmpDF_mRMR = pd.read_csv(f, header=None).values
        mRMR_dict[ds_name] = tmpDF_mRMR.reshape(tmpDF_mRMR.shape[1])
        mRMR_dict[ds_name] = mRMR_dict[ds_name] - 1


# === === === ===
# Carrying out feature selection for each dataset
dataset_inds_topFeatures = defaultdict()

elapsed_times_perDS = defaultdict()

for dataset in datasets_dict.keys():
    elapsed_times = defaultdict()

    print(f"Dealing with {dataset} ... ")
    X = datasets_dict[dataset]['X']
    y = datasets_dict[dataset]['y']
    y_mapper = datasets_dict[dataset]['y_mapper']


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
    MSurf.fit(X,y)
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
        n_estimators=1000, criterion="gini", random_state=0,
        n_jobs=-1
    )
    rfGini.fit(X,y)
    rfGini_featureImportance = rfGini.feature_importances_
    tRF_stop = process_time()
    tRF = tRF_stop - tRF_start

    # === === === === === === ===
    # GETTING TOP N FEATURES
    inds_topFeatures_RlfF = get_indsTopnFeatures(
        RlfF.feature_importances_, nRetainedFeatures
    )
    inds_topFeatures_MSurf = get_indsTopnFeatures(
        MSurf.feature_importances_, nRetainedFeatures
    )
    inds_topFeatures_RlfI = get_indsTopnFeatures(
        ReliefI_dict[dataset], nRetainedFeatures
    )
    inds_topFeatures_RlfLM = get_indsTopnFeatures(
        ReliefLM_dict[dataset], nRetainedFeatures
    )
    inds_topFeatures_RFGini = get_indsTopnFeatures(
        rfGini_featureImportance, nRetainedFeatures
    )
    inds_topFeatures_MI = get_indsTopnFeatures(
        resMI, nRetainedFeatures
    )
    inds_topFeatures_mRMR = mRMR_dict[dataset][:nRetainedFeatures]
    inds_topFeatures_FT = get_indsTopnFeatures(
        resFT_stat, nRetainedFeatures
    )

    inds_topFeatures = {
        "RlfF": inds_topFeatures_RlfF,
        "MSurf": inds_topFeatures_MSurf,
        "RlfI": inds_topFeatures_RlfI,
        "RlfLM": inds_topFeatures_RlfLM,
        "RFGini": inds_topFeatures_RFGini,
        "MI": inds_topFeatures_MI,
        "mRMR": inds_topFeatures_mRMR,
        "FT": inds_topFeatures_FT
    }
    dataset_inds_topFeatures[dataset] = inds_topFeatures

    # === === === === === === ===
    # GET ELAPSED TIME
    elapsed_times = {
        "RlfF": tRlfF,
        "MSurf": tMSurf,
        "MI": tMI,
        "RFGini": tRF,
        "FT": tFT
    }
    elapsed_times_perDS[dataset] = elapsed_times

with open(savefolder.joinpath(f"top{nRetainedFeatures}Features.pkl"), "wb") as handle:
    pickle.dump(dataset_inds_topFeatures, handle)
with open(savefolder.joinpath(f"fsElapsedTimes.pkl"), "wb") as handle:
    pickle.dump(elapsed_times_perDS, handle)

sys.exit(0)
