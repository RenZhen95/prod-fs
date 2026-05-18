import pickle
import os, sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from collections import defaultdict
from scipy.stats import zscore, alexandergovern

def updateLabel(Y):
    uy = list(set(Y))
    miny = min(uy)
    mapper = {}
    if miny==-1:
        newY = (Y+1)/2+1
    else:
        if not isinstance(Y[0], str):
            newY = Y-miny+1
        else:
            for i, e in enumerate(uy):
                mapper[e] = i

            newY = [mapper[e] for e in Y]

    return newY, mapper

def preprocessing_features(_X, _y, sig_level):
    y_uniqueIDS = list(set(_y))

    inds_y_uniqueList = []
    for y_unique in y_uniqueIDS:
        inds_y_uniqueList.append(list(np.where(_y==y_unique)[0]))

    inds_sig_features = []
    for feat_idx in range(_X.shape[1]):
        groups = []
        for inds in inds_y_uniqueList:
            groups.append(_X[inds, feat_idx])

        # Conduct the Alexander-Govern test
        # - Similar to one way ANOVA but relaxes the condition for homogenous variances
        res = alexandergovern(*groups)

        if res.pvalue < sig_level:
            inds_sig_features.append(feat_idx)

    return _X[:, inds_sig_features]

if len(sys.argv) < 3:
    print("Possible usage: python3 prep_datasets.py <datasetsFolder> <sigLevel>")
    sys.exit(1)
else:
    datasetsFolder = Path(sys.argv[1])
    sigLevel = float(sys.argv[2])

processedDatasets_dict = defaultdict()
processedDatasets_noANOVAcut_dict = defaultdict()

for f in os.scandir(datasetsFolder):
    datasetName = f.name.split('.')[0]
    datasetDetected = False

    # Loading .mat files
    if f.name[-3:] == "mat":
        if not "PersonGaitDataSet" in f.name:
            data = loadmat(f)["data"]
            X = data[:,1:]
            y = data[:,0]
        else:
            data = loadmat(f)
            X = data['X']
            y = data['Y'].reshape(data['Y'].shape[0],)

        datasetDetected = True

    # Gene expression dataset
    elif f.name[-3:] == "csv":
        dataset = pd.read_csv(f, index_col=0)
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].to_numpy()
        datasetDetected = True

    if datasetDetected:
        print(f"Processing {datasetName} ... ")

        # Normalizing the training data
        X_st = zscore(X, axis=0, ddof=1, nan_policy='propagate')

        # Standardizing the y labels
        # Examples: {-1, 1} > {1, 2}
        y_st, y_mapper = updateLabel(y)

        # Preprocessing step:
        # 1. Removing features with p-values < sigLevel (Feature - Label)
        if isinstance(y_st, list):
            y_st = np.array(y_st)
        X_streduced = preprocessing_features(X_st, y_st, sigLevel)

        # 2. Remove features with nan
        nanCheck = np.isnan(X_st)
        if np.any(nanCheck):
            nan_y = np.where(nanCheck)
            nan_y = list(set(nan_y[1]))
            X_st = np.delete(X_st, nan_y, 1)

        print(X_st)
        print(X_st.shape)
        dataset_dict_noANOVAcut = {'X': X_st, 'y': y_st, 'y_mapper': y_mapper}
        processedDatasets_noANOVAcut_dict[datasetName] = dataset_dict_noANOVAcut
        
        print(X_streduced)
        print(X_streduced.shape)
        dataset_dict = {'X': X_streduced, 'y': y_st, 'y_mapper': y_mapper}
        processedDatasets_dict[datasetName] = dataset_dict

        print(y_st)
        print(y_mapper)

with open(f"processedDatasets_{sigLevel}.pkl", "wb") as handle:
    pickle.dump(processedDatasets_dict, handle)

with open(f"processedDatasets_noANOVAcut.pkl", "wb") as handle:
    pickle.dump(processedDatasets_noANOVAcut_dict, handle)

sys.exit(0)
