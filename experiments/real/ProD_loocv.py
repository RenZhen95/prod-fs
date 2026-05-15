import pickle
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 4:
    print(
        "Possible usage: python3 PDE-S_loocv.py <processedDatasets> " +
        "<selFeatures> <saveFolder>"
    )
    sys.exit(1)
else:
    processedDatasets_pkl = Path(sys.argv[1])
    selFeatures_pkl = Path(sys.argv[2])
    saveFolder = Path(sys.argv[3])

with open(processedDatasets_pkl, "rb") as handle:
    processedDatasets_dict = pickle.load(handle)

with open(selFeatures_pkl, "rb") as handle:
    topFeatures = pickle.load(handle)

# Experiment with the following number of retained features
nRetainedFeatures = [25, 50, 75, 100]

for ds in topFeatures.keys():
    print(f"Dataset: {ds}")
    print(f" - Feature selection scheme: PDE-S")

    ds_results = pd.DataFrame(columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"])

    for n_ in nRetainedFeatures:
        print(f" - Number of retained features: {n_}")
        X = processedDatasets_dict[ds]['X']
        y = processedDatasets_dict[ds]['y']
        inds_topFeatures = topFeatures[ds][:n_]

        X_reduced = X[:,inds_topFeatures]

        ypredArray_kNN = np.zeros((X.shape[0],))
        ypredArray_SVM = np.zeros((X.shape[0],))
        ypredArray_NBy = np.zeros((X.shape[0],))
        ypredArray_LDA = np.zeros((X.shape[0],))
        ypredArray_DT  = np.zeros((X.shape[0],))

        # LOOCV
        for n_val in tqdm(range(X_reduced.shape[0])):
            X_validate = X_reduced[n_val,:]
            X_validate = X_validate.reshape((1, X_validate.shape[0]))

            train_inds = [n for n in range(X_reduced.shape[0]) if n != n_val]
            X_train = X_reduced[train_inds,:]
            y_train = y[train_inds]

            # 3-fold grid search cross-validation
            CV_3fold = KFold(n_splits=3, shuffle=True, random_state=0)

            # kNN
            kNN = KNeighborsClassifier(weights="uniform")
            kNN_params = {'n_neighbors': [5,7,9]}
            clfkNN_GS = GridSearchCV(
                kNN, kNN_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
            )
            clfkNN_GS.fit(X_train, y_train)
            ypredArray_kNN[n_val] = clfkNN_GS.predict(X_validate)[0]

            # SVM
            svm_clf = SVC()
            svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
            clfsvm_GS = GridSearchCV(
                svm_clf, svm_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
            )
            clfsvm_GS.fit(X_train, y_train)
            ypredArray_SVM[n_val] = clfsvm_GS.predict(X_validate)[0]

            # Gaussian Naive-Bayes
            naiveBayesClf = GaussianNB()
            naiveBayesClf.fit(X_train, y_train)
            ypredArray_NBy[n_val] = naiveBayesClf.predict(X_validate)[0]

            # LDA
            ldaClf = LinearDiscriminantAnalysis()
            ldaClf.fit(X_train, y_train)
            ypredArray_LDA[n_val] = ldaClf.predict(X_validate)[0]

            # DT
            dt_clf = DecisionTreeClassifier(random_state=0)
            dt_params = {'splitter': ["best", "random"], 'max_depth': [3,5,7,9]}
            clfdt_GS = GridSearchCV(
                dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
            )
            clfdt_GS.fit(X_train, y_train)
            ypredArray_DT[n_val] = clfdt_GS.predict(X_validate)[0]


        balAcc_kNN = balanced_accuracy_score(y, ypredArray_kNN)
        balAcc_SVM = balanced_accuracy_score(y, ypredArray_SVM)
        balAcc_NBy = balanced_accuracy_score(y, ypredArray_NBy)
        balAcc_LDA = balanced_accuracy_score(y, ypredArray_LDA)
        balAcc_DT  = balanced_accuracy_score(y, ypredArray_DT)
        ds_results = pd.concat(
            [
                ds_results,
                pd.DataFrame(
                    data=[[balAcc_kNN, balAcc_SVM, balAcc_NBy, balAcc_LDA, balAcc_DT]],
                    index=[f"PDE-S-{n_}"], columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
                )
            ]
        )

    # Saving results per dataset
    ds_results.to_csv(saveFolder.joinpath(f"{ds}PDE-S_LOOCV.csv"))

sys.exit(0)
