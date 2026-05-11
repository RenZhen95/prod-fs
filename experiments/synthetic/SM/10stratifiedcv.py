import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 2:
    print(
        "Possible usage: python3.11 10stratifiedcv.py <datasetsFolder> <folder>"
    )
    sys.exit(1)
else:
    datasetsFolder = Path(sys.argv[1])
    folder = Path(sys.argv[2])

XFolder = SDIFolder.joinpath('X')
yFolder = SDIFolder.joinpath('y')

# Getting datasets
nClass2_idxs = [16, 43, 70, 97]
nClass3_idxs = [17, 44, 71, 98]
nClass4_idxs = [18, 45, 72, 99]

def read_X_y(_idxs):
    X_dict = defaultdict(); y_dict = defaultdict()

    for i, idx in enumerate(_idxs):
        X = pd.read_csv(XFolder.joinpath(f"{idx}_X.csv"), sep='\s+', header=None)
        y = pd.read_csv(yFolder.joinpath(f"{idx}_y.csv"), header=None)

        X_dict[i] = X
        y_dict[i] = np.reshape(y, -1)

    return X_dict, y_dict

nClass2_X, nClass2_y = read_X_y(nClass2_idxs)
nClass3_X, nClass3_y = read_X_y(nClass3_idxs)
nClass4_X, nClass4_y = read_X_y(nClass4_idxs)

X_dict = {2: nClass2_X, 3: nClass3_X, 4: nClass4_X}
y_dict = {2: nClass2_y, 3: nClass3_y, 4: nClass4_y}

# Reading the top 120 features
combinedFolder = folder.joinpath("Combined")
ranks_df = pd.read_csv(
    combinedFolder.joinpath("SDIranks.csv"), index_col=0
)

ranks_nClass2 = ranks_df[ranks_df["nClass"] == 2.0]
ranks_nClass3 = ranks_df[ranks_df["nClass"] == 3.0]
ranks_nClass4 = ranks_df[ranks_df["nClass"] == 4.0]
ranks = {2: ranks_nClass2, 3: ranks_nClass3, 4: ranks_nClass4}

fs_methods = [
    "RlfF", "MSurf", "IRlf", "LHRlf", "mRMR", "RFGini", "MI", "FT"
]

# 3 nClass x 4 iterations x 8 FS x 5 Classifiers
performance_df = pd.DataFrame(
    data=np.zeros((3*4*8*5, 5)), columns=["Bal.Acc", "nClass", "Iteration", "FS", "Clf"]
)
performance_df["FS"] = performance_df["FS"].astype("object")
performance_df["Clf"] = performance_df["Clf"].astype("object")

skf = StratifiedKFold(n_splits=10)

count = 0
for nClass in [2, 3, 4]:
    for itr in range(4):
        ranks_peritr = ranks[nClass][ranks[nClass]["iteration"] == itr]
        ranks_peritr = ranks_peritr.drop(
            columns=["rank", "iteration", "nClass"]
        )
        print(f"nClass: {nClass} | itr: {itr}")
        print(ranks_peritr)

        X = X_dict[nClass][itr].values
        y = y_dict[nClass][itr]
        print(X.shape)

        for fs in fs_methods:
            top_features = ranks_peritr[fs].to_numpy()
            top_features = list(map(int, top_features))

            X_reduced = X[:,top_features]

            balAcc_kNN = np.zeros(10)
            balAcc_SVM = np.zeros(10)
            balAcc_NB  = np.zeros(10)
            balAcc_LDA = np.zeros(10)
            balAcc_DT  = np.zeros(10)

            # Carry out stratified k-fold
            for fold, (train_index, test_index) in enumerate(skf.split(X_reduced, y)):
                X_validate = X_reduced[test_index, :]
                y_validate = y[test_index]

                X_train = X_reduced[train_index, :]
                y_train = y[train_index]

                # 3-fold grid search cross-validation
                CV_3fold = KFold(n_splits=3, shuffle=True, random_state=0)

                # kNN
                kNN = KNeighborsClassifier(weights="uniform")
                kNN_params = {'n_neighbors': [5,7,9]}
                clfkNN_GS = GridSearchCV(
                    kNN, kNN_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
                )
                clfkNN_GS.fit(X_train, y_train)
                balAcc_kNN[fold] = balanced_accuracy_score(
                    y_validate, clfkNN_GS.predict(X_validate)
                )

                # SVM
                svm_clf = SVC()
                svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
                clfsvm_GS = GridSearchCV(
                    svm_clf, svm_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy",
                    error_score="raise"
                )
                clfsvm_GS.fit(X_train, y_train)
                balAcc_SVM[fold] = balanced_accuracy_score(
                    y_validate, clfsvm_GS.predict(X_validate)
                )

                # Gaussian Naive-Bayes
                naiveBayesClf = GaussianNB()
                naiveBayesClf.fit(X_train, y_train)
                balAcc_NB[fold] = balanced_accuracy_score(
                    y_validate, naiveBayesClf.predict(X_validate)
                )

                # LDA
                ldaClf = LinearDiscriminantAnalysis()
                ldaClf.fit(X_train, y_train)
                balAcc_LDA[fold] = balanced_accuracy_score(
                    y_validate, ldaClf.predict(X_validate)
                )

                # DT
                dt_clf = DecisionTreeClassifier(random_state=0)
                dt_params = {'splitter': ["best","random"], 'max_depth': [3,5,7,9]}
                clfdt_GS = GridSearchCV(
                    dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
                )
                clfdt_GS.fit(X_train, y_train)
                balAcc_DT[fold] = balanced_accuracy_score(
                    y_validate, clfdt_GS.predict(X_validate)
                )

            performance_df.at[count, "Bal.Acc"] = balAcc_kNN.mean()
            performance_df.at[count, "nClass"] = nClass
            performance_df.at[count, "Iteration"] = itr
            performance_df.at[count, "FS"] = fs
            performance_df.at[count, "Clf"] = "kNN"

            performance_df.at[count+1, "Bal.Acc"] = balAcc_SVM.mean()
            performance_df.at[count+1, "nClass"] = nClass
            performance_df.at[count+1, "Iteration"] = itr
            performance_df.at[count+1, "FS"] = fs
            performance_df.at[count+1, "Clf"] = "SVM"

            performance_df.at[count+2, "Bal.Acc"] = balAcc_NB.mean()
            performance_df.at[count+2, "nClass"] = nClass
            performance_df.at[count+2, "Iteration"] = itr
            performance_df.at[count+2, "FS"] = fs
            performance_df.at[count+2, "Clf"] = "NB"

            performance_df.at[count+3, "Bal.Acc"] = balAcc_LDA.mean()
            performance_df.at[count+3, "nClass"] = nClass
            performance_df.at[count+3, "Iteration"] = itr
            performance_df.at[count+3, "FS"] = fs
            performance_df.at[count+3, "Clf"] = "LDA"

            performance_df.at[count+4, "Bal.Acc"] = balAcc_DT.mean()
            performance_df.at[count+4, "nClass"] = nClass
            performance_df.at[count+4, "Iteration"] = itr
            performance_df.at[count+4, "FS"] = fs
            performance_df.at[count+4, "Clf"] = "DT"
            count += 5

performance_df.to_csv("10foldcv.csv")

averaged_df = performance_df.groupby(["nClass", "FS", "Clf"]).mean()
averaged_df = averaged_df.drop(columns=["Iteration"])
averaged_df.to_csv("10foldcv_averaged.csv")

sys.exit(0)
