import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import balanced_accuracy_score

if len(sys.argv) < 4:
    print(
        "Possible usage: python3.11 ProD_10stratifiedfoldcv.py <datasetsFolder> " +
        "<datasetName> <folder>"
    )
    sys.exit(1)
else:
    datasetsFolder = Path(sys.argv[1])
    datasetName = sys.argv[2]
    folder = Path(sys.argv[3])

# Loading datasets
with open(datasetsFolder.joinpath(f"{datasetName}_datasets.pkl"), "rb") as handle:
    datasets = pickle.load(handle)

# Reading the top 10 features
with open(folder.joinpath(f"ProD/{datasetName}ProD_ranks.pkl"), "rb") as handle:
    ranks_df = pickle.load(handle)

ranks_n30 = ranks_df[ranks_df["n_obs"] == 30.0]
ranks_n50 = ranks_df[ranks_df["n_obs"] == 50.0]
ranks_n70 = ranks_df[ranks_df["n_obs"] == 70.0]
ranks = {30: ranks_n30, 50: ranks_n50, 70: ranks_n70}

# 3 nObs x 50 iterations x 5 Classifiers (only for ProD)
performance_df = pd.DataFrame(
    data=np.zeros((3*50*5, 5)), columns=["Bal.Acc", "nObs", "Iteration", "FS", "Clf"]
)
performance_df["FS"] = performance_df["FS"].astype("object")
performance_df["Clf"] = performance_df["Clf"].astype("object")

skf = StratifiedKFold(n_splits=10)

count = 0
for nObs in [30, 50, 70]:
    for itr in range(50):
        ranks_peritr = ranks[nObs][ranks[nObs]["iteration"] == itr]
        ranks_peritr = ranks_peritr.drop(
            columns=["rank", "iteration", "n_obs"]
        )
        print(f"nObs: {nObs} | itr: {itr}")

        X = datasets[nObs][itr]['X']
        y = datasets[nObs][itr]['y']

        top_features = ranks_peritr["ProD"].to_numpy()
        top_features = list(map(int, top_features))
        print(top_features)

        X_reduced = X[:, top_features]

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
        performance_df.at[count, "nObs"] = nObs
        performance_df.at[count, "Iteration"] = itr
        performance_df.at[count, "FS"] = "ProD"
        performance_df.at[count, "Clf"] = "kNN"

        performance_df.at[count+1, "Bal.Acc"] = balAcc_SVM.mean()
        performance_df.at[count+1, "nObs"] = nObs
        performance_df.at[count+1, "Iteration"] = itr
        performance_df.at[count+1, "FS"] = "ProD"
        performance_df.at[count+1, "Clf"] = "SVM"

        performance_df.at[count+2, "Bal.Acc"] = balAcc_NB.mean()
        performance_df.at[count+2, "nObs"] = nObs
        performance_df.at[count+2, "Iteration"] = itr
        performance_df.at[count+2, "FS"] = "ProD"
        performance_df.at[count+2, "Clf"] = "NB"

        performance_df.at[count+3, "Bal.Acc"] = balAcc_LDA.mean()
        performance_df.at[count+3, "nObs"] = nObs
        performance_df.at[count+3, "Iteration"] = itr
        performance_df.at[count+3, "FS"] = "ProD"
        performance_df.at[count+3, "Clf"] = "LDA"

        performance_df.at[count+4, "Bal.Acc"] = balAcc_DT.mean()
        performance_df.at[count+4, "nObs"] = nObs
        performance_df.at[count+4, "Iteration"] = itr
        performance_df.at[count+4, "FS"] = "ProD"
        performance_df.at[count+4, "Clf"] = "DT"
        count += 5

performance_df.to_csv(f"{datasetName}10foldcvProD.csv")

averaged_df = performance_df.groupby(["nObs", "FS", "Clf"]).mean()
averaged_df = averaged_df.drop(columns=["Iteration"])
averaged_df.to_csv(f"{datasetName}10foldcvProD_averaged.csv")

sys.exit(0)
