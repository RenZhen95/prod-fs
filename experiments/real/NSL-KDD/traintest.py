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

if len(sys.argv) < 3:
    print("Possible usage: python3 traintest.py <datasetsFolder> <ranks>")
    sys.exit(1)
else:
    datasetsFolder = Path(sys.argv[1])
    ranks = Path(sys.argv[2])

fsorder = [
    "RlfF", "MSurf", "IRlf", "LHRlf",
    "RFGini", "MI", "mRMR", "FT", "ProD"
]
feature_ranks = pd.read_csv(ranks, index_col=0)

Xtrain = pd.read_csv(datasetsFolder.joinpath("Xtrain20.csv"), index_col=0).values
ytrain = pd.read_csv(datasetsFolder.joinpath("ytrain20.csv"), index_col=0)
ytrain = np.ravel(ytrain)

Xtest  = pd.read_csv(datasetsFolder.joinpath("Xtest.csv"), index_col=0).values
ytest = pd.read_csv(datasetsFolder.joinpath("ytest.csv"), index_col=0)
ytest = np.ravel(ytest)

results = pd.DataFrame(
    data=np.zeros((len(fsorder), 5)), index=fsorder,
    columns=["kNN", "SVM", "Gaussian-NB", "LDA", "DT"]
)

for fs in fsorder:
    print(f" - Feature selection scheme: {fs}")
    top_features = (feature_ranks[fs].values).astype(np.int64)

    Xtrain_reduced = Xtrain[:, top_features]
    Xtest_reduced  = Xtest[:, top_features]

    # 3-fold grid search cross-validation
    CV_3fold = KFold(n_splits=3, shuffle=True, random_state=0)

    # kNN
    kNN = KNeighborsClassifier(weights="uniform")
    kNN_params = {'n_neighbors': [5,7,9]}
    clfkNN_GS = GridSearchCV(
        kNN, kNN_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
    )
    clfkNN_GS.fit(Xtrain_reduced, ytrain)
    results.at[fs, "kNN"] = balanced_accuracy_score(
        ytest, clfkNN_GS.predict(Xtest_reduced)
    )
    
    # SVM
    svm_clf = SVC()
    svm_params = {'C': [1,10,100,1000], 'gamma': [0.001,0.0001], 'kernel': ['rbf']}
    clfsvm_GS = GridSearchCV(
        svm_clf, svm_params, cv=CV_3fold, n_jobs=-1, scoring="balanced_accuracy"
    )
    clfsvm_GS.fit(Xtrain_reduced, ytrain)
    results.at[fs, "SVM"] = balanced_accuracy_score(
        ytest, clfsvm_GS.predict(Xtest_reduced)
    )
    
    # Gaussian Naive-Bayes
    naiveBayesClf = GaussianNB()
    naiveBayesClf.fit(Xtrain_reduced, ytrain)
    results.at[fs, "Gaussian-NB"] = balanced_accuracy_score(
        ytest, naiveBayesClf.predict(Xtest_reduced)
    )
    
    # LDA
    ldaClf = LinearDiscriminantAnalysis()
    ldaClf.fit(Xtrain_reduced, ytrain)
    results.at[fs, "LDA"] = balanced_accuracy_score(
        ytest, ldaClf.predict(Xtest_reduced)
    )

    # DT
    dt_clf = DecisionTreeClassifier(random_state=0)
    dt_params = {'splitter': ["best", "random"], 'max_depth': [3,5,7,9]}
    clfdt_GS = GridSearchCV(
        dt_clf, dt_params, cv=CV_3fold, scoring="balanced_accuracy"
    )
    clfdt_GS.fit(Xtrain_reduced, ytrain)
    results.at[fs, "DT"] = balanced_accuracy_score(
        ytest, clfdt_GS.predict(Xtest_reduced)
    )

results.to_csv("traintest_clf.csv", sep=',')

sys.exit(0)
