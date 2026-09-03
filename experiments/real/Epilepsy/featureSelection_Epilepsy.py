import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time

from skrebate import ReliefF, MultiSURF
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

# Taking the top nRetainedFeatures
def get_indsTopnFeatures(_importances, _n):
    return sorted(
        range(len(_importances)),
        key=lambda i: _importances[i], reverse=True
    )[:_n]

X = pd.read_csv(Path("data") / "Xtrain.csv", index_col=0)
y = pd.read_csv(Path("data") / "ytrain.csv", index_col=0)
y = np.reshape(y, -1)

# Convert to float
X = np.array(X.values, dtype=float)

# According to Canedo (2012), 3 % of 178 features = 6 features
nRetainedFeatures = 6

# === === === ===
# Carrying out feature selection for each dataset
elapsed_times = pd.Series(
    data=np.zeros(5),
    index=[
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT"
    ]
)

scores_df = pd.DataFrame(
    data=np.zeros((X.shape[1], 6)),
    columns=[
        "feature",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT"
    ]
)
scores_df["feature"] = np.arange(0, X.shape[1], 1)

rank_df = pd.DataFrame(
    data=np.zeros((nRetainedFeatures, 6)),
    columns=[
        "rank",
        "RlfF",
        "MSurf",
        "RFGini",
        "MI",
        "FT"
    ]
)
rank_df["rank"] = np.arange(0, nRetainedFeatures, 1)


# === === === === === === ===
# FEATURE RANKING METHODS
# From scikit-rebate (https://github.com/EpistasisLab/scikit-rebate)
# ReliefF
print("ReliefF")
tRlfF_start = process_time()
RlfF = ReliefF(n_neighbors=7, n_jobs=-1) # From Cai, 2014
RlfF.fit(X, y)
tRlfF_stop = process_time()
tRlfF = tRlfF_stop - tRlfF_start

# MultiSURF
print("MultiSURF")
tMSurf_start = process_time()
MSurf = MultiSURF(n_jobs=-1)
MSurf.fit(X, y)
tMSurf_stop = process_time()
tMSurf = tMSurf_stop - tMSurf_start

# From scikit-learn
# Mutual Information
print("Mutual Information")
tMI_start = process_time()
resMI = mutual_info_classif(X, y, n_neighbors=7, random_state=0, n_jobs=-1)
tMI_stop = process_time()
tMI = tMI_stop - tMI_start

# ANOVA F-value
print("ANOVA-F")
tFT_start = process_time()
resFT_stat, resFT_p = f_classif(X, y)
tFT_stop = process_time()
tFT = tFT_stop - tFT_start

# Random forest ensemble data mining to increase information gain/reduce impurity
print("RFGini")
tRF_start = process_time()
rfGini = RandomForestClassifier(
    n_estimators=1000, criterion="gini", random_state=0, n_jobs=-1
)
rfGini.fit(X, y)
tRF_stop = process_time()
tRF = tRF_stop - tRF_start

# === === === === === === ===
# GET ELAPSED TIME
elapsed_times.at["RlfF"] = tRlfF
elapsed_times.at["MSurf"] = tMSurf
elapsed_times.at["RFGini"] = tRF
elapsed_times.at["MI"] = tMI
elapsed_times.at["FT"] = tFT

# === === === === === === ===
# GETTING TOP N FEATURES
rank_df.loc[:, "RlfF"] = get_indsTopnFeatures(
    RlfF.feature_importances_, nRetainedFeatures
)
rank_df.loc[:, "MSurf"] = get_indsTopnFeatures(
    MSurf.feature_importances_, nRetainedFeatures
)
rank_df.loc[:, "MI"] = get_indsTopnFeatures(
    resMI, nRetainedFeatures
)
rank_df.loc[:, "RFGini"] = get_indsTopnFeatures(
    rfGini.feature_importances_, nRetainedFeatures
)
rank_df.loc[:, "FT"] = get_indsTopnFeatures(
    resFT_stat, nRetainedFeatures
)

scores_df.loc[:, "RlfF"] = RlfF.feature_importances_
scores_df.loc[:, "MSurf"] = MSurf.feature_importances_
scores_df.loc[:, "MI"] = resMI
scores_df.loc[:, "RFGini"] = rfGini.feature_importances_
scores_df.loc[:, "FT"] = resFT_stat

elapsed_times.to_csv("Epilepsyelapsed_times.csv")
rank_df.to_csv("Epilepsyrank.csv")
scores_df.to_csv("Epilepsyscores_df.csv")

sys.exit(0)
