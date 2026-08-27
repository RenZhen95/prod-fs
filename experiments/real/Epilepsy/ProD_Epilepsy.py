import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time

from prod_fs import ProD

X = pd.read_csv(Path("data") / "Xtrain.csv", index_col=0)
y = pd.read_csv(Path("data") / "ytrain.csv", index_col=0)
y = np.reshape(y, -1)

# Convert to float
X = np.array(X.values, dtype=float)

# According to Canedo (2012), 3 % of 178 features = 6 features
nRetainedFeatures = 6

def experiment_loop(bw_method, suffix):
    # === === === ===
    # Carrying out feature selection for each dataset
    elapsed_times = pd.Series(
        data=np.zeros(1), index=[f"ProD-{suffix}"]
    )
    
    scores_df = pd.DataFrame(
        data=np.zeros((X.shape[1], 2)),
        columns=["feature", f"ProD-{suffix}"]
    )
    scores_df["feature"] = np.arange(0, X.shape[1], 1)
    
    rank_df = pd.DataFrame(
        data=np.zeros((nRetainedFeatures, 2)),
        columns=["rank", f"ProD-{suffix}"]
    )
    rank_df["rank"] = np.arange(0, nRetainedFeatures, 1)
    
    # Proposed algorithm
    tProD_start = process_time()
    prodRanker = ProD(
        integration_method="trapz", delta=500, bw_method=bw_method,
        k=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5,
        averaging_method="mean"
    )
    prodRanker.fit(X, y)
    tProD_stop = process_time()
    tProD = tProD_stop - tProD_start
    
    # === === === === === === ===
    # GET ELAPSED TIME
    elapsed_times.at[f"ProD-{suffix}"] = tProD
    
    # === === === === === === ===
    # GETTING TOP N FEATURES
    rank_df.loc[:, f"ProD-{suffix}"] = prodRanker.get_topnFeatures(nRetainedFeatures)
    
    scores_df.loc[:, f"ProD-{suffix}"] = prodRanker.feature_importances_
    
    elapsed_times.to_csv(f"ProD-{suffix}_elapsed_times.csv")
    rank_df.to_csv(f"ProD-{suffix}_rank.csv")
    scores_df.to_csv(f"ProD-{suffix}_scores_df.csv")

# Call experiment loop
experiment_loop("scott", "Sco")
experiment_loop("customSilverman", "Slv")

sys.exit(0)
