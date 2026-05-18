import pickle
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from time import process_time
from collections import defaultdict

from prod_fs import ProD

if len(sys.argv) < 3:
    print(
        "Possible usage: python3 ProD_real.py <processedDatasets> <savefolder>"
    )
    sys.exit(1)
else:
    processedDatasets = Path(sys.argv[1])
    savefolder = Path(sys.argv[2])
    nRetainedFeatures = 100

with open(processedDatasets, "rb") as handle:
    datasets_dict = pickle.load(handle)

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

    # Proposed algorithm
    tProD_start = process_time()
    prodRanker = ProD(
        integration_method="trapz", delta=500, bw_method="scott",
        k=2, n_jobs=-1, mode="release", lower_end=-1.5, upper_end=2.5,
        averaging_method="weighted"
    )
    prodRanker.fit(X, y)
    tProD_stop = process_time()
    tProD = tProD_stop - tProD_start

    # === === === === === === ===
    # GETTING TOP N FEATURES
    inds_topFeatures_ProD = prodRanker.get_topnFeatures(
        nRetainedFeatures
    )
    dataset_inds_topFeatures[dataset] = inds_topFeatures_ProD

    # === === === === === === ===
    # GET ELAPSED TIME
    elapsed_times_perDS[dataset] = tProD

with open(savefolder.joinpath(f"top{nRetainedFeatures}Features_ProD.pkl"), "wb") as handle:
    pickle.dump(dataset_inds_topFeatures, handle)
with open(savefolder.joinpath(f"fsElapsedTimes_ProD.pkl"), "wb") as handle:
    pickle.dump(elapsed_times_perDS, handle)

sys.exit(0)
