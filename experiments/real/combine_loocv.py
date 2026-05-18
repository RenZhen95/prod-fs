import os, sys
import pickle
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3.11 combine_pyFS.py <loocvFolder>")
    print(
        "Script to combine the loocv and computational time from FS methods " +
        "implemented in Python"
    )
    sys.exit(1)
else:
    loocvFolder = Path(sys.argv[1])

# First deal with results from LOOCV
loocvProdFiles = {}
loocvOthrFiles = {}
for f in os.scandir(loocvFolder):
    # Separate files for other FS methods and ProD
    if "_LOOCV.csv" in f.name:
        if "ProD" in f.name:
            loocvProdFiles[f.name.split("ProD")[0]] = f
        else:
            loocvOthrFiles[f.name.split('_')[0]] = f

for k, v in loocvOthrFiles.items():

    print(f"Combining the LOOCV for the dataset {k} ... ")

    df_othrfs = pd.read_csv(v, index_col=0)
    df_prod   = pd.read_csv(loocvProdFiles[k], index_col=0)

    combined_df = pd.concat([df_othrfs, df_prod])
    combined_df.to_csv(f"{k}_LOOCV.csv")

with open(loocvFolder.joinpath("fsElapsedTimes.pkl"), "rb") as handle:
    othrfs_time = pickle.load(handle)

with open(loocvFolder.joinpath("fsElapsedTimes_ProD.pkl"), "rb") as handle:
    prod_time = pickle.load(handle)

for ds in othrfs_time.keys():
    othrfs_time[ds].update({"ProD": prod_time[ds]})

with open("fsElapsedTimes.pkl", "wb") as handle:
    pickle.dump(othrfs_time, handle)

sys.exit(0)
