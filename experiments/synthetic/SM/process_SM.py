"""
Simulated datasets from Ramon Diaz-Uriarte (2006)

1. Different number of classes: [2, 3, 4]
2. Number of independent dimensions: [1, 2, 3]
 - Each independent dimension has same relevance for discrimination of classes
3. Number of genes per dimension: [5, 20, 100]
4. Four replicate datasets for each level of number of classes with non-signals
 - All 4000 genes are random

25 subjects per class

Random features 
 - 2000 random normal variates (mean: 0, variance; 1)
 - 2000 random uniform [-1, 1] variates

Number of combinations: 3 x 3 x 3 = 27

"""
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Possible usage: python3.11 <SDIfolder>")
    sys.exit(1)
else:
    SDIFolder = Path(sys.argv[1])

XFolders = SDIFolder.joinpath('X')
yFolders = SDIFolder.joinpath('y')
trueSignatureFolders = SDIFolder.joinpath("trueSignatures")
n_classes = pd.read_csv(SDIFolder.joinpath("n_classes.csv"), header=None)

datasets_meta = pd.DataFrame(
    data=np.zeros((108, 3)),
    columns=["nClass", "nDimension", "nGenes_perDim"]
)

for i in range(1, 109):
    trueSignatures = pd.read_csv(
        trueSignatureFolders.joinpath(f"{i}_trueSignatures.csv"), header=None
    )
    datasets_meta.at[i-1, "nDimension"] = trueSignatures.shape[0]
    datasets_meta.at[i-1, "nGenes_perDim"] = trueSignatures.shape[1]

    y = pd.read_table(yFolders.joinpath(f"{i}_y.csv"), sep='\s+', header=None)
    sety = list(set(y.stack().values))
    datasets_meta.at[i-1, "nClass"] = len(sety)

datasets_meta_nClass2 = datasets_meta[datasets_meta["nClass"]==2.0]
datasets_meta_nClass3 = datasets_meta[datasets_meta["nClass"]==3.0]
datasets_meta_nClass4 = datasets_meta[datasets_meta["nClass"]==4.0]

datasets_meta_nClass2.to_csv("SDI_nClass2.csv")
datasets_meta_nClass3.to_csv("SDI_nClass3.csv")
datasets_meta_nClass4.to_csv("SDI_nClass4.csv")

sys.exit(0)
