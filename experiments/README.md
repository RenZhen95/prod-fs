## Steps to reproduce results
To reconstruct results from the paper, run the scripts in the following order as shown below.

**Note**: 
- The following feature selection methods are implemented in MATLAB:
  - IRelief [(Sun et al., 2010)][IRlf]
  - LHR [(Cai et al., 2014)][LHR]
  - mRMR [(Ding and Peng, 2003)][mRMR]

- All the scripts require modifications to the paths of corresponding input files, either as input arguments or hard-coded in the scripts 

## Results from Synthetic Datasets
### Electrical Datasets
**Step 0**

Generate the datasets by runing the script (code adapted from paper by [Kamalov et al., 2022][syntElectrical]):
```
python3 synthetic/Electrical/generate_artificialdatasets.py
```

This will produce four Python-pickled binary files:
- ANDOR continuous datasets
- ADDER continuous datasets
- ANDOR discrete datasets
- ADDER discrete datasets

Generated datasets might show random variations due to how computers handle randomness. For an exact reproduction, use the datasets in `Electrical-Datasets.zip`.

**Step 1**

Create a folder for each of the datasets above, each with the following subfolders:
-  ProD
-  OtherFS
-  IRelief
-  LHRelief
-  mRMR
-  Combined

Now, for each dataset carry out **Steps 2-5**.

**Step 2**

Carry out feature selection with ProD:
```  
python3 synthetic/Electrical/ProD_Electrical.py
```

Save the output files in the ProD subfolder made in **Step 0**.

For the argument `<dataset_name>`, keep them consistent across the next few Python and MATLAB scripts. So for example use `ANDORdiscrete` for the ANDOR discrete datasets and make sure this is consistently used from Steps 2-5.

**Step 3**

Carry out feature selection with other feature selection methods implemented in Python:
```  
python3 synthetic/Electrical/featureSelection_Electrical.py
```

Save the output files in the OtherFS subfolder made in **Step 0**.

Also carry out feature selection with methods implemented in MATLAB:
- synthetic/matlabFS/featureSelection_Electrical.m (I-Relief and LHR)
- synthetic/matlabFS/featureSelection_mRMR_Electrical.m (mRMR for continuous electrical datasets)
- synthetic/matlabFS/featureSelection_mRMR_ElectricalDiscrete.m (mRMR for discrete electrical datasets)

Save the corresponding output files in the IRelief, LHRelief, and mRMR subfolders made in **Step 0**.

**Step 4**

Carry out preprocessing step to combine feature selection results from the different scripts.
```
python3 synthetic/Electrical/combine_fss.py
```

**Step 5**

Evaluations
- Success rates according to [Bolón-Canedo et al., 2013][SucRate]
  ```
  python3 synthetic/Electrical/evaluate_fss.py
  ```

- Classification accuracy via 10-fold stratified cross-validation
  ```
  python3 synthetic/Electrical/ProD_10stratifiedcv.py
  python3 synthetic/Electrical/10stratifiedcv.py
  ```

  The scripts for running the stratified 10-fold CV are split between ProD and the other FS as part of the development process, but they are essentially the same. Feel free to combine them.

### SM Datasets
**Step 0**

The SM datasets are generated according to the method proposed by [Diaz et al., 2006][SM]. The synthetic datasets are also available at [https://github.com/rdiaz02/varSelRF-suppl-mat][SMDatasets]. However, for a more accurate reproduction, unzip the compressed folder `SM-Datasets.zip` to use the datasets used in our paper, which have been preprocessed to better suit our "style".

**Step 1**

Create the following subfolders:
-  ProD
-  OtherFS
-  IRelief
-  LHRelief
-  mRMR
-  Combined

**Step 2**

Carry out feature selection with ProD:
```  
python3 synthetic/SM/ProD_SM.py
```

Save the output files in the ProD subfolder made in **Step 0**.

**Step 3**

Carry out feature selection with other feature selection methods implemented in Python:
```  
python3 synthetic/SM/featureSelection_SM.py
```

Save the output files in the OtherFS subfolder made in **Step 0**.

Also carry out feature selection with methods implemented in MATLAB:
- synthetic/matlabFS/featureSelection_SM.m (I-Relief and LHR)
- synthetic/matlabFS/featureSelection_mRMR_SM.m (mRMR)

Save the corresponding output files in the IRelief, LHRelief, and mRMR subfolders made in **Step 0**.

**Step 4**

Carry out preprocessing step to combine feature selection results from the different scripts.
```
python3 synthetic/SM/combine_fss.py
```

**Step 5**

Evaluations
- Success rates according to [Bolón-Canedo et al., 2013][SucRate]
  ```
  python3 synthetic/SM/evaluate_fss.py
  ```

  The `<trueSignatures_folder>` should be the path to the folder `trueSignatures` zipped in SM-Datasets.zip

- Classification accuracy via 10-fold stratified cross-validation
  ```
  python3 synthetic/SM/ProD_10stratifiedcv.py
  python3 synthetic/SM/10stratifiedcv.py
  ```

  The scripts for running the stratified 10-fold CV are split between ProD and the other FS as part of the development process, but they are essentially the same. Feel free to combine them.

## Results from Real Datasets
**Step 0**

Download the datasets from the links provided in the paper, but for an exact reproduction, use the pickled datasets in `ProcessedDatasets.pkl`, zipped in `ProcessedDatasets.zip`. This will yield the following datasets:
1. CNS
2. Lung
3. Leukemia
4. Colon
5. Prostrate3
6. GCM
7. Cancer
8. Gait

The NSL-KDD dataset is handled differently as it has a separate training and test dataset. The dataset and code is found within the `real/NSL-KDD` folder. The dataset can be unzipped from `real/NSL-KDD/DatasetNSL-KDD.zip`. Steps to reproduce results for the NSL-KDD dataset is described in next section.

**Step 1**

Carry out feature selection with ProD:
```
python3 real/ProD_real.py
```

**Step 2**

Carry out feature selection with other feature selection methods implemented in Python:
```  
python3 real/featureSelection_real.py
```

**Step 3**

Carry out feature selection with feature selection methods implemented in MATLAB. The scripts are found in `real/matlabFS`. Adjust the paths to the datasets accordingly.


**Step 4**

Carry out Leave-One-Out-Cross-Validation on datasets using the features ranked by ProD
```
python3 real/ProD_loocv.py
```

The argument `<selFeatures>` should be the path to the pickled Python file containing the ranking of top features by ProD, that was output from **Step 1**.

**Step 5**

Carry out Leave-One-Out-Cross-Validation (LOOCV) on datasets using the features ranked by other feature selection methods
```
python3 real/loocv.py
```

Similarly, the argument `<selFeatures>` should be the path to the pickled Python file containing the ranking of top features from the other feature selection methods, that was output from **Step 2**.

**Step 6**

Combine the LOOCV results from the other FS methods and ProD, to save the results in a CSV file
```
python3 real/combine_loocv.py
```

This will output the following files:
- cns_LOOCV.csv
- lung_LOOCV.csv
- leuk_LOOCV.csv
- colon_LOOCV.csv
- pros3_LOOCV.csv
- gcm_LOOCV.csv
- cancer_LOOCV.csv
- gait_LOOCV.csv
- fsElapsedTimes.pkl

## Results from NSL-KDD Dataset
**Step 0**

The NSL-KDD dataset is handled differently as it has a separate training and test dataset. The data can be downloaded [here][NSL-KDD]. Cite the paper by Tavallaee et al., 2009. Otherwise, unzip `real/NSL-KDD/DatasetNSL-KDD.zip`.

**Step 1**

Carry out feature selection with ProD:
```
python3 real/NSL-KDD/ProD_NSLKDD.py
```

**Step 2**

Carry out feature selection with feature selection methods implemented in MATLAB. The script needed is `real/NSL-KDD/matlabFS/featureSelection_NSLKDD.m`. Adjust the paths to the datasets accordingly.

**Step 3**

Carry out feature selection with other feature selection methods implemented in Python:
```  
python3 real/NSL-KDD/featureSelection_NSLKDD.py
```

**Step 4**

Combine all the feature scores/rankings from the different feature selection methods to be used for subsequent training and testing.
```
python3 real/NSL-KDD/combine_fss.py
```

**Step 5**
Train classifiers based on features selected
```
python3 real/NSL-KDD/traintest.py
```

The argument `<ranks>` should be `ranks.csv` that was output earlier from **Step 4**.

[syntElectrical]: <https://doi.org/10.48550/arXiv.2211.03035>
[IRlf]: <https://doi.org/10.1109/TPAMI.2009.190>
[LHR]: <https://doi.org/10.1186/1471-2105-15-70>
[mRMR]: <https://doi.org/10.1109/CSB.2003.1227396>
[SucRate]: <https://doi.org/10.1007/s10115-012-0487-8>
[SM]: <https://doi.org/10.1186/1471-2105-7-3>
[SMDatasets]: <https://github.com/rdiaz02/varSelRF-suppl-mat>
[NSL-KDD]: <https://www.kaggle.com/datasets/hassan06/nslkdd>
