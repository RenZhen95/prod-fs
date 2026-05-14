## Steps to reproduce results
To reconstruct results from the paper, run the scripts in the following order as shown below.

**Note**: 
- The following feature selection methods are implemented in Matlab:
  - IRelief [(Sun et al., 2010)][IRlf]
  - LHR [(Cai et al., 2014)][LHR]
  - mRMR [(Ding and Peng, 2003)][mRMR]

- All the scripts require modifications to the paths of corresponding input files, either as input arguments or hard-coded in the scripts 

## Results from synthetic dataset
### Electrical datasets
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

Generated datasets might show random variations due to how computers handle randomness. For an exact reproduction, use the datasets in Electrical-Datasets.zip

**Step 1**

Create a folder for each of the datasets above, each with the following subfolders:
-  ProD
-  OtherFS
-  IRelief
-  LHRelief
-  mRMR
-  Combined

---
Now, for each dataset carry out **Steps 2-5** ... 

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

Also carry out feature selection with methods implemented in Matlab:
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

### SM datasets
**Step 0**

The SM datasets are generated according to the method proposed by [Diaz et al., 2006][SM]. The synthetic datasets are also available at [https://github.com/rdiaz02/varSelRF-suppl-mat][SMDatasets]. However, for a more accurate reproduction, unzip the compressed folder SM-Datasets.zip to use the datasets used in our paper, which have been preprocessed to better suit our "style".

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

Also carry out feature selection with methods implemented in Matlab:
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

  The <trueSignatures_folder> should be the path to the folder `trueSignatures` zipped in SM-Datasets.zip

- Classification accuracy via 10-fold stratified cross-validation
  ```
  python3 synthetic/SM/ProD_10stratifiedcv.py
  python3 synthetic/SM/10stratifiedcv.py
  ```

  The scripts for running the stratified 10-fold CV are split between ProD and the other FS as part of the development process, but they are essentially the same. Feel free to combine them.

---
[syntElectrical]: <https://doi.org/10.48550/arXiv.2211.03035>
[IRlf]: <https://doi.org/10.1109/TPAMI.2009.190>
[LHR]: <https://doi.org/10.1186/1471-2105-15-70>
[mRMR]: <https://doi.org/10.1109/CSB.2003.1227396>
[SucRate]: <https://doi.org/10.1007/s10115-012-0487-8>
[SM]: <https://doi.org/10.1186/1471-2105-7-3>
[SMDatasets]: <https://github.com/rdiaz02/varSelRF-suppl-mat>
