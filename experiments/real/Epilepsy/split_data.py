import sys
import pandas as pd
from sklearn.model_selection import train_test_split

'''
There are 5 labels (1, 2, 3, 4, 5) but most researchers treat this dataset
as a binary classification task, where 1 indicates an epileptic seizure, and
2, 3, 4, and 5 otherwise.
'''

df = pd.read_csv("data/original.csv", index_col=0)
Xdf = df.iloc[:,:-1]
ydf = df.iloc[:,-1]

# Group all non-epileptic recordings together
label_map = {1:1, 2:2, 3:2, 4:2, 5:2}
ydf = ydf.map(label_map)

y1 = ydf[ydf == 1]
y2 = ydf[ydf == 2]

print(f"Number of epileptic cases     : {y1.count()}")
print(f"Number of non-epileptic cases : {y2.count()}")

# Split dataset 80/20
X, Xtest, y, ytest = train_test_split(
    Xdf, ydf, test_size=0.2, random_state=0, stratify=ydf
)

print("----------------")
print("Training dataset")
print("----------------")
print(X)
print(X.shape)
X.to_csv("data/Xtrain.csv")
print(y)
print(y.shape)
y.to_csv("data/ytrain.csv")

print("------------")
print("Test dataset")
print("------------")
print(Xtest)
print(Xtest.shape)
Xtest.to_csv("data/Xtest.csv")
print(ytest)
print(ytest.shape)
ytest.to_csv("data/ytest.csv")
