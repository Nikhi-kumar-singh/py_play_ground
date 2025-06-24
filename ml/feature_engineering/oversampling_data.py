from sklearn.datasets  import make_classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

x,y=make_classification(
 n_samples=1000,
 n_redundant=0,
 n_features=2,
 n_clusters_per_class=1,
 weights=[0.90],
 random_state=12
)

df1=pd.DataFrame(x,columns=["f1","f2"])
df2=pd.DataFrame(y,columns=["target"])



df=pd.concat([df1,df2],axis=1)

print(f"shape of the dataset : {df.shape}")
print(df.head())

plt.scatter(
 df["f1"],
 df["f2"],
 c=df["target"]
)
plt.show()

print(f"SMOTE is successfully installed")

oversample=SMOTE()

x,y=oversample.fit_resample(df[["f1","f2"]],df["target"])

df1=pd.concat([x,y],axis=1)

print(f"shape of the dataset : {df1.shape}")
print(df1.head())

plt.title("old dataset")
plt.scatter(df["f1"],df["f2"],c=df["target"])
plt.show()

plt.title("oversampled dataset")
plt.scatter(df1["f1"],df1["f2"],c=df1["target"])
plt.show()

n1=df["target"].value_counts()
n2=df1["target"].value_counts()

print(f"the data of old dataset : {n1}")
print(f"the data of new oversampled data : {n2}")




