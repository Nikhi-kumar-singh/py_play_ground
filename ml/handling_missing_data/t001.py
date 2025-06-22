import seaborn as snb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv("titanic.csv")

print(f"head of the dataset : \n{df.head()}")

print(f"shape of the dataset : \n{df.shape}")

print(f"size of the dataset : \n{df.size}")

print(f"description of the dataset : \n{df.describe()}")

print(f"number of null values : \n {df.isnull().sum()}")

print(f"shape of null values : \n {df.isnull().shape}")

df1=df.dropna()
print(f"shape of non-null values : \n {df1.shape}")

print(f"number of non-null values : \n {df1.isnull().sum()}")


df["age_mean"]=df["age"].fillna(df["age"].mean())
df["age_median"]=df["age"].fillna(df["age"].median())
mode_val=df["age"].mode()[0]
df["age_mode"]=df["age"].fillna(mode_val)

print("comparison among the new imputed data and old data:\n")
print(df[["age","age_mean","age_median","age_mode"]])


snb.displot(df,x="age",kde=True)
snb.displot(df,x="age_mean",kde=True)
snb.displot(df,x="age_median",kde=True)
snb.displot(df,x="age_mode",kde=True)

plt.show()