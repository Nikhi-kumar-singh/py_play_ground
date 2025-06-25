import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("winequality-red.csv")

print(f"the dataset head part : \n{df.head()}")

print(f"columns : {df.columns}")

print(f"data types of dataset : \n {df.dtypes} ")

print(f"shape of the dataset : {df.shape}")

print(f"information about the data : \n {df.info()}")

print(f"mathematical description about the dataset : \n {df.describe()}")

print(f"qualities of wine : {df['quality'].unique()}")

print(f"number of null values : \n{df.isnull().sum()}")

print(f"duplicate values :\n{df.duplicated()}")

print(f"number of duplicate values :\n{df.duplicated().sum()}")

df1=df.drop_duplicates(inplace=False)

print(f"shape of not duplicated values in dataset : {df1.shape}")

print(f"correlation among different features :\n {df.corr()}")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr())
plt.show()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

print(f"quality count(check balance in data) :\n {df['quality'].value_counts()}")

# Valid plot kinds: ('line', 'bar', 'barh', 'kde', 'density', 'area', 'hist', 'box', 'pie', 'scatter', 'hexbin')
df["quality"].value_counts().plot(kind="pie")
plt.show()

df["quality"].value_counts().plot(kind="hist")
plt.show()

df["quality"].value_counts().plot(kind="bar")
plt.show()

df["quality"].value_counts().plot(kind="kde")
plt.show()

for column in df.columns:
 sns.histplot(df[column],kde=True)
 plt.show()

for column in df.columns:
 sns.histplot(df[column],kde=True)
plt.show()

sns.pairplot(df)
plt.show()

sns.catplot(x="alcohol",y="quality",data=df,kind="bar")
plt.show()

sns.scatterplot(x="alcohol",y="pH",hue="quality",data=df)
plt.show()

