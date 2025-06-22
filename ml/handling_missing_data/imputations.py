import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv("titanic.csv")
#print(df.shape)
#print(df.isnull().sum())


sns.displot(df["age"],kde=True)
plt.show()
















