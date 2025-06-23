import pandas as pd
import numpy as np
from sklearn.utils import resample


r1=0.1
r2=1-r1

n=1000
n1=int(r1*n)
n2=int(r2*n)

class1=pd.DataFrame({ 
 "f1":np.random.normal(loc=0,scale=1,size=n1),
 "f2":np.random.normal(loc=0,scale=1,size=n1),
 "target":[1]*n1
                     })



class2=pd.DataFrame({
 "f1":np.random.normal(loc=0,scale=1,size=n2),
 "f2":np.random.normal(loc=0,scale=1,size=n2),
 "target":[2]*n2
 })


print(f"class1 shape : {class1.shape}")
print(f"class2 shape : {class2.shape}")



df=pd.concat([class1,class2]).reset_index(drop=True)
#print(f"df : \n {df.head()}")


df_min=df[df["target"]==1]
df_maj=df[df["target"]==2]

df_min_up_sampled=resample(
 df_min,
 replace=True,
 n_samples=len(df_maj),
 random_state=43
)



df_maj_down_sampled=resample(
 df_maj,
 replace=True,
 n_samples=len(df_min),
 random_state=43
)

print(f"minority data shape: {df_min.shape}")
print(f"minority up samples data shape : {df_min_up_sampled.shape}")

print(f"majority data shape: {df_maj.shape}")
print(f"majority down samples data shape : {df_maj_down_sampled.shape}")

df_up_sampled=pd.concat([df_maj,df_min_up_sampled]).reset_index(drop=True)
df_down_sampled=pd.concat([df_maj_down_sampled,df_min]).reset_index(drop=True)


print(f"shape of up sampled data : {df_up_sampled.shape}")
print(f"shape of down sampled data : {df_down_sampled.shape}")
