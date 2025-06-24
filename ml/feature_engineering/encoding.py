import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df=pd.DataFrame({
    'color':['red','blue','green','green','red','blue'],
    'size':['small','medium','large','medium','small','large'],
    'gender':['male','female','male','female','male','female'],
})

df.head()

encoder=OneHotEncoder()

encoder.fit_transform(df[["color"]])

encoded_color=encoder.fit_transform(df[["color"]]).toarray()
encoded_color

encoded_size=encoder.fit_transform(df[["size"]]).toarray()
print(encoded_size)

df=pd.DataFrame(encoded_color,columns=encoder.get_feature_names_out())
print(df)



import seaborn as snb

df=snb.load_dataset("tips")

print(type(df))

print(f"{df.columns}")

print(f"{df.head()}")

print(f"sex : {df['sex'].unique()}")
print(f"smoker : {df['smoker'].unique()}")
print(f"day : {df['day'].unique()}")
print(f"time : {df['time'].unique()}")

print(df["sex"].value_counts())


print("implementing hot encoding\n")
lb_encoder=LabelEncoder()

df_sex_encoded=lb_encoder.fit_transform(df["sex"])
df_smoker_encoded=lb_encoder.fit_transform(df["smoker"])
df_day_encoded=lb_encoder.fit_transform(df["day"])
df_time_encoded=lb_encoder.fit_transform(df["time"])

print(type(df_sex_encoded))

df_sex_encoded=pd.DataFrame(df_sex_encoded,columns=["sex"])
df_smoker_encoded=pd.DataFrame(df_smoker_encoded,columns=["smoker"])
df_day_encoded=pd.DataFrame(df_day_encoded,columns=["day"])
df_time_encoded=pd.DataFrame(df_time_encoded,columns=["time"])

df1=pd.concat([df_sex_encoded,df_smoker_encoded,df_day_encoded,df_time_encoded],axis=1)

print(f"{df1.head()}")

print(f"{df1.columns}")

print(f"unique sex encoded : {df1['sex'].value_counts()}")
print(f"unique smoker encoded : {df1['smoker'].value_counts()}")
print(f"unique days encoded : {df1['day'].value_counts()}")
print(f"unique time encoded : {df1['time'].value_counts()}")


print("implementing ordinal encoder based on the rank of the values\n")
from sklearn.preprocessing import OrdinalEncoder


df=pd.DataFrame({
    "education":["school","college","graduate","post_graduate"]
})

print(df.head())

encoder=OrdinalEncoder(categories=[["school","college","graduate","post_graduate"]])

encoder.fit_transform(df[["education"]])


transformed_school = encoder.transform([["school"]])
transformed_college = encoder.transform([["college"]])
transformed_graduate = encoder.transform([["graduate"]])
transformed_pg = encoder.transform([["post_graduate"]])


print(f"school : {transformed_school}")
print(f"college : {transformed_college}")
print(f"graduate : {transformed_graduate}")
print(f"post graduate : {transformed_pg}")


print(f"implementing target guided ordinal encoding\n")

df=pd.DataFrame({
    "city""":["new_york","london","paris","tokyo","new_york","paris"],
    "price":[200,150,300,250,150,300]
})


print(df.head())

city_mean_price=df.groupby("city")["price"].mean()
print(city_mean_price)


df["city_encoded"]=df["city"].map(city_mean_price)

print(df.head())

x=df["city"]
y=df["city_encoded"]
print([x,y])







