import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

marks=np.random.randint(20,100,(100),int)

print(f"marks : {marks}")

minimum,q1,median,q3,maximum=np.quantile(marks,[0.0,0.25,0.50,0.75,1.0])

print(f"minimum : {minimum}")
print(f"q1 : {q1}")
print(f"median : {median}")
print(f"q3 : {q3}")
print(f"maximum : {maximum}")

#iqr = inter quartile range
iqr=q3-q1
print(iqr)

lower_fence=q1-1.5*iqr
upper_fence=q3+1.5*iqr

print(f"lower fence : {lower_fence}")
print(f"upper fence : {upper_fence}")

marks=np.append(marks,[lower_fence-5,lower_fence-10,upper_fence+10,upper_fence+5])

outliers=marks[(marks<lower_fence) | (marks>upper_fence)]

print(f"outliers.size : {outliers.size}")
print(f"outliers : {outliers}")


sns.boxplot(marks)

marks=np.append(marks,[lower_fence-50,lower_fence-100,upper_fence+100,upper_fence+50])

sns.boxplot(marks)
plt.show()
