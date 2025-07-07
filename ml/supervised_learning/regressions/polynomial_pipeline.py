import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score



def polynomial_regression(degree,x,y):
    poly_features=PolynomialFeatures(degree=degree,include_bias=True)
    lin_reg=LinearRegression()

    model=Pipeline([
        ("poly_features",poly_features),
        ("lin_reg",lin_reg)
    ])

    # model=Pipeline([
    #     ("feature_expansion",poly_features),
    #     ("regressor",lin_reg)
    # ])

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    plt.scatter(x_train,y_train,color="g")
    plt.scatter(x_test,y_test,color="b")
    plt.plot(x_test,y_pred,color="yellow")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    
    print(f"degree : {degree}")
    print(f"mse : {mse}")
    print(f"rmse : {rmse}")
    print(f"mae : {mae}")
    print(f"r2 : {r2}\n")


def main():
    np.random.seed(43)
    x=np.random.rand(1000,1)*9
    y=7*x**2+8*x+np.random.rand(1000,1)
    max_degree=4

    for degree in range(1,max_degree+1):
        polynomial_regression(degree,x,y)

if __name__=="__main__":
    main()

