

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
# from google.colab import files
import warnings
warnings.filterwarnings('ignore')
from time import time

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    SGDRegressor,
    SGDClassifier,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV
)
from sklearn.svm import (
    SVR,
    SVC
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.naive_bayes import(
    GaussianNB,
    MultinomialNB,
    BernoulliNB
)
from sklearn import tree
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    BaggingClassifier,
    BaggingRegressor,
    VotingClassifier,
    VotingRegressor
)

# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostClassifier, CatBoostRegressor

# from sklearn.discriminant_analysis import (
#     LinearDiscriminantAnalysis,
#     QuadraticDiscriminantAnalysis
# )

from sklearn.neural_network import (
    MLPClassifier,
    MLPRegressor
)


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def input_data_file(file_name):
  df=pd.read_csv(file_name)
  return df

def data_cleaning_file(df):
  num_features=df.select_dtypes(exclude="O").columns
  cat_features=df.select_dtypes(include="O").columns

  for column in df.columns:
    if column in cat_features:
      df[column]=df[column].fillna(df[column].mode()[0])
    else:
      df[column]=df[column].fillna(df[column].mean())

  return df

def data_transformation_file(df,output,num_scaler=StandardScaler(),cat_scaler=OneHotEncoder(drop="first",handle_unknown="ignore")):
  num_features=df.select_dtypes(exclude="O").columns
  cat_features=df.select_dtypes(include="O").columns

  num_features=[num for num in num_features if num!=output]

  x=df.drop([output],axis=1)
  y=pd.DataFrame(df[output],columns=[output])
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

  scaler=ColumnTransformer([
      ("num_feature_scaler",num_scaler,num_features),
      ("cat_feature_scaler",cat_scaler,cat_features)
  ])

  scaler.fit_transform(x_train)
  x_train_scaled=scaler.transform(x_train)
  x_test_scaled=scaler.transform(x_test)

  return x_train_scaled,x_test_scaled,y_train,y_test

def data_pca_transformation_file(x_train,x_test,pca_n_components=2):
  pca=PCA(n_components=pca_n_components)
  pca.fit(x_train)
  x_train_scaled=pca.transform(x_train)
  x_test_scaled=pca.transform(x_test)

  return x_train_scaled,x_test_scaled

def model_tuner_file(model_name,params,x_train,y_train,tuner=RandomizedSearchCV):
  tuner_model=tuner(
      estimator=model_name(),
      param_distributions=params,
      scoring="accuracy",
      cv=5,
      n_jobs=-1,
      verbose=2
  )

  tuner_model.fit(x_train,y_train)

  return tuner_model.best_estimator_,tuner_model.best_score_

def test_model_file(model,x_test,y_test,regression=True):
  y_pred=model.predict(x_test)

  if regression:
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((y_test-y_pred)/y_test))*100
    print(f"R2 Score: {r2}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

  else:
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    roc_auc=roc_auc_score(y_test,y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")

models = [
    (
        "RandomForestClassifier",
        RandomForestClassifier,
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False]
        }
    ),
    (
        "GradientBoostingClassifier",
        GradientBoostingClassifier,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.6, 0.8, 1.0],
            "max_features": ["auto", "sqrt", "log2", None]
        }
    ),
    (
        "AdaBoostClassifier",
        AdaBoostClassifier,
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        }
    ),
    (
        "ExtraTreesClassifier",
        ExtraTreesClassifier,
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False]
        }
    ),
    (
        "HistGradientBoostingClassifier",
        HistGradientBoostingClassifier,
        {
            "max_iter": [100, 200, 300],
            "max_leaf_nodes": [31, 63, 127],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.1, 1.0],
            "early_stopping": [True, False]
        }
    ),
    (
        "BaggingClassifier",
        BaggingClassifier,
        {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.7, 1.0],
            "max_features": [0.5, 0.7, 1.0],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False]
        }
    )
]


def select_tuner_length_file():
  return len(models)

def select_tuner_file(n):
  if len(models)>n:
    return models[n]
  else:
    return models[0]

def print_models(models_config):
  for model_name,model,score in models_config:
    print(f"model : {model}")
    print(f"model_name : {model_name}")
    print(f"score : {score}")
    print()


def print_best_models(best_models_config):
  print(f"best models : \n")
  for pca,model,score in best_models_config:
    print(f"pca : {pca}")
    print(f"model : {model}")
    print(f"score : {score}")
    print()




def main():
  preprocessing_start=time()
  data=load_breast_cancer()
  df=pd.DataFrame(data["data"],columns=data["feature_names"])
  df["target"]=data["target"]
  output="target"

  data_cleaning_file(df)

  x_train,x_test,y_train,y_test=data_transformation_file(df,output)

  preprocessing_end=time()

  tuner=RandomizedSearchCV

  models_config=[]
  best_models_config=[]

  n=df.shape[-1]-1


  processing_start=time()
  for i in range(2,n,2):

    pca_n_components=i
    x_train_scaled,x_test_scaled=data_pca_transformation_file(x_train,x_test,pca_n_components=pca_n_components)
    best_model=""
    best_score=0
    print(f"pca : {pca_n_components}")

    for j in range(n):
      model_name_name,model_name,params=select_tuner_file(j)
      print(f"executing model : {model_name_name}")

      model,score=model_tuner_file(model_name,params,x_train_scaled,y_train,tuner)
      models_config.append((model_name,model,score))

      if score > best_score:
        best_score=score
        best_model=model

    best_models_config.append((pca_n_components,best_model,best_score))

  processing_end=time()

  return preprocessing_end-preprocessing_start,processing_end-processing_start,models_config,best_models_config


if __name__=="__main__":
  preprocessing_time,processing_time,models_config,best_models_config=main()

  # print(f"preprocessing time : {preprocessing_time}")
  # print(f"processing time : {processing_time}")

  # print_models(models_config)
  # print_best_models(best_models_config)

  file_mode="w"
  with open("process_time.txt", file_mode) as file:
      file.write(f"preprocessing time : {preprocessing_time}\n")
      file.write(f"processing time : {processing_time}\n")

  with open("models_config.txt", file_mode) as file:
      for name, model, score in models_config:
          file.write(f"Model: {name}\n Score: {score:.4f} \n Estimator: {model}\n\n")
          
  with open("best_models_config.txt", file_mode) as file:
      for pca, model, score in best_models_config:
          file.write(f"PCA: {pca} \n Best Score: {score:.4f} \n Best Model: {model}\n\n")

