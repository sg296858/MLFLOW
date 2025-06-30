import numpy
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

wine=load_wine()

X=wine.data
y=wine.target

mlflow.set_tracking_uri("http://127.0.0.1:5000")

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)

params=[
    {
        "max_depth":[2,4,5,6,7,8,9,10,11,12],
        "n_estimators":[5,10,20,30,40,50]
    }
]

mlflow.set_experiment("Hyperparametertuning using GridSearchCV using MLFLOW")

with mlflow.start_run():
    model=RandomForestClassifier()
    gridsearch=GridSearchCV(estimator=model,cv=5,scoring='accuracy',param_grid=params)
    gridsearch.fit(X_train,Y_train)

    best_params=gridsearch.best_params_
    best_score=gridsearch.best_score_

    best_model=gridsearch.best_estimator_
    best_model.fit(X_train,Y_train)
    y_pred=best_model.predict(X_test)
    accuracy=accuracy_score(y_pred,Y_test)
    precision = precision_score(Y_test, y_pred, average='macro')
    recall = recall_score(Y_test, y_pred, average='macro')
    f1 = f1_score(Y_test, y_pred, average='macro')

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_model,"best_model")

    print(accuracy,best_params,best_score)