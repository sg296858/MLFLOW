import numpy as np
import optuna
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
from sklearn.model_selection import  cross_val_score

wine=load_wine()

X=wine.data
y=wine.target

mlflow.set_tracking_uri("http://127.0.0.1:5000")

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)

mlflow.set_experiment("Hyperparameter tuning using Optuna for MLFLOW")

def objective(trial):
    n_estimators=trial.suggest_int('n_estimators',5,100)
    max_depth=trial.suggest_int('max_depth',5,50)
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,random_state=31)
    score=cross_val_score(model,X_train,Y_train,cv=5,scoring='accuracy').mean()
    return score


with mlflow.start_run():
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    study.optimize(objective,n_trials=20)

    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_score", study.best_value)
    

    best_model=RandomForestClassifier(**study.best_params,random_state=11)
    best_model.fit(X_train,Y_train)
    y_pred=best_model.predict(X_test)
    accuracy=accuracy_score(y_pred,Y_test)
    mlflow.log_metric("accuracy",accuracy)

    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(best_model,"RFmodel")

    print(accuracy)
    print(study.best_params)
    print(study.best_trials)
    print(study.best_trial.value)


    



