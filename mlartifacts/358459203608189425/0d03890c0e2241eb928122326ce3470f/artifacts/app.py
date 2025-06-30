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

mlflow.set_tracking_uri("http://127.0.0.1:5000")

X=wine.data
y=wine.target

#Train-test-split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#set hyperparameters
max_depth=8
n_estimators=50

#set experiment name
mlflow.set_experiment("MLOps Experiment 1")

with mlflow.start_run():
    model=RandomForestClassifier(max_depth=8,n_estimators=50,random_state=12)
    model.fit(X_train,Y_train)
    y_pred=model.predict(X_test)
    accuracy=accuracy_score(y_pred,Y_test)
    
    
    
    #logging the parameters
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("n_estimnators",n_estimators)
    
    #Logging the metrics
    mlflow.log_metric("accuracy",accuracy)
   
    
    

    #Logging the artifcats confusion matrix
    cm=confusion_matrix(y_pred,Y_test)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Predicted value")
    plt.ylabel("Actual value")
    plt.title('Confusion Matrix')
    plt.savefig("Confusion matrix.png")

    #Logging artifacts app.py file and confusion matrix
    mlflow.log_artifact("Confusion matrix.png")
    mlflow.log_artifact(__file__)

    #Adding tags
    mlflow.set_tags({"Author":"Shubham","Project":"Wine classification"})

    #logging the model
    mlflow.sklearn.log_model("rf",model)

    print(accuracy)


