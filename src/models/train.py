import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import lightgbm as lgm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation.preprocess import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , f1_score , recall_score ,precision_score , confusion_matrix , classification_report ,ConfusionMatrixDisplay
from src.config import Configuration
import mlflow

class training(preprocessing):

    def data_spliting(self):
        X = self.df.drop(columns=['Machine_failure','Product_ID',"TWF","HDF","PWF","OSF","RNF"])
        y = self.df['Machine_failure']
        X_train ,X_test , y_train , y_test =  train_test_split(X,y,test_size=0.20,random_state=42)

        return X_train , X_test , y_train , y_test
    
    def dividing_Feature(self):
        X_train , _ , _ , _ = self.data_spliting() 
        numerical_Features = X_train.select_dtypes(include=np.number)
        catecorcal_Features = X_train.select_dtypes(include=['object',"string"])
        return numerical_Features , catecorcal_Features

    def handle_data_for_train(self):
        categorcal_encoder = Pipeline([
            ("encoder",OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_scaling = Pipeline([
            ("scaler",StandardScaler())
        ])
        return categorcal_encoder , numerical_scaling
    
    def colPipline(self):
        num_feature, cat_feature = self.dividing_Feature()
        encoder_categorcal ,scaler_numerical = self.handle_data_for_train() 

        prepross = ColumnTransformer([
            ('cat',encoder_categorcal,cat_feature.columns),
            ('num',scaler_numerical,num_feature.columns)
        ])

        return prepross
    

    def model_training(self):
        X_train ,X_test , y_train , y_test = self.data_spliting()
        model = lgm.LGBMClassifier(class_weight='balanced',n_estimators=500,boosting_type='gbdt',learning_rate=0.05,importance_type='split')
        pipe = Pipeline([
            ("process",self.colPipline()),
            ('model',model)
        ])
        model = pipe.fit(X_train,y_train)
        return model 


    def ModelTracking(self,experiment_name,artifacts_location_saving,mlflow_tracking_URI="file:./mlruns"):

        artifacts = str(Path(artifacts_location_saving).resolve())

        mlflow.set_tracking_uri(mlflow_tracking_URI)

        try:
            mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifacts
            )
            mlflow.set_tracking_uri(mlflow_tracking_URI)

        except mlflow.exceptions.MlflowException:
            print("The Experments aleary exist")
            pass
            
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="LGBMClassifier_runing"):

            model = self.model_training()

            _ , X_test , _ , y_test = self.data_spliting()

            predict = model.predict(X_test)


            mlflow.log_params({
                "test_size": 0.20,
                "random_state":42,
                "model_type":"LGBMClassifier",
                "class_weight":"balanced",
                "n_estimators":300,
                "boosting_type":'gbdt',
                "Learning__rate":0.01,
                "importance_type":"split",
            })

            fig,_ = plt.subplots()
            matrix = confusion_matrix(y_test,predict)
            sns.heatmap(matrix,annot=True,fmt=".2f",cmap="Blues",linewidths=0.5)

            mlflow.log_figure(fig,"confusion_matrix.png")
            mlflow.log_metric("Recall_Score",recall_score(y_test,predict))
            mlflow.log_metric("precision_score",precision_score(y_test,predict))
            mlflow.log_metric("f1_score",f1_score(y_test,predict))
            mlflow.log_metric("FDR",1-precision_score(y_test,predict))

            mlflow.lightgbm.log_model(model,artifact_path="model",serialization_format="cloudpickle")

            print("Experiment logged successfully!")
            


sittings = Configuration()
train = training(sittings.cleaned_Data_dir())
model = train.model_training()
train.ModelTracking(sittings.mlflow_experiment_name,sittings.artifacts_directory(),sittings.mlflow_tracking_URI)

