import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import lightgbm as lgm
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation.preprocess import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , f1_score , recall_score ,precision_score , confusion_matrix , classification_report ,ConfusionMatrixDisplay
from src.config import Configuration
from src.data_preparation.feature_engineering import FeatureEngineering
import mlflow

class training(preprocessing):

    def FEngineering(self):
        df = FeatureEngineering(self.df)
        return df

    def data_spliting(self):
        X = self.df.drop(columns=['Product_ID',"Machine_failure","TWF","HDF","PWF","OSF","RNF"])
        y = self.df[["TWF","HDF","PWF","OSF","RNF"]]
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
        model = MultiOutputClassifier(XGBClassifier(class_weight='balanced'))
        pipe = Pipeline([
            ("process",self.colPipline()),
            ('model',model)
        ])
        model = pipe.fit(X_train,y_train)
        return model 


    def ModelTracking(self,artifacts_location_saving,mlflow_tracking_URI="file:./mlruns",experiment_name="XGBClassifier_experiment"):

        # artifacts = str(Path(artifacts_location_saving).resolve())

        # mlflow.set_tracking_uri(mlflow_tracking_URI)

        # try:
        #     mlflow.create_experiment(
        #         name=experiment_name,
        #         artifact_location=artifacts
        #     )
        #     mlflow.set_tracking_uri(mlflow_tracking_URI)

        # except mlflow.exceptions.MlflowException:
        #     print("The Experments aleary exist")
        #     pass
            
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="XGBClassifier_runing"):

            model = self.model_training()

            _ , X_test , _ , y_test = self.data_spliting()

            predict = model.predict(X_test)

            mlflow.log_params({
                "test_size": 0.20,
                "random_state":42,
                "model_type":"XGBClassifier",
                "class_weight":"balanced",
            })

            # fig,_ = plt.subplots()
            # matrix = confusion_matrix(y_test,predict)
            # sns.heatmap(matrix,annot=True,fmt=".2f",cmap="Blues",linewidths=0.5)

            # mlflow.log_figure(fig,"confusion_matrix.png")
            mlflow.log_metric("Recall_Score",recall_score(y_test,predict,average="weighted",zero_division=0))
            mlflow.log_metric("precision_score",precision_score(y_test,predict,average="weighted",zero_division=0))
            mlflow.log_metric("f1_score",f1_score(y_test,predict,average="weighted",zero_division=0))
            mlflow.log_metric("FDR",1-precision_score(y_test,predict,average="weighted",zero_division=0))

            mlflow.sklearn.log_model(model,artifact_path="model_MFT",serialization_format="cloudpickle")

            print("Experiment logged successfully!")
            


sittings = Configuration()
train = training(sittings.cleaned_Data_dir())
df = train.FEngineering()
print(df)
model = train.model_training()
train.ModelTracking(sittings.mlflow_Type_experiment_name,sittings.artifacts_directory(),sittings.mlflow_tracking_URI)

