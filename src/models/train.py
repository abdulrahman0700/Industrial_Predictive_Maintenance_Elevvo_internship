import numpy as np
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import lightgbm as lgm
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation.preprocess import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , f1_score , recall_score ,precision_score , confusion_matrix , roc_curve
from src.config import Configuration
import mlflow

class training(preprocessing):

    def data_spliting(self):
        X = self.df.drop(columns=['Machine_failure','Product_ID',"TWF","HDF","PWF","OSF","RNF"])
        y = self.df['Machine_failure']
        X_train ,X_test , y_train , y_test =  train_test_split(X,y,test_size=0.25,random_state=42)

        return X_train , X_test , y_train , y_test
    
    def dividing_Feature(self):
        X_train , _ , _ , _ = self.data_spliting() 
        numerical_Features = X_train.select_dtypes(include=np.number)
        catecorcal_Features = X_train.select_dtypes(include='object')
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
        model = lgm.LGBMClassifier(class_weight='balanced',n_estimators=500,boosting_type='dart',learning_rate=0.01,importance_type='split')
        pipe = Pipeline([
            ("process",self.colPipline()),
            ('model',model)
        ])
        model = pipe.fit(X_train,y_train)
        return model 

    def ModelTracking(self,experiment_name,mlflow_tracking_URI):

        mlflow.create_experiment(experiment_name)
        mlflow.set_tracking_uri(mlflow_tracking_URI)

        with mlflow.start_run(run_name="LGBMClassifier_runing"):

            mlflow.set_experiment(experiment_name)

            model = self.model_training()

            _ , X_test , _ , y_test = self.data_spliting()

            predict = model.predict(X_test)

            


    def model_saving(self,model):
        pass


sittings = Configuration()
train = training(sittings.cleaned_Data_dir())
model = train.model_training()
_ , X_test , _ , _ = train.data_spliting()
pred = model.predict(X_test)
print(pred)

