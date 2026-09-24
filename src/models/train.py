import numpy as np
from sklearn.preprocessing import StandardScaler , OneHotEncoder
import lightgbm as lgm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation.preprocess import preprocessing
from sklearn.model_selection import train_test_split 
from src.config import Configuration

class training(preprocessing):

    def trainingPrepare(self):
        self.df.drop(columns="Product_ID",inplace=True)
        return self.df

    def dividing_Feature(self):
        numerical_Features = self.df.select_dtypes(include=np.number)
        catecorcal_Features = self.df.select_dtypes(include='object')
        return numerical_Features , catecorcal_Features
    
    def handle_data_for_train(self):
        categorcal_encoder = Pipeline([
            ("Scaler",OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_scaling = Pipeline([
            ("scaler",StandardScaler())
        ])

sittings = Configuration()
d = training(sittings.cleaned_Data_dir())
d.trainingPrepare()

