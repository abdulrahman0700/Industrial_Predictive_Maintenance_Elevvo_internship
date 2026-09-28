import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import recall_score , f1_score , precision_score
from src.config import Configuration
from src.data_preparation.preprocess import preprocessing
from src.data_preparation.feature_engineering import FeatureEngineering

class Prediction(preprocessing):
   
   def predicting(self,model,data):
      result = model.predict(data)
      return result

   def predicting_proba(self,model,data):
      proba_result = model.predict_proba(data)
      return proba_result
   
      
sittings = Configuration()

pred = Prediction(sittings.testing_unseen_data())

model = joblib.load(sittings.failure_machine())

df_sample_unseen = pred.renaming()

df_sample_unseen = FeatureEngineering(df_sample_unseen)

pred.saving_data(df_sample_unseen,sittings.testing_unseen_data())

df_unseen = df_sample_unseen.drop(columns=['Machine_failure','Product_ID','TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

predict = pred.predicting(model,df_unseen)
proba_result = pred.predicting_proba(model,df_unseen)

print(predict ,proba_result)


# recall ,f1score ,precision,False_Dicovery_Rate = pred.metrices(df_sample_unseen['Machine_failure'],model,df_pred)

# print(recall,f1score,precision,False_Dicovery_Rate)

# print(proba_result)