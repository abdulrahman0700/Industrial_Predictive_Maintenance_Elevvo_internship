import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import recall_score , f1_score , precision_score , confusion_matrix
from src.config import Configuration
from src.data_preparation.preprocess import preprocessing
from src.models.predict import Prediction

  
def metrices(true_values,model,data):
    predict = model.predict(data)
    recall = recall_score(true_values,predict)
    f1Score = f1_score(true_values,predict)
    precisionScore = precision_score(true_values,predict)
    FDR = 1 - precisionScore
    metrices_df = pd.DataFrame({
        "recall":recall,
        "f1_score":f1Score,
        "precision":precisionScore,
        "False_Dicovery_Rate":FDR
    },index=[0])

    sns.heatmap(metrices_df,annot=True)
    plt.show()

    return recall , f1Score , precisionScore , FDR

# sitings = Configuration()
# predicts = Prediction(sitings.testing_unseen_data())
# process = preprocessing(sitings.testing_unseen_data())
# df = process.df
# r , f ,p ,f = metrices(df['Machine_failure'],predicts.TheModel(sitings.failure_machine()),df.drop(columns=['Machine_failure','Product_ID','TWF', 'HDF', 'PWF', 'OSF', 'RNF']))
# print(r,f,p,f)