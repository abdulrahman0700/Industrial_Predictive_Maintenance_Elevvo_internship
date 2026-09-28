import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import recall_score , f1_score , precision_score , confusion_matrix
from src.config import Configuration
from predict import Prediction

  
def metrices(self,true_values,model,data):
    predict = self.predicting(model,data)
    recall = recall_score(true_values,predict)
    f1Score = f1_score(true_values,predict)
    precisionScore = precision_score(true_values,predict)
    FDR = 1 - precisionScore
    return recall , f1Score , precisionScore , FDR

sitings = Configuration()
predict = Prediction(sitings.failure_machine())

predict.predicting