import numpy as np
import pandas as pd
from .preprocess import preprocessing
from src.config import Configuration

def FeatureEngineering(df):
    df['Power_W'] = round(df['Torque__Nm'] * (df['Rotational_speed__rpm'] * (2 * np.pi / 60)),2)
    df['strain'] = df['Tool_wear__min'] * df['Torque__Nm']
    df['Temperature_difference'] = df['Process_temperature__K'] - df['Air_temperature__K']

    return df

sittings = Configuration()
cleanedData = sittings.cleaned_Data_dir()
preprocess = preprocessing(cleanedData)
cleaned_data = FeatureEngineering(preprocess.df)
preprocess.saving_data(cleaned_data)
print(preprocess.df)




        