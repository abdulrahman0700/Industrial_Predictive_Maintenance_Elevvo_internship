import numpy as np
import pandas as pd
from src.config import Configuration


class preprocessing():
    def __init__(self,Path):
        self.DATA_PATH =  Path
        self.df = pd.read_csv(self.DATA_PATH) 

    def intial_data_exploration(self):
        print("The First 10 rows of the data")
        print(self.df.head(10))
        print("Summary Statisical :- ")
        print(self.df.describe().T)
        print("Information about the data")
        print(self.df.info())

    def dataChecking(self):

        report =pd.DataFrame(
            {
            "Null_Values":self.df.isnull().sum(),
            "average_Null_values":self.df.isnull().mean(),
            })
        return report


path_settings = Configuration()

path = path_settings.rows_data()

preprocess = preprocessing(Path=path)

report = preprocess.dataChecking()

data_explore = preprocess.intial_data_exploration()
print(data_explore)
print(report)
