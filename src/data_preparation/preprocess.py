import numpy as np
import pandas as pd
from src.config import Configuration


class preprocessing():
    def __init__(self,Path):
        self.DATA_PATH =  Path
        self.df = pd.read_csv(self.DATA_PATH) 

    def dataSample(self):
        print("10 sample rows of the data")
        print(self.df.sample(15))

    def SummaryStatistical(self):
        print(self.df.describe())

    def dataChecking(self):

        report =pd.DataFrame(
            {
            "Null_Values":self.df.isnull().sum(),
            "average_Null_values":self.df.isnull().mean(),
            }
        )
        return report
    
    def renaming(self):
        for column in self.df.columns:
            column_v = column.replace(" ","_").replace("[","_").replace("]","")
            self.df.rename(columns={column:column_v},inplace=True)
        print(self.df)

path_settings = Configuration()

path = path_settings.rows_data()

preprocess = preprocessing(Path=path)

preprocess.renaming()


report = preprocess.dataChecking()

# preprocess.dataSample()
# preprocess.SummaryStatistical()
# print(report)
