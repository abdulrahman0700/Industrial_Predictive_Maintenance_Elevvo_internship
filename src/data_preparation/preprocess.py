import numpy as np
import pandas as pd
from src.config import Configuration


class preprocessing():
    def __init__(self,Path):
        self.DATA_PATH =  Path
        self.df = pd.read_csv(self.DATA_PATH,index_col=0) 

    def dataSample(self):
        print("10 sample rows of the data")
        print(self.df.sample(15))

    def SummaryStatistics(self):
        print(self.df.describe())

    def dataChecking(self):

        report =pd.DataFrame(
            {
            "Null_Values":self.df.isnull().sum(),
            "average_Null_values":self.df.isnull().mean(),
            }
        )
        return report
    
    def dataInfo(self):
        print(self.df.info())
    
    def renaming(self):
        for column in self.df.columns:
            column_v = column.replace(" ","_").replace("[","_").replace("]","")
            self.df.rename(columns={column:column_v},inplace=True)
        
        return self.df

    def saving_data(self,clean_data):
        clean_data.to_csv(f"{path_settings.cleaned_Data_dir()}")

    

path_settings = Configuration()

data_path = path_settings.rows_data_dir()

preprocess = preprocessing(Path=data_path)

process_data = preprocess.renaming()

preprocess.saving_data(process_data)
