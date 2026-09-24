from src.config import Configuration
import numpy as np
import pandas as pd


class FeatureEngineering():
    def __init__(self,Path):
        self.DATA_PATH = Path
        self.df = pd.read_csv(self.DATA_PATH)

    def CreateFeatures(self):
        pass


Path_Sitings = Configuration()

FeatureEngineering(Path_Sitings.cleaned_Data)
        