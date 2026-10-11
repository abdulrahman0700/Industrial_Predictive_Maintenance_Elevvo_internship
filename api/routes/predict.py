from fastapi import APIRouter 
from src.schemas.Sensor_schema import Sensor_validation
from src.data_preparation.feature_engineering import FeatureEngineering
from src.database.connection import databaseConnection
from src.config import Configuration
import joblib
import pandas as pd

router2 = APIRouter(prefix='/predict',tags=['Predictions'])

Sittings = Configuration()

model = joblib.load(Sittings.failure_machine())

@router2.post("/")
def predicting(request : Sensor_validation):
    dataPrepare = pd.DataFrame({"Type":request.Type,
                              "Air_temperature__K":request.Air_temperature__K,
                              "Process_temperature__K":request.Process_temperature__K,
                              "Torque__Nm":request.Torque__Nm,
                              "Tool_wear__min":request.Tool_wear__min,
                              "Rotational_speed__rpm":request.Rotational_speed__rpm},index=[0])
    dataReady = FeatureEngineering(dataPrepare)
    pred = model.predict(dataReady)

    print(pred)
    