from fastapi import APIRouter 
from src.schemas.Sensor_schema import Sensor_validation ,Predcation_Validation
from src.data_preparation.feature_engineering import FeatureEngineering
from src.database.connection import databaseConnection
from src.config import Configuration
import joblib
import pandas as pd

router2 = APIRouter(prefix='/predict',tags=['Predictions'])

Sittings = Configuration()
database = databaseConnection()
model = joblib.load(Sittings.failure_machine())

@router2.post("/")
def predicting(request : Sensor_validation):
    dataPrepare = pd.DataFrame({"Type":[request.Type],
                              "Air_temperature__K":[request.Air_temperature__K],
                              "Process_temperature__K":[request.Process_temperature__K],
                              "Torque__Nm":[request.Torque__Nm],
                              "Tool_wear__min":[request.Tool_wear__min],
                              "Rotational_speed__rpm":[request.Rotational_speed__rpm]},index=[0])
    
    database.insert_data(request)

    dataReady = FeatureEngineering(dataPrepare)

    pred = model.predict(dataReady)

    prediction_value = int(pred[0]) if hasattr(pred, "__len__") else int(pred)

    print(prediction_value)
    
    return {"Status":200,
            "Predction":prediction_value}