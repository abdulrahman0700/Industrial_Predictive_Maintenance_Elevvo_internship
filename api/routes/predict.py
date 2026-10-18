from fastapi import APIRouter 
from src.schemas.Sensor_schema import Sensor_validation ,Predcation_Validation
from src.data_preparation.feature_engineering import FeatureEngineering
from src.database.connection import databaseConnection
from api.routes.explain import llmExplaination
from src.config import Configuration
import joblib
import numpy as np
import pandas as pd

router2 = APIRouter(prefix='/predict',tags=['Predictions'])

Sittings = Configuration()
database = databaseConnection()
MF_model = joblib.load(Sittings.failure_machine())
MFT_model = joblib.load(Sittings.failure_Machine_type())

@router2.post("/")
def predicting(request : Sensor_validation):
    dataPrepare = pd.DataFrame({"Type":[request.Type],
                              "Air_temperature__K":[request.Air_temperature__K],
                              "Process_temperature__K":[request.Process_temperature__K],
                              "Torque__Nm":[request.Torque__Nm],
                              "Tool_wear__min":[request.Tool_wear__min],
                              "Rotational_speed__rpm":[request.Rotational_speed__rpm]},index=[0])
    
    machine_id = database.insert_data(request)

    dataReady = FeatureEngineering(dataPrepare)

    MFpred = MF_model.predict(dataReady)

    probMFPred = MF_model.predict_proba(dataReady)

    print(max(probMFPred.tolist()[0]))

    prediction_value = int(MFpred[0]) if hasattr(MFpred, "__len__") else int(MFpred)

    print(prediction_value)

    if prediction_value == 1 :
        data_json = {"Type":[request.Type],
                    "Air_temperature__K":[request.Air_temperature__K],
                    "Process_temperature__K":[request.Process_temperature__K],
                    "Torque__Nm":[request.Torque__Nm],
                    "Tool_wear__min":[request.Tool_wear__min],
                    "Rotational_speed__rpm":[request.Rotational_speed__rpm],
                    "machine_Failure":[prediction_value]}
        
        type_df = pd.DataFrame(data_json,index=[0])
        
        type_df_ready = FeatureEngineering(type_df)

        typeMF_Predction = MFT_model.predict(type_df_ready)
        
        typeMF_Predction = typeMF_Predction.tolist()[0]

        predicted_type_json ={"failure_predicted":prediction_value,
                              "failure_probability":max(probMFPred.tolist()[0]),
                              "TWF":typeMF_Predction[0],
                              "HDF":typeMF_Predction[1],
                              "PWF":typeMF_Predction[2],
                              "OSF":typeMF_Predction[3],
                              "RNF":typeMF_Predction[4]}

        database.insert_Predction_data(Predcation_Validation(
            failure_predicted=prediction_value,
            failure_probability=max(probMFPred.tolist()[0]),
            TWF=typeMF_Predction[0],
            HDF=typeMF_Predction[1],
            PWF=typeMF_Predction[2],
            OSF=typeMF_Predction[3],
            RNF=typeMF_Predction[4],
            predictedMachine=machine_id.UDI
        ))

        llmsexplain = llmExplaination(data_json,predicted_type_json)

        print(llmsexplain)

        return {"Status":200,
                "Tool Wear Failures":typeMF_Predction[0],
                "Heat Disspation Failures":typeMF_Predction[1],
                "Power Failures":typeMF_Predction[2],
                "Overstrain Failures":typeMF_Predction[3],
                "Random Failures":typeMF_Predction[4],
                "LLM_Explaination_Failure":llmsexplain}

    else:
        database.insert_Predction_data(Predcation_Validation(
        failure_predicted=prediction_value,
        failure_probability=max(probMFPred.tolist()[0]),
        TWF=0,
        HDF=0,
        PWF=0,
        OSF=0,
        RNF=0,
        predictedMachine=machine_id.UDI))
        
        return {"Status":200,
                "MF_Predction":prediction_value}