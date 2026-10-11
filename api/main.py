from fastapi import FastAPI ,requests , HTTPException 
from fastapi.responses import HTMLResponse
from src.database.connection import databaseConnection
from src.config import Configuration
from api.routes.health import checking_server
from src.schemas.Sensor_schema import Sensor_validation
from src.data_preparation.feature_engineering import FeatureEngineering
import plotly.express as px
import plotly
import pandas as pd
import joblib


app = FastAPI(title="Predictive_Maintenance",version="1.0.0")

database = databaseConnection()

sitings = Configuration()

model = joblib.load(sitings.failure_machine())

@app.get('/')
def HealthyCheck():
    return checking_server()


@app.get("/data/{id}")
def GettingData(id : int):
    data = database.read_specific_data(id)
    if data is None :
        raise HTTPException(
            status_code=404,
            detail=f"data with id {id} doesn't exist"
        )
    return {"data":data}

@app.get("/dashboard")
def Visulization():
    row_data = database.read_all_data()
    if not row_data :
        raise HTTPException(
            status_code=404,
            detail="There is no existing data"
        )
    
    airTemp = [df.Air_temperature__K for df in row_data]
    process_temp = [df.Process_temperature__K for df in row_data]
    RotationalSpeed = [df.Rotational_speed__rpm for df in row_data] 
    ToolWear = [df.Tool_wear__min for df in row_data]
    Torque = [df.Torque__Nm for df in row_data]

    fig_1 = px.scatter(
                  x=ToolWear,
                  y=Torque,
                  title="Relationship between Tool Wear and Torque",
                  labels={"x":"Tool Wear Min" , 'y':"Torque NM"},
                  trendline="ols")
    fig_2 = px.density_heatmap(x=RotationalSpeed,y=process_temp,labels={"x":"Rotation_Speed",'y':"Process_Temperature"})
    
    html_content_num1 = fig_1.to_html(full_html=False,include_plotlyjs="cdn") 
    html_content_num2 = fig_2.to_html(full_html=False,include_plotlyjs="cdn")

    return  HTMLResponse(content=html_content_num1) 

# Postman API Testing Verified
@app.post("/predict")
def Predictions(request : Sensor_validation):

    r_data = pd.DataFrame({"Type":request.Type,
                              "Air_temperature__K":request.Air_temperature__K,
                              "Process_temperature__K":request.Process_temperature__K,
                              "Rotational_speed__rpm":request.Rotational_speed__rpm,
                              "Tool_wear__min":request.Tool_wear__min,
                              "Torque__Nm":request.Torque__Nm},index=[0])
    dataReady = FeatureEngineering(r_data)
    
    prediction = model.predict(dataReady)

    print(prediction)

    # return prediction 
     
    # database.insert_data(request)

    # return{"ProductId":request.Product_ID ,
    #         "type":request.Type,
    #         "airTemp":request.Air_temperature__K,
    #         "ProecessTemp":request.Process_temperature__K  ,
    #         "RotationlSpeed":request.Rotational_speed__rpm ,
    #         "Torque":request.Torque__Nm,
    #         "ToolWear":request.Tool_wear__min,
    #         "MachineFailure":request.Machine_failure,
    #         "twf":request.TWF,
    #         "hdf":request.HDF,
    #         "pwf":request.PWF,
    #         "osf":request.OSF,
    #         "rnf":request.RNF}