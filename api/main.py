from fastapi import FastAPI ,requests , HTTPException  , APIRouter
from fastapi.responses import HTMLResponse
from src.database.connection import databaseConnection
from src.config import Configuration
from api.routes.health import router1
from api.routes.predict import router2
from src.schemas.Sensor_schema import Sensor_validation
from src.data_preparation.feature_engineering import FeatureEngineering
import plotly.express as px
import pandas as pd

app = FastAPI(title="Predictive_Maintenance",version="1.0.0")

app.include_router(router1)
app.include_router(router2)

database = databaseConnection()

sitings = Configuration()

@app.get("/data/{id}")
def gettingIDData(id : int):
    data = database.read_specific_record_data(id)
    if data is None :
        raise HTTPException(
            status_code=404,
            detail=f"data with id {id} doesn't exist"
        )
    return {"data":data}

@app.get("/all/data")
def gettingAllData():
    all_data = database.read_record_data()
    return {"data":all_data}

@app.get("/all/Predictiondata")
def gettingPredictedRedcords():
    predicted_records = database.read_all_prediction_data()
    return {"predicted_data":predicted_records}

# @app.get("/dashboard")
# def Visulization():
#     row_data = database.read_all_data()
#     if not row_data :
#         raise HTTPException(
#             status_code=404,
#             detail="There is no existing data"
#         )
    
#     airTemp = [df.Air_temperature__K for df in row_data]
#     process_temp = [df.Process_temperature__K for df in row_data]
#     RotationalSpeed = [df.Rotational_speed__rpm for df in row_data] 
#     ToolWear = [df.Tool_wear__min for df in row_data]
#     Torque = [df.Torque__Nm for df in row_data]

#     fig_1 = px.scatter(
#                   x=ToolWear,
#                   y=Torque,
#                   title="Relationship between Tool Wear and Torque",
#                   labels={"x":"Tool Wear Min" , 'y':"Torque NM"},
#                   trendline="ols")
#     fig_2 = px.density_heatmap(x=RotationalSpeed,y=process_temp,labels={"x":"Rotation_Speed",'y':"Process_Temperature"})
    
#     html_content_num1 = fig_1.to_html(full_html=False,include_plotlyjs="cdn") 
#     html_content_num2 = fig_2.to_html(full_html=False,include_plotlyjs=False)

#     return  HTMLResponse(content=html_content_num1) 