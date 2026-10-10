from fastapi import FastAPI ,requests , HTTPException 
from src.database.connection import databaseConnection
from src.config import Configuration
from api.routes.health import checking_server
from src.schemas.Sensor_schema import Sensor_validation
import plotly.express as px
import pandas as pd


app = FastAPI(title="Predictive_Maintenance",version="1.0.0")

database = databaseConnection()

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
    airTemp = [df.Air_temperature__K for df in row_data]
    print(df)
    # if not row_data :
    #     raise HTTPException(
    #         status_code=404,
    #         details="data does not exist"
    #     )
    # dataframe = pd.DataFrame(row_data)
    # print(f"the dataframe{dataframe.head()}")
    # numerical_feature = dataframe.select_dtypes(exclude='number')
    # print(f"the numbers {numerical_feature}")
    
    # d = sns.histplot(numerical_feature)
    # plt.show()
    # return d

# Postman API Testing Verified
@app.post("/predict")
def Predictions(request : Sensor_validation):
     
     database.insert_data(request)

     return{"ProductId":request.Product_ID ,
            "type":request.Type,
            "airTemp":request.Air_temperature__K,
            "ProecessTemp":request.Process_temperature__K  ,
            "RotationlSpeed":request.Rotational_speed__rpm ,
            "Torque":request.Torque__Nm,
            "ToolWear":request.Tool_wear__min,
            "MachineFailure":request.Machine_failure,
            "twf":request.TWF,
            "hdf":request.HDF,
            "pwf":request.PWF,
            "osf":request.OSF,
            "rnf":request.RNF}