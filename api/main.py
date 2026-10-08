from fastapi import FastAPI ,requests , HTTPException 
from src.database.connection import databaseConnection
from src.config import Configuration
from api.routes.health import checking_server
from src.schemas.Sensor_schema import Sensor_validation

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

# Postman API Testing Verified
@app.post("/predict")
def Predictions(request : Sensor_validation):
     Product_ID = request.Product_ID 
     Type = request.Type
     Air_temperature__K=request.Air_temperature__K 
     Process_temperature__K=request.Process_temperature__K  
     Rotational_speed__rpm=request.Rotational_speed__rpm 
     Torque__Nm=request.Torque__Nm
     Tool_wear__min=request.Tool_wear__min
     Machine_failure=request.Machine_failure 
     TWF=request.TWF
     HDF=request.HDF
     PWF=request.PWF
     OSF=request.OSF
     RNF=request.RNF

     database.insert_data(request)

     return{"ProductId":Product_ID,
            "type":Type,
            "airTemp":Air_temperature__K,
            "ProecessTemp":Process_temperature__K,
            "RotationlSpeed":Rotational_speed__rpm,
            "Torque":Torque__Nm,
            "ToolWear":Tool_wear__min,
            "MachineFailure":Machine_failure,
            "twf":TWF,
            "hdf":HDF,
            "pwf":PWF,
            "osf":OSF,
            "rnf":RNF}