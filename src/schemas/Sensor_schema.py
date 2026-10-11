from pydantic import Field , BaseModel 
from datetime import datetime

class Sensor_validation(BaseModel):
    Product_ID : str = Field(...,min_length=6,max_length=8 ,validation_alias="Product_ID") 
    Type : str = Field(...,min_length=1,max_length=1,validation_alias="Type") 
    # PredctionTime : datetime
    Air_temperature__K : float = Field(...,ge=288.0,le=312.0, validation_alias="Air_temperature__K")
    Process_temperature__K : float = Field(...,ge=299.0,le=322.0,validation_alias="Process_temperature__K")
    Rotational_speed__rpm : int = Field(...,ge=885,le=2175,validation_alias="Rotational_speed__rpm")
    Torque__Nm : float = Field(...,ge=2.5,le=77.5,validation_alias="Torque__Nm")
    Tool_wear__min : int = Field(...,ge=0 , le=250,validation_alias="Tool_wear__min")
    failure_probability : float
    failure_predicted : bool 
    TWF:bool= Field(...,validation_alias='TWF',description='Tool Wear Failures')
    HDF:bool= Field(...,validation_alias='HDF',description="Heat Disspation Failures")
    PWF:bool= Field(...,validation_alias='PWF',description="Power Failures")
    OSF:bool= Field(...,validation_alias='OSF',description="Overstrain Failures")
    RNF:bool= Field(...,validation_alias='RNF',description="Random Failures")


