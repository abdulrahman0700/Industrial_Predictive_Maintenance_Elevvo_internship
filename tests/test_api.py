import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from api.routes.health import router1
from src.schemas.Sensor_schema import Sensor_validation

app = FastAPI()
app.include_router(router1)
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy", "code": 200}



def valid_data():
    return {
        "Product_ID": "M14860",
        "Type": "M",
        "Air_temperature__K": 298.1,
        "Process_temperature__K": 308.6,
        "Rotational_speed__rpm": 1551,
        "Torque__Nm": 42.8,
        "Tool_wear__min": 0,
    }


def test_valid_data_is_accepted():
    Sensor_validation(**valid_data())   


def test_torque_too_high_is_rejected():
    data = valid_data()
    data["Torque__Nm"] = 100            
    with pytest.raises(ValidationError):
        Sensor_validation(**data)


def test_tool_wear_too_high_is_rejected():
    data = valid_data()
    data["Tool_wear__min"] = 251       
    with pytest.raises(ValidationError):
        Sensor_validation(**data)


def test_type_must_be_one_character():
    data = valid_data()
    data["Type"] = "MM"
    with pytest.raises(ValidationError):
        Sensor_validation(**data)


def test_missing_field_is_rejected():
    data = valid_data()
    del data["Torque__Nm"]
    with pytest.raises(ValidationError):
        Sensor_validation(**data)
