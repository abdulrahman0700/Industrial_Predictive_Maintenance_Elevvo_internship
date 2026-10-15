import numpy as np
import pandas as pd

from src.data_preparation.feature_engineering import FeatureEngineering
from src.data_preparation.preprocess import preprocessing
from src.models.train import training  


def make_row():
    return pd.DataFrame({
        "Torque__Nm": [40.0],
        "Rotational_speed__rpm": [1500],
        "Tool_wear__min": [100],
        "Air_temperature__K": [298.0],
        "Process_temperature__K": [308.0],
    })


def test_temperature_difference():
    df = FeatureEngineering(make_row())
    assert df["Temperature_difference"][0] == 10.0      


def test_strain():
    df = FeatureEngineering(make_row())
    assert df["strain"][0] == 4000.0                    


def test_power():
    df = FeatureEngineering(make_row())
    assert abs(df["Power_W"][0] - 6283.19) < 0.01



def test_renaming(tmp_path):
    raw = pd.DataFrame({
        "UDI": [1, 2],
        "Air temperature [K]": [298.1, 298.2],
        "Machine failure": [0, 1],
    })
    csv_path = tmp_path / "raw.csv"
    raw.to_csv(csv_path, index=False)

    df = preprocessing(csv_path).renaming()

    assert "Air_temperature__K" in df.columns
    assert "Machine_failure" in df.columns


def test_null_counting(tmp_path):
    raw = pd.DataFrame({
        "UDI": [1, 2, 3],
        "Torque [Nm]": [40.0, np.nan, np.nan],
    })
    csv_path = tmp_path / "raw.csv"
    raw.to_csv(csv_path, index=False)

    report = preprocessing(csv_path).dataChecking()

    assert report.loc["Torque [Nm]", "Null_Values"] == 2



def test_target_and_ids_are_not_in_features(tmp_path):
    data = pd.DataFrame({
        "UDI": range(1, 11),
        "Product_ID": [f"M{i}" for i in range(10)],
        "Type": ["L", "M"] * 5,
        "Air_temperature__K": [298.0] * 10,
        "Process_temperature__K": [308.0] * 10,
        "Rotational_speed__rpm": [1500] * 10,
        "Torque__Nm": [40.0] * 10,
        "Tool_wear__min": [10] * 10,
        "Machine_failure": [0, 1] * 5,
        "TWF": [0] * 10, "HDF": [0] * 10, "PWF": [0] * 10,
        "OSF": [0] * 10, "RNF": [0] * 10,
    })
    csv_path = tmp_path / "cleaned.csv"
    data.to_csv(csv_path, index=False)

    X_train, X_test, y_train, y_test = training(csv_path).data_spliting()

    for column in ["Machine_failure", "Product_ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert column not in X_train.columns
