import pickle
from pathlib import Path
from pydantic_settings import BaseSettings , SettingsConfigDict 

class Configuration(BaseSettings):

    model_config = SettingsConfigDict(env_file='.env',env_file_encoding='utf-8',protected_namespaces=('settings_',))

#------------databaseConfig-------------- 
 
    DATABASE_URL :str = "postgresql://postgres:123456789@localhost:5432/"

#--------------MLflowConfig--------------

    mlflow_tracking_URI : str = ""

    mlflow_experiment_name : str = ""

#---------------modelConfig----------------
    
    model_artifacts_dir : Path = Path("models_artifacts")
    model_failure : str = "models_artifacts/lgbm_model_failure.joblib"
    model_failure_type : str = "models_artifacts/XGBoosts_model_type_failure.joblib"

#---------------DataConfig-----------------
    data : Path = Path("data")  
    row_data : str = "Row_data/ai4i_predictive_maintenance.csv"
    cleaned_data : str = "processed/df_cleaned_v1.csv"

#--------------API Config------------------

    API_host :str = "0.0.0.0"
    API_Port :int = 8000


#-------------- logging--------------

    log_level : str = "INFO"

    
    def failure_machine(self) -> Path :
        return self.model_artifacts_dir / self.model_failure

    def failure_Machine_type(self) -> Path :
        return self.model_artifacts_dir / self.model_failure_type


    def rows_data(self) -> Path :
        return self.data / self.row_data

    def cleaned_Data(self) -> Path :
        return self.data / self.cleaned_data


    Threshold : float = 0.70


Settings = Configuration()