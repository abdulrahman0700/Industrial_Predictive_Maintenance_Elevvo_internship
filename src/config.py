import pickle
from pathlib import Path
from pydantic_settings import BaseSettings , SettingsConfigDict 

class Configuration(BaseSettings):

    model_config = SettingsConfigDict(env_file='.env',env_file_encoding='utf-8',protected_namespaces=('settings_',))

#------------databaseConfig-------------- 
 
    DATABASE_URL :str = "postgresql://postgres:123456789@localhost:5432/postgres"

#--------------MLflowConfig--------------

    mlflow_tracking_URI : str = "sqlite:///mlflow.db" # the database db that will be saving the artifacts

    mlflow_experiment_name : str = "Machine_Failure_Prediction"

#---------------modelConfig----------------
    
    Mlflow_model_artifacts_dir : Path = Path("models_artifacts")
    # model_failure : str = "lgbm_model_failure.joblib"
    model_failure : str = "model.pkl"
    model_failure_type : str = "XGBoosts_model_type_failure.joblib"
    mlflow_info : str = "mlflow_info/" 

#---------------DataConfig-----------------
    data : Path = Path("data")  
    row_data : str = "Row_data/ai4i_predictive_maintenance.csv"
    cleaned_data : str = "processed/df_cleaned_v1.csv"
    sample_testing : str = "Sample/ai4i2020_2000_high_failure.csv" # Generated Data with the same distrubution of original data and fixing the imbalamce problem

#--------------API Config------------------

    API_host :str = "0.0.0.0"
    API_Port :int = 8000


#-------------- logging--------------

    log_level : str = "INFO"

    def artifacts_directory(self) -> Path:
        return self.Mlflow_model_artifacts_dir / self.mlflow_info
    
    def failure_machine(self) -> Path :
        return self.Mlflow_model_artifacts_dir / self.model_failure

    def failure_Machine_type(self) -> Path :
        return self.Mlflow_model_artifacts_dir / self.model_failure_type

    def rows_data_dir(self) -> Path :
        return self.data / self.row_data

    def cleaned_Data_dir(self) -> Path :
        return self.data / self.cleaned_data

    def testing_unseen_data(self) -> Path :
        return self.data / self.sample_testing


    Threshold : float = 0.70

