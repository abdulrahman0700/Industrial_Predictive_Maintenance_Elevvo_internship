from sqlalchemy import Column , create_engine, Integer , String , Float , Boolean , select,update,delete
from sqlalchemy.orm import declarative_base , sessionmaker 
from src.config import Configuration
from src.database.models import AI4I_Predictive_maintance , Base , PredictionsRecords
from src.schemas.Sensor_schema import Sensor_validation , Predcation_Validation

config = Configuration()

class databaseConnection():
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
        self.Base = Base
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def CreateDatabase(self):
        self.Base.metadata.create_all(self.engine)

    def insert_data(self,Pydandic_validation : Sensor_validation):
        data = Pydandic_validation.model_dump()
        records = AI4I_Predictive_maintance(**data)
        self.session.add(records)
        self.session.commit()
        return records

    def insert_Predction_data(self,Pydantic_predction : PredictionsRecords):
        records = Pydantic_predction.model_dump()
        d_records = PredictionsRecords(**records)
        self.session.add(d_records)
        self.session.commit()

    def read_all_prediction_data(self):
        predRecord = select(PredictionsRecords)
        dataFetching = self.session.scalars(predRecord).all()
        return dataFetching

    def read_specific_Predction_data(self,id):
        Sdata = self.session.get(PredictionsRecords,id)
        return Sdata

    def read_record_data(self):
        all_data = select(AI4I_Predictive_maintance)
        fetched_data = self.session.scalars(all_data).all()
        return fetched_data

    def read_specific_record_data(self,id):
        Specific_row = self.session.get(AI4I_Predictive_maintance,id)
        return Specific_row

# row_input = {
#     "Product_ID":"M37684",
#     "Type":"M",
#     "Air_temperature__K":305.0,
#     "Process_temperature__K":314.1,
#     "Rotational_speed__rpm":1358,
#     "Torque__Nm":36.3,
#     "Tool_wear__min":67,
#     "Machine_failure":0,
#     "TWF":0,
#     "HDF":0,
#     "PWF":0,
#     "OSF":0,
#     "RNF":0}

database = databaseConnection()
database.CreateDatabase()

# database.insert_data(Sensor_validation(**row_input))