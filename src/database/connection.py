from sqlalchemy import Column , create_engine, Integer , String , Float , Boolean , select,update,delete
from sqlalchemy.orm import declarative_base , sessionmaker 
from src.config import Configuration
from src.database.models import AI4I_Predictive_maintance , Base
from src.schemas.Sensor_schema import Sensor_validation

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

    def read_all_data(self):
        all_data = select(AI4I_Predictive_maintance)
        fetched_data = self.session.scalars(all_data).all()
        return fetched_data

    def read_specific_data(self,id):
        try:
            Specific_row = self.session.get(AI4I_Predictive_maintance,id)
        except Exception as e :
            raise Exception(f"An error occurred: {e}")
        except ValueError :
            raise "Invalid input Error"
        else:
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

# database = databaseConnection()
# database.CreateDatabase()

# database.insert_data(Sensor_validation(**row_input))