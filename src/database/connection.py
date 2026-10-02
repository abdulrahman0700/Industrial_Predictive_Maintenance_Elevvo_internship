from sqlalchemy import Column , create_engine, Integer , String , Float , Boolean
from sqlalchemy.orm import declarative_base , sessionmaker 
from src.config import Configuration
from src.database.models import AI4I_Predictive_maintance , Base

config = Configuration()

class databaseConnection():
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
        self.Base = Base
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def CreateDatabase(self):
        self.Base.metadata.create_all(self.engine)

    def insert_data(self,**data):
        records = AI4I_Predictive_maintance(**data)
        self.session.add(records)
        self.session.commit()

database = databaseConnection()
database.CreateDatabase()
database.insert_data(UDI=1,
    Product_ID="L73864",
    Type="L",
    Air_temperature__K=298.30,
    Process_temperature__K=307.748,
    Rotational_speed__rpm=1776,
    Torque__Nm=26.63,
    Tool_wear__min=189,
    Machine_failure=0,
    TWF=0,
    HDF=0,
    PWF=0,
    OSF=0,
    RNF=0)