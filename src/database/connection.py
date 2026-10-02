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

    def insert_data(self,data):
        self.session.add(data)
        self.session.commit()

database = databaseConnection()
database.CreateDatabase()


