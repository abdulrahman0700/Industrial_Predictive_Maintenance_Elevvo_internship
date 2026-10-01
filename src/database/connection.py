from sqlalchemy import Column , create_engine, Integer , String , Float , Boolean
from sqlalchemy.orm import declarative_base , sessionmaker 
from src.config import Configuration
from src.database.models import AI4I_Predictive_maintance


class databaseConnection():
    def __init__(self,URL):
        self.engine = URL
        self.Base = declarative_base()
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def CreateDatabase(self):
        self.Base.metadata.create_all(self.engine)

    def insert_data(self,data):
        self.session.add(data)
        self.session.commit()

config = Configuration()
database = databaseConnection(config.DATABASE_URL)
dataCreation = AI4I_Predictive_maintance()
database.CreateDatabase()


