from sqlalchemy import create_engine , Column , String , Integer , Float , Boolean , TIMESTAMP , ForeignKey
from sqlalchemy.orm import declarative_base , relationship
import datetime

Base = declarative_base()

class AI4I_Predictive_maintance(Base):  

    __tablename__ = "AI4I"

    UDI = Column(Integer,primary_key=True,autoincrement=True)
    Product_ID = Column(String,nullable=False)
    Type = Column(String(1),nullable=False)
    Air_temperature__K = Column(Float,nullable=False)
    Process_temperature__K = Column(Float,nullable=False) 
    Rotational_speed__rpm = Column(Integer,nullable=False)
    Torque__Nm = Column(Float,nullable=False)
    Tool_wear__min = Column(Integer,nullable=False)

    related_data = relationship("PredictionsRecords",back_populates='related_record')


class PredictionsRecords(Base):

    __tablename__ = "PredctionTable"

    id = Column(Integer,primary_key=True)
    failure_probability = Column(Float,nullable=False)
    failure_predicted = Column(Boolean,nullable=False)
    predictionTime = Column(TIMESTAMP,default=datetime.date) 
    TWF=Column(Boolean,nullable=False)
    HDF=Column(Boolean,nullable=False)
    PWF=Column(Boolean,nullable=False)
    OSF=Column(Boolean,nullable=False)
    RNF=Column(Boolean,nullable=False)
    predictedMachine = Column(Integer,ForeignKey("AI4I.UDI"))

    related_record = relationship("AI4I_Predictive_maintance",back_populates='related_data')


