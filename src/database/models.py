from sqlalchemy import create_engine , Column , String , Integer , Float , Boolean
from sqlalchemy.orm import declarative_base 

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
    Machine_failure = Column(Boolean,nullable=False)
    TWF=Column(Boolean,nullable=False)
    HDF=Column(Boolean,nullable=False)
    PWF=Column(Boolean,nullable=False)
    OSF=Column(Boolean,nullable=False)
    RNF=Column(Boolean,nullable=False)
