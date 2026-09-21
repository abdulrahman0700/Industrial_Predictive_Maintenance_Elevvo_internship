import psycopg2
import pickle


DB_NAME = ""
DB_PASS = ""
DB_HOST = ""
DB_PORT = ""
DB_USER = ""


DATA_PATH = "D:\Python\Industrial_Predictive_Maintenance_Elevvo_internship\data\processed\df_cleaned.csv"

MODEL_V1_PATH = "D:\Python\Industrial_Predictive_Maintenance_Elevvo_internship\models_artifacts\lgbm_model_failure.joblib"
MODEL_V2_PATH = "D:\Python\Industrial_Predictive_Maintenance_Elevvo_internship\models_artifacts\XGBoosts_model_type_failure.joblib"

class dataBase_configuration():
    def __init__(self,DB_NAME,DB_USER,DB_PASS,DB_HOST,DB_PORT):
        try:
            self.connect = psycopg2.connect(database=DB_NAME,
                                            user=DB_USER,
                                            password=DB_PASS,
                                            port=DB_PORT,
                                            host=DB_HOST)
            print("Database Connected Sucessfully")
        except:
            print("Error in Connecting DataBase") 

        self.cur = self.connect.cursor()

    def CREATE_TABLE(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS AI4I(
        ID INT PRIMARY KEY NOT NULL,
        Product_ID TEXT NOT NULL,
        Type CHAR NOT NULL,
        Air_temperature_K FLOAT NOT NULL,
        Process_temperature_K FLOAT NOT NULL,
        Rotational_speed_rpm INT NOT NULL,
        Torque_Nm FLOAT NOT NULL,
        Tool_wear_min INT NOT NULL,
        Machine_failure BOOL NOT NULL,
        Tool_Wear_Failure_TWF BOOL NOT NULL ,
        Heat_Dissipation_Failure_HDF BOOL NOT NULL,
        Power_Failure_PWF BOOL NOT NULL,
        Overstrain_Failure_OSF BOOL NOT NULL,
        Random_Failures_RNF BOOL NOT NULL
        )
        """)

    def READ(self):
        self.cur.execute(f"""
        SELECT * FROM AI4I
        """)

    def WRITE(self,*columns):
        columns[0] += 1
        self.cur.execute(f"""
        INSERT INTO AI4I {columns}
        VALUES {"(" + ",".join(["%s"]*len(columns)) + ")"}
        """,(columns))

        self.connect.commit()

    def UPDATE(self,col):
        self.cur.execute("""
        UPDATE AI4I SET(
        ID={},
        Product_ID={},
        Type={},
        Air_temperature_K={},
        Process_temperature_K={},
        Rotational_speed_rpm={},
        Torque_Nm={},
        Tool_wear_min={},
        Machine_failure={},
        Tool_Wear_Failure_TWF={},
        Heat_Dissipation_Failure_HDF={},
        Power_Failure_PWF={},
        Overstrain_Failure_OSF={},
        Random_Failures_RNF={}
        WHERE ID={}
        )
        """.format(col[1:],col[0]))

        self.connect.commit()

    def DELETE(self,id):
        self.cur.execute("""DELETE FROM AIAI WHERE ID={}""",id)
        self.connect.commit()



