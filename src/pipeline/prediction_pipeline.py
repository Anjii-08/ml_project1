import os
import sys
from src.utils import load_model
from src.exception import CustomException
from src.logger import logging
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifact","preprocessor.pkl")
            model_path=os.path.join("artifact","model.pkl")

            preprocessor=load_model(preprocessor_path)
            model=load_model(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Error in prediction of data")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,carat:float,depth:float,table:float,x:float,y:float,z:float,cut:str,color:str,clarity:str):
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color=color
        self.clarity=clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }

            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created")

            return df
        except Exception as e:
            logging.info("Error in making dataframe")
            raise CustomException(e,sys)