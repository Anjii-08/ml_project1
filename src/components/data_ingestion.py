import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split



class DataIngestionConfi():
    raw_data_path=os.path.join("artifact","raw.csv")
    train_data_path=os.path.join("artifact","train.csv")
    test_data_path=os.path.join("artifact","test.csv")


class DataIngestion():
    def __init__(self):
        self.ingestion_config=DataIngestionConfi()


    def initiate_data_ingestion(self):
        logging.info("Starting my data ingesion")
        
        try:
            df=pd.read_csv("https://raw.githubusercontent.com/sunnysavita10/ML_Project_With_ContinuesTraining/main/notebooks/data/gemstone.csv")
            logging.info("raw data reading complete")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path)

            train_set,test_set=train_test_split(df,test_size=0.25,random_state=42)
            logging.info("Train test split is done")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("data ingestion is complete")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path
                )


        except Exception as e:
            logging.info("Error in initiate_data_ingesion")
            raise CustomException(e,sys)

    
if __name__=="__main__":
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()