import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.utils import save_model
from src.utils import evaluate_model
from dataclasses import dataclass
import os
import sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }

            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            print(model_report)
            print('\n========================================================================================')
            logging.info(f"Model Report: {model_report}")

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")
            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            # Save the best model
            save_model(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            logging.info("Exception occurred in initiate_model_trainer")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)