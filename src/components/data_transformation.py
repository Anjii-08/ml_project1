import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_model 
from src.components.data_ingestion import DataIngestion
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()

            
    def get_data_transfromation_object(self):
        try:
            logging.info("DAta transformation initiate")
            cat_cols=["cut","color","clarity"]
            num_cols=["carat","depth","table","x","y","z"]
            

            cut_cat=["Fair","Good","Very Good","Premium","Ideal"]
            col_cat=['F' ,'J', 'G' ,'E', 'D' ,'H' ,'I']
            clar_cat=['VS2' ,'SI2' ,'VS1', 'SI1', 'IF' ,'VVS2' ,'VVS1' ,'I1']
            
            logging.info('pipeline initiate')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler()),
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('encoder',OrdinalEncoder(categories=[cut_cat,col_cat,clar_cat])),
                    ('scaler',StandardScaler())
                ]
            )
            pre_processor=ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )
            logging.info("pipeline created")
            return pre_processor
        except Exception as e:
            logging.info("Error in data transformation")
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('read train and test data completed')
            logging.info(f"train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"train dataframe head : n{test_df.head().to_string()}")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transfromation_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("applying preprocessing object on training and testng datasets")

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_model(
                file_path=self.data_transformation.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            logging.info("preprocessor pickle files save")

            return (train_arr,test_arr,self.data_transformation.preprocessor_obj_file)
        except Exception as e:
            logging.info('Exception occured in initiate data transfromation')
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path,test_data_path)