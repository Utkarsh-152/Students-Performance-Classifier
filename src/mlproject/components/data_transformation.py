import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_obj
import os

@dataclass
class DataTransfromationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransfromation:
    def __init__(self):
        self.data_transformation_config =DataTransfromationConfig()
        
    def get_data_transformer_obj(self):
        ''''this function is responsible for data transformation'''
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                
            ]
            
            num_pipeline=Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scalar",StandardScaler())])
            
            cat_pipeline=Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder()),
                ("scalar",StandardScaler(with_mean=False))])
            
            logging.info(f"categorical columns:{categorical_features}")
            logging.info(f"numerical columns:{numerical_features}")
            
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline, numerical_features),
                    ("categorical_pipeline",cat_pipeline, categorical_features)
                ]
            )
            
            return preprocessor        
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiatiate_transfromation(self, train_path, test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("reading the train and test file")
            
            preprocessing_obj =self.get_data_transformer_obj()
            
            target_column_name= "math_score"
            numerical_column_name=["writing_score", "reading_score"]
            
            #dividing the train dataset to independent and dependent feature
            imput_features_train_df=train_df.drop([target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            #dividing the test dataset to independent and dependent feature
            imput_features_test_df=test_df.drop([target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying preprocessing on training and test Dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(imput_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(imput_features_test_df)
            
            train_arr = np.c_[                
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[                
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved Preprocessing Object")
            
            save_obj(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
            )
            
             
            
        except Exception as e:
            raise CustomException(e,sys)
            
            
        