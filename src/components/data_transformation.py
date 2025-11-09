import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from imblearn.combine import SMOTETomek
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import SimpleImputerConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_tranasformation_config: DataTransformationConfig):
       
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_tranasformation_config
        self.data_ingestion = DataIngestion()

        self.imputer_config = SimpleImputerConfig()

        self.utils = MainUtils()
        self._schema_config = self.utils.read_schema_config_file()
        self._raw_column_names = self._extract_column_names(self._schema_config.get("columns", []))
        self._engineered_feature_columns = [col.strip() for col in self._schema_config.get("engineered_feature_columns", [])]
        self._engineered_column_names = [col.strip() for col in self._schema_config.get("engineered_columns", [])]
        
        
        
    
    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e,sys)

    @staticmethod
    def _extract_column_names(columns_config) -> list:
        names = []
        for entry in columns_config or []:
            if isinstance(entry, dict):
                names.extend(key.strip() for key in entry.keys())
            else:
                names.append(str(entry).strip())
        return names
        
        
    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        
        """
        method: get_new_features 
        objective:
                The following code creates features that would be helpful to describe the profile of the customer 
            recodes the customer's education level to numeric form (0: high-school, 1: diploma, 2: bachelors, 3: masters, and 4: doctorates)
            creates a new field to store the household size """
        
        raw_column_set = set(self._raw_column_names)
        engineered_column_set = set(self._engineered_column_names)

        feature_columns_order = self._engineered_feature_columns

        def _transform(dataset: DataFrame) -> DataFrame:
            df = dataset.copy()
            dataset_column_set = set(df.columns)

            if raw_column_set and dataset_column_set == raw_column_set:
                df['Age'] = 2022 - df['Year_Birth']
                df['Education'] = df['Education'].replace({"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4})
                df['Marital_Status'] = df['Marital_Status'].replace({"Married": 1, "Together": 1, "Absurd": 0, "Widow": 0, "YOLO": 0, "Divorced": 0, "Single": 0, "Alone": 0})
                df['Children'] = df['Kidhome'] + df['Teenhome']
                df['Family_Size'] = df['Marital_Status'] + df['Children'] + 1

                df['Total_Spending'] = (
                    df["MntWines"]
                    + df["MntFruits"]
                    + df["MntMeatProducts"]
                    + df["MntFishProducts"]
                    + df["MntSweetProducts"]
                    + df["MntGoldProds"]
                )
                df["Total Promo"] = (
                    df["AcceptedCmp1"]
                    + df["AcceptedCmp2"]
                    + df["AcceptedCmp3"]
                    + df["AcceptedCmp4"]
                    + df["AcceptedCmp5"]
                )

                df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
                today = datetime.today()
                df['Days_as_Customer'] = (today - df['Dt_Customer']).dt.days
                df['Offers_Responded_To'] = (
                    df['AcceptedCmp1']
                    + df['AcceptedCmp2']
                    + df['AcceptedCmp3']
                    + df['AcceptedCmp4']
                    + df['AcceptedCmp5']
                    + df['Response']
                )
                df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)

                df.drop(columns=['Year_Birth', 'Kidhome', 'Teenhome'], inplace=True)
                df.rename(
                    columns={
                        "Marital_Status": "Marital Status",
                        "MntWines": "Wines",
                        "MntFruits": "Fruits",
                        "MntMeatProducts": "Meat",
                        "MntFishProducts": "Fish",
                        "MntSweetProducts": "Sweets",
                        "MntGoldProds": "Gold",
                        "NumWebPurchases": "Web",
                        "NumCatalogPurchases": "Catalog",
                        "NumStorePurchases": "Store",
                        "NumDealsPurchases": "Discount Purchases",
                    },
                    inplace=True,
                )

                selected_columns = feature_columns_order or [
                    "Age",
                    "Education",
                    "Marital Status",
                    "Parental Status",
                    "Children",
                    "Income",
                    "Total_Spending",
                    "Days_as_Customer",
                    "Recency",
                    "Wines",
                    "Fruits",
                    "Meat",
                    "Fish",
                    "Sweets",
                    "Gold",
                    "Web",
                    "Catalog",
                    "Store",
                    "Discount Purchases",
                    "Total Promo",
                    "NumWebVisitsMonth",
                ]

                df = df[selected_columns]
                return df

            if engineered_column_set and dataset_column_set == engineered_column_set:
                df = df.drop(columns=[TARGET_COLUMN], errors='ignore')
                if not feature_columns_order:
                    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
                else:
                    feature_columns = feature_columns_order

                missing = [column for column in feature_columns if column not in df.columns]
                if missing:
                    raise CustomerException(
                        f"Engineered dataset missing expected feature columns: {missing}",
                        sys,
                    )

                df = df[feature_columns]
                return df

            raise CustomerException(
                f"Dataset columns {sorted(dataset_column_set)} do not match expected raw or engineered schemas",
                sys,
            )

        train_features = _transform(train_set)
        test_features = _transform(test_set)

        logging.info("Prepared feature set using %s schema", "raw" if set(train_set.columns) == raw_column_set else "engineered")
        return train_features, test_features
                
    


    def transform_data(self,train_set:DataFrame, test_set:DataFrame) -> DataFrame:
        """
        Method Name :   transform_data
        Description :   This method applies feature transformation and other feature
                        engineering operations and returns train and test datasets. 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logging.info("Got numerical cols from schema config")
            
            
            numeric_features = [feature for feature in train_set.columns if train_set[feature].dtype != 'O']


            outlier_features = ["Wines","Fruits","Meat","Fish","Sweets","Gold","Age","Total_Spending"]
            numeric_features = [x for x in numeric_features if x not in outlier_features]

 

            logging.info("Initialized StandardScaler, SimpleImputer")

            numeric_pipeline = Pipeline(steps=
                                        [("Imputer", SimpleImputer(**self.imputer_config.__dict__)), 
                                         ("StandardScaler", StandardScaler())]
            )
            
            outlier_features_pipeline = Pipeline(steps=
                                                 [("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                                                  ("transformer", PowerTransformer(standardize=True))]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numeric pipeline",numeric_pipeline, numeric_features),
                    ("Outliers Features Pipeline", outlier_features_pipeline, outlier_features)
            ]
            )
            
          
            

            
            
            preprocessed_train_set = preprocessor.fit_transform(train_set)
            preprocessed_test_set = preprocessor.transform(test_set)
            
            
            columns = train_set.columns
            preprocessed_train_set =  pd.DataFrame(preprocessed_train_set, columns=columns)
            preprocessed_test_set = pd.DataFrame(preprocessed_test_set, columns=columns)
            
            preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(preprocessor_obj_dir, exist_ok=True)
            self.utils.save_object(self.data_transformation_config.transformed_object_file_path , preprocessor)
            logging.info("Saved Preprocessor object to {}".format(preprocessor_obj_dir))


            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )

            return preprocessed_train_set, preprocessed_test_set

        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_data_transformation(self) :
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered initiate_data_transformation method of Data_Transformation class"
        )

        try:
            if self.data_validation_artifact.validation_status:
                train_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
                train_set, test_set = self.get_new_features(train_set, test_set)


                logging.info("Got the preprocessor object")
                
                preprocessed_train_set,  preprocessed_test_set  = self.transform_data(train_set, test_set)
                
                cluster_creator = CreateClusters()

                labelled_train_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_train_set)
                labelled_test_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_test_set)
                
                
                
                X_train = labelled_train_set.drop(columns=[TARGET_COLUMN], axis=1)
                y_train = labelled_train_set[TARGET_COLUMN]
                
                X_test = labelled_test_set.drop(columns=[TARGET_COLUMN], axis=1)
                y_test = labelled_test_set[TARGET_COLUMN]
                
                train_arr = np.c_[
                    np.array(X_train), np.array(y_train)
                ]
                
                test_arr = np.c_[
                    np.array(X_test), np.array(y_test)
                ]
                
                self.utils.save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                self.utils.save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                
                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
            
            
                return data_transformation_artifact
            
            else:
                raise Exception("Data Validation Failed.")



        except Exception as e:
            raise CustomerException(e, sys) from e
