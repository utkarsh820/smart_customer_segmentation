import sys
from typing import Tuple
import os
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constant.database import DATABASE_NAME, COLLECTION_NAME
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.customer_data import CustomerData
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    def split_data_as_train_test(self, dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
        try:
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio
            )
            
            ingested_data_dir = self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir, exist_ok=True)
            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info("Train-test split completed")
        except Exception as e:
            raise CustomerException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            customer_data = CustomerData()
            customer_dataframe = customer_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
            
            logging.info(f"Dataframe shape: {customer_dataframe.shape}")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            
            customer_dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("Data exported to feature store")
            
            return customer_dataframe
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            
            schema_config = self.utils.read_schema_config_file()
            drop_columns = [col.strip() for col in schema_config.get("drop_columns", [])]
            available_drop_columns = [col for col in drop_columns if col in dataframe.columns]

            if available_drop_columns:
                dataframe = dataframe.drop(columns=available_drop_columns, axis=1)
            missing_columns = sorted(set(drop_columns) - set(available_drop_columns))
            if missing_columns:
                logging.warning(
                    "Configured drop columns not present in source data: %s",
                    ", ".join(missing_columns),
                )
            
            self.split_data_as_train_test(dataframe)
            
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            
            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerException(e, sys)
