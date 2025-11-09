import json
import sys
from typing import Tuple, Union
import pandas as pd
from pandas import DataFrame

_EVIDENTLY_API_MODE = "missing"

try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset

    _EVIDENTLY_API_MODE = "new"
except ImportError:
    try:
        from evidently.model_profile import Profile
        from evidently.model_profile.sections import DataDriftProfileSection

        _EVIDENTLY_API_MODE = "legacy"
    except ImportError:
        _EVIDENTLY_API_MODE = "missing"

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig ):
        
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        
        self.utils = MainUtils()

        self._schema_config = self.utils.read_schema_config_file()
        self._raw_column_names = self._extract_column_names(self._schema_config.get("columns", []))
        self._engineered_column_names = [col.strip() for col in self._schema_config.get("engineered_columns", [])]
        self._detected_schema_type: Union[str, None] = None

    @staticmethod
    def _extract_column_names(columns_config) -> list:
        names = []
        for entry in columns_config or []:
            if isinstance(entry, dict):
                names.extend(key.strip() for key in entry.keys())
            else:
                names.append(str(entry).strip())
        return names

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_schema_columns
        Description :   This method validates the schema columns for the particular dataframe 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            dataframe_columns = [column.strip() for column in dataframe.columns]
            dataframe_column_set = set(dataframe_columns)

            raw_column_set = set(self._raw_column_names)
            engineered_column_set = set(self._engineered_column_names)

            expected_schema_type = self._detected_schema_type

            if expected_schema_type == "raw":
                status = dataframe_column_set == raw_column_set
                logging.info("Schema validation against raw column layout: %s", status)
                return status

            if expected_schema_type == "engineered":
                status = dataframe_column_set == engineered_column_set
                logging.info("Schema validation against engineered column layout: %s", status)
                return status

            if raw_column_set and dataframe_column_set == raw_column_set:
                self._detected_schema_type = "raw"
                logging.info("Detected raw schema layout for dataset")
                return True

            if engineered_column_set and dataframe_column_set == engineered_column_set:
                self._detected_schema_type = "engineered"
                logging.info("Detected engineered schema layout for dataset")
                return True

            logging.error("Dataset columns do not match any known schema layout")
            logging.error("Columns detected: %s", dataframe_columns)
            logging.error("Expected raw columns: %s", sorted(raw_column_set))
            logging.error("Expected engineered columns: %s", sorted(engineered_column_set))

            return False

        except Exception as e:
            raise CustomerException(e, sys) from e

   

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        """
        Method Name :   validate_dataset_schema_columns
        Description :   This method validates the schema for schema columns for both train and test set 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered validate_dataset_schema_columns method of Data_Validation class"
        )

        try:
            logging.info("Validating dataset schema columns")

            train_schema_status = self.validate_schema_columns(train_set)

            logging.info("Validated dataset schema columns on the train set")

            test_schema_status = self.validate_schema_columns(test_set)

            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")

            return train_schema_status, test_schema_status

        except Exception as e:
            raise CustomerException(e, sys) from e

    

    def detect_dataset_drift(
        self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   This method detects the dataset drift using the reference and production dataframe 
        
        Output      :   Returns bool or float value based on the get_ration parameter
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            if _EVIDENTLY_API_MODE == "new":
                data_drift_report = Report(metrics=[DataDriftPreset()])
                data_drift_report.run(reference_data=reference_df, current_data=current_df)

                report_dict = data_drift_report.as_dict()
                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=report_dict,
                )

                drift_metric = {}
                for metric in report_dict.get("metrics", []):
                    metric_info = metric.get("metric")
                    if isinstance(metric_info, str):
                        metric_type = metric_info
                    elif isinstance(metric_info, dict):
                        metric_type = metric_info.get("type")
                    else:
                        metric_type = None

                    if metric_type == "DataDriftPreset":
                        drift_metric = metric.get("result", {})
                        break

                n_features = drift_metric.get("number_of_columns", 0)
                n_drifted_features = drift_metric.get("number_of_drifted_columns", 0)

                logging.info(f"{n_drifted_features}/{n_features} drift detected.")

                drift_status = drift_metric.get("dataset_drift", False)
            elif _EVIDENTLY_API_MODE == "legacy":
                data_drift_profile = Profile(sections=[DataDriftProfileSection()])
                data_drift_profile.calculate(reference_df, current_df)

                report = data_drift_profile.json()
                json_report = json.loads(report)
                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=json_report,
                )

                n_features = json_report["data_drift"]["data"]["metrics"].get("n_features", 0)
                n_drifted_features = json_report["data_drift"]["data"]["metrics"].get(
                    "n_drifted_features", 0
                )

                logging.info(f"{n_drifted_features}/{n_features} drift detected.")

                drift_status = json_report["data_drift"]["data"]["metrics"].get("dataset_drift", False)
            else:
                logging.warning(
                    "Evidently not installed; skipping drift detection and marking dataset as stable."
                )

                fallback_report = {
                    "dataset_drift": False,
                    "details": "Evidently package not available; drift check skipped.",
                }
                write_yaml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content=fallback_report,
                )

                n_features = len(reference_df.columns)
                n_drifted_features = 0

                drift_status = False

            return drift_status
        
        except Exception as e:
            raise CustomerException(e, sys) from e
        
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")

            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            
            
            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            logging.info("Validated dataset schema columns")

            

            validation_status = schema_train_col_status and schema_test_col_status

            if not validation_status:
                logging.error("Dataset schema validation failed for train/test sets")
            elif drift is True:
                logging.warning("Data drift detected, continuing with latest dataset anyway")
            else:
                logging.info("Dataset schema validation completed")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
