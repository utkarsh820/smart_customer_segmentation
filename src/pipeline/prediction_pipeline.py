import sys
import pandas as pd
from pandas import DataFrame
from src.ml.model.b2_estimator import B2ModelEstimator
from src.logger import logging
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig, ModelTrainerConfig
from src.utils.main_utils import MainUtils
from src.exception import CustomerException





class CustomerData:
    @staticmethod
    def _cast_value(value, target_type, column_name):
        if target_type in (int, "int", "int64", "int32"):
            if isinstance(value, str):
                cleaned_value = value.strip()
                categorical_mappings = {
                    "Education": {
                        "Basic": 0,
                        "2n Cycle": 1,
                        "Graduation": 2,
                        "Master": 3,
                        "PhD": 4,
                    },
                    "Marital Status": {
                        "Single": 0,
                        "Divorced": 0,
                        "Absurd": 0,
                        "Widow": 0,
                        "YOLO": 0,
                        "Alone": 0,
                        "Married": 1,
                        "Together": 1,
                        "Parent": 1,
                        "Non-Parent": 0,
                        "Yes": 1,
                        "No": 0,
                    },
                    "Parental Status": {
                        "Parent": 1,
                        "Non-Parent": 0,
                    },
                }

                column_mapping = categorical_mappings.get(column_name, {})
                if cleaned_value in column_mapping:
                    return int(column_mapping[cleaned_value])

                try:
                    return int(float(cleaned_value))
                except ValueError:
                    raise CustomerException(
                        f"Cannot convert value '{value}' for column '{column_name}' to integer.",
                        sys,
                    )
            return int(value)

        if target_type in (float, "float", "float64", "float32"):
            try:
                return float(value)
            except ValueError:
                raise CustomerException(
                    f"Cannot convert value '{value}' for column '{column_name}' to float.",
                    sys,
                )

        return value

    def get_input_dataset(self, column_schema: dict, input_data):
        columns = list(column_schema.keys())
        if len(columns) != len(input_data):
            raise CustomerException(
                f"Input data length {len(input_data)} does not match expected columns {len(columns)}",
                sys,
            )

        constructed_row = []
        for idx, column in enumerate(columns):
            dtype_label = column_schema[column]
            target_type = dtype_label if isinstance(dtype_label, type) else str(dtype_label).strip()
            value = input_data[idx]
            constructed_row.append(self._cast_value(value, target_type, column))

        input_dataset = pd.DataFrame([constructed_row], columns=columns)
        return input_dataset

    @staticmethod
    def form_input_dataframe(data):
        prediction_config = Prediction_config()
        column_schema = prediction_config.prediction_schema['columns']
        customerData = CustomerData()
        return customerData.get_input_dataset(column_schema=column_schema, input_data=data)
        
        
    


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()
        # Ensure environment variables (e.g., B2 credentials) are loaded when running in app contexts
        self.utils.load_dotenv_if_available()
        
    def prepare_input_data(self, input_data: list) -> pd.DataFrame:
        try:
            customerDataframe = CustomerData.form_input_dataframe(data=input_data)
            logging.info("Customer dataframe created")
            return customerDataframe
        except Exception as e:
            raise CustomerException(e, sys)
        
   
        
    
        
    def get_trained_model(self):
        try:
            prediction_config = PredictionPipelineConfig()
            model = B2ModelEstimator(
                bucket_name=prediction_config.model_bucket_name,
                model_path=prediction_config.model_file_name
            )
            return model
        except Exception as e:
            raise CustomerException(e, sys)
        
    def run_pipeline(self, input_data: list):
        try:
            input_dataframe = self.prepare_input_data(input_data)
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
        except Exception as e:
            raise CustomerException(e, sys)
            
            
        
            
        

 
        

        