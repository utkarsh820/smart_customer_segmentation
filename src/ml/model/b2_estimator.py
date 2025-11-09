import sys
from pandas import DataFrame
from src.cloud_storage.b2_storage import B2Storage
from src.exception import CustomerException
from src.ml.model.estimator import CustomerSegmentationModel


class B2ModelEstimator:
    def __init__(self, bucket_name: str, model_path: str):
        self.bucket_name = bucket_name
        self.b2 = B2Storage()
        self.model_path = model_path
        self.loaded_model: CustomerSegmentationModel = None

    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.b2.file_exists(bucket_name=self.bucket_name, file_name=model_path)
        except Exception as e:
            print(e)
            return False

    def load_model(self) -> CustomerSegmentationModel:
        return self.b2.load_model(bucket_name=self.bucket_name, model_path=self.model_path)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        try:
            self.b2.upload_file(
                bucket_name=self.bucket_name,
                local_path=from_file,
                file_name=self.model_path,
                remove=remove
            )
        except Exception as e:
            raise CustomerException(e, sys)

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise CustomerException(e, sys)
