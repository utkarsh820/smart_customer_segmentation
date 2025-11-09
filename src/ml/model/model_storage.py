import os
import sys
from pandas import DataFrame
from src.exception import CustomerException
from src.storage.local_storage import LocalStorage
from src.ml.model.estimator import CustomerSegmentationModel


class ModelStorage:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.storage = LocalStorage(base_path=model_dir)
        self.loaded_model: CustomerSegmentationModel = None

    def get_model_path(self, model_name: str = "model.pkl") -> str:
        return os.path.join(self.model_dir, model_name)

    def is_model_present(self, model_name: str = "model.pkl") -> bool:
        model_path = self.get_model_path(model_name)
        return self.storage.file_exists(model_path)

    def load_model(self, model_name: str = "model.pkl") -> CustomerSegmentationModel:
        try:
            model_path = self.get_model_path(model_name)
            return self.storage.load_object(model_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def save_model(self, model: CustomerSegmentationModel, model_name: str = "model.pkl") -> None:
        try:
            model_path = self.get_model_path(model_name)
            self.storage.save_object(model_path, model)
        except Exception as e:
            raise CustomerException(e, sys)

    def predict(self, dataframe: DataFrame, model_name: str = "model.pkl"):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model(model_name)
            return self.loaded_model.predict(dataframe)
        except Exception as e:
            raise CustomerException(e, sys)
