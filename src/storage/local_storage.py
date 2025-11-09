import os
import pickle
import sys
from typing import Any
from src.exception import CustomerException
from src.logger import logging


class LocalStorage:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def save_object(self, file_path: str, obj: Any) -> None:
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f"Object saved to {file_path}")
        except Exception as e:
            raise CustomerException(e, sys)

    def load_object(self, file_path: str) -> Any:
        try:
            if not os.path.exists(file_path):
                raise Exception(f"File {file_path} does not exist")
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
            logging.info(f"Object loaded from {file_path}")
            return obj
        except Exception as e:
            raise CustomerException(e, sys)

    def file_exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)
