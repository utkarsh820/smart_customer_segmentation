import os
import sys
import pickle
from io import BytesIO

from src.configuration.b2_connection import B2Client
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class B2Storage:
    def __init__(self):
        self._ensure_environment()
        b2_client = B2Client()
        self.b2_api = b2_client.b2_api

    @staticmethod
    def _ensure_environment():
        env_loader = MainUtils()
        env_loader.load_dotenv_if_available()

    def file_exists(self, bucket_name: str, file_name: str) -> bool:
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            try:
                bucket.get_file_info_by_name(file_name)
                return True
            except:
                return False
        except Exception as e:
            raise CustomerException(e, sys)

    def download_file(self, bucket_name: str, file_name: str, local_path: str) -> None:
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            bucket.download_file_by_name(file_name).save_to(local_path)
            logging.info(f"Downloaded {file_name} from {bucket_name} to {local_path}")
        except Exception as e:
            raise CustomerException(e, sys)

    def upload_file(self, bucket_name: str, local_path: str, file_name: str, remove: bool = False) -> None:
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            bucket.upload_local_file(local_file=local_path, file_name=file_name)
            logging.info(f"Uploaded {local_path} to {bucket_name}/{file_name}")
            if remove:
                os.remove(local_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def load_model(self, bucket_name: str, model_path: str):
        try:
            bucket = self.b2_api.get_bucket_by_name(bucket_name)
            downloaded_file = bucket.download_file_by_name(model_path)
            file_data = BytesIO()
            downloaded_file.save(file_data)
            file_data.seek(0)
            model = pickle.load(file_data)
            logging.info(f"Loaded model from {bucket_name}/{model_path}")
            return model
        except Exception as e:
            raise CustomerException(e, sys)
