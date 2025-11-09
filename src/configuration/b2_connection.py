import os
import sys
from b2sdk.v2 import B2Api, InMemoryAccountInfo
from src.constant.env_variable import B2_APPLICATION_KEY_ID, B2_APPLICATION_KEY
from src.exception import CustomerException


class B2Client:
    b2_api = None
    
    def __init__(self):
        if B2Client.b2_api is None:
            key_id = os.getenv(B2_APPLICATION_KEY_ID)
            app_key = os.getenv(B2_APPLICATION_KEY)
            
            if key_id:
                key_id = key_id.strip()
            if app_key:
                app_key = app_key.strip()

            if not key_id:
                raise Exception(f"Environment variable: {B2_APPLICATION_KEY_ID} is not set.")
            if not app_key:
                raise Exception(f"Environment variable: {B2_APPLICATION_KEY} is not set.")
            
            try:
                info = InMemoryAccountInfo()
                B2Client.b2_api = B2Api(info)
                B2Client.b2_api.authorize_account("production", key_id, app_key)
            except Exception as e:
                raise CustomerException(e, sys)
        
        self.b2_api = B2Client.b2_api
