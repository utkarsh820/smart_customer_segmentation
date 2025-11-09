import sys
from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelPusherConfig
from src.exception import CustomerException
from src.logger import logging
from src.ml.model.b2_estimator import B2ModelEstimator


class ModelPusher:
    def __init__(self, model_trainer_artifact: ModelTrainerArtifact, model_pusher_config: ModelPusherConfig):
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.estimator = B2ModelEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.b2_model_key_path,
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Uploading model to B2 bucket")
            self.estimator.save_model(from_file=self.model_trainer_artifact.trained_model_file_path)
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.b2_model_key_path,
            )
            logging.info("Model uploaded to B2 bucket")
            return model_pusher_artifact
        except Exception as e:
            raise CustomerException(e, sys)
