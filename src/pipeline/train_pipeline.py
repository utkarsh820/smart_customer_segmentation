import sys
from typing import Tuple

from pandas import DataFrame


from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.components.model_pusher import ModelPusher

from src.exception import CustomerException
from src.logger import logging
from src.entity.artifact_entity import (DataIngestionArtifact,
                                           DataTransformationArtifact,
                                           DataValidationArtifact,
                                           ModelEvaluationArtifact,
                                           ModelTrainerArtifact)
from src.entity.config_entity import (DataIngestionConfig,
                                         DataTransformationConfig,
                                         DataValidationConfig,
                                         ModelEvaluationConfig,
                                         ModelPusherConfig, ModelTrainerConfig)



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_tranasformation_config=self.data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    

    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed")
            return model_trainer_artifact
        except Exception as e:
            raise CustomerException(e, sys)
        
    def start_model_evaluation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ) -> ModelEvaluationArtifact:
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact= data_transformation_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info("Starting model push to B2")
            model_pusher = ModelPusher(
                model_trainer_artifact=model_trainer_artifact,
                model_pusher_config=self.model_pusher_config,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pushed to B2 successfully")
            return model_pusher_artifact
        except Exception as e:
            raise CustomerException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("Starting training pipeline")
            
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact=data_ingestion_artifact,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            
            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted")
                return None
            
            model_pusher_artifact = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
            
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            raise CustomerException(e, sys)
