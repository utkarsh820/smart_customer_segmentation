import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

from src.pipeline.train_pipeline import TrainPipeline
from src.logger import logging

def main():
    try:
        logging.info("="*50)
        logging.info("Starting Training Pipeline")
        logging.info("="*50)
        
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        
        logging.info("="*50)
        logging.info("Training Pipeline Completed Successfully")
        logging.info("Model has been pushed to B2 bucket")
        logging.info("="*50)
        
        print("\n✅ Training completed successfully!")
        print("📦 Model has been pushed to Backblaze B2 bucket")
        print("🎯 You can now use the model for predictions\n")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        print(f"\n❌ Training failed: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
