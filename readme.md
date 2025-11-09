# Customer Categorizer

ML-powered customer segmentation system using clustering and classification.

## Features

- Customer category prediction
- Model training pipeline
- MongoDB data integration
- Backblaze B2 model storage
- Streamlit web interface
- Docker support

## Quick Start

### Using Docker

1. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

	Required keys:
	- `MONGO_DB_URL`
	- `B2_APPLICATION_KEY_ID`
	- `B2_APPLICATION_KEY`
	- `B2_BUCKET_NAME` (only if your Backblaze key is scoped to a specific bucket)

2. Build and run:
```bash
docker-compose up --build
```

3. Access at `http://localhost:8501`

### Local Development

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
2. Install Poetry:
```bash
poetry install
```
3. Activate Enviroment:
### one time plugin install
```bash
poetry self add poetry-plugin-shell
```
---
### enviroment activate (whenever want to enter in enviroment)
```bash
poetry shell
```

4. Set environment variables:
```bash
export MONGO_DB_URL="your_mongodb_url"
export B2_APPLICATION_KEY_ID="your_key_id"
export B2_APPLICATION_KEY="your_app_key"
export B2_BUCKET_NAME="Bucket_Name"   # optional if your key is bucket-restricted
```

5. Run app:
```bash
streamlit run streamlit_app.py
```

The app loads `.env` automatically when available. When running inside Streamlit Cloud, provide the same keys in **App Settings → Secrets**.

## Environment Variables

- `MONGO_DB_URL`: MongoDB connection string
- `B2_APPLICATION_KEY_ID`: Backblaze B2 key ID
- `B2_APPLICATION_KEY`: Backblaze B2 application key
- `B2_BUCKET_NAME`: Override bucket when key is restricted (defaults to `customer0`)

## Project Structure

```
├── src/
│   ├── components/      # Pipeline components
│   ├── configuration/   # DB and B2 connections
│   ├── pipeline/        # Train and prediction pipelines
│   ├── ml/             # Model definitions
│   └── cloud_storage/  # B2 storage handler
├── data/               # Raw and processed data
├── models/             # Trained models
├── streamlit_app.py    # Streamlit UI
├── Dockerfile          # Docker configuration
└── docker-compose.yml  # Docker Compose setup
```

## Training

### Upload Data (First Time)
```bash
python upload_data.py
```

### Train Model
```bash
python train.py
```

Model is automatically pushed to B2 after training.

## Prediction

```python
from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()
result = pipeline.run_pipeline(customer_data)
```

## Deploying on Streamlit Cloud

1. Push the repository (including `streamlit_app.py` and `requirements.txt`) to GitHub.
2. In Streamlit Cloud, create a new app → select this repo and point to `streamlit_app.py`.
3. In *App Settings → Secrets*, add `MONGO_DB_URL`, `B2_APPLICATION_KEY_ID`, `B2_APPLICATION_KEY`, and `B2_BUCKET_NAME` if needed. These values are injected automatically by the app through `st.secrets`.
4. Ensure the trained model (`model.pkl`) exists in the configured Backblaze bucket so predictions can load it at startup.
