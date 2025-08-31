# Flight Delay Prediction API

## Overview
This project serves a machine learning model as a **web API** to predict departure delays for flights departing from Atlanta (ATL).
It was originally developed for WGU’s **MSDA D602 Task 3**, and demonstrates **end-to-end model deployment** using FastAPI.

The goal is to show how a predictive model can be deployed as a production-ready service that accepts live inputs and returns results instantly.

## Project Highlights
- Machine learning model: Ridge regression with polynomial features (trained on ATL flight delay data)
- API: FastAPI REST endpoint serving predictions in real-time
- Testing: Unit tests included to validate prediction logic
- Experiment tracking: MLflow used to track training runs and model performance

## Tools & Libraries
- FastAPI – web framework for building the REST API
- scikit-learn – Ridge regression model
- MLflow – experiment tracking
- Pandas / NumPy – data preparation
- Uvicorn – ASGI server for running the API

## Key Files
- `main.py` – API entry point (prediction endpoint lives here)
- `utils.py` – helper functions for preprocessing & predictions
- `ridge_model.pkl` – trained Ridge regression model
- `exploration.ipynb` – notebook for data exploration & feature engineering
- `test_api.py` – unit tests for the API
- `requirements.txt` – dependencies
- `README.md` – project documentation

## How to Run Locally
Clone the repo:

    git clone https://github.com/crystalmford/flight-delay-prediction-api.git
    cd flight-delay-prediction-api

Install dependencies:

    pip install -r requirements.txt

Start the API:

    uvicorn main:app --reload

Send a sample request:

    curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"month": 6, "day": 15, "hour": 14, "airline": "DL", "destination": "LAX"}'

Example response:

    {
      "predicted_delay": 12.5
    }

## Notes
- Built as part of **WGU MSDA D602 Task 3** (educational project).
- Dataset: flights departing Atlanta (ATL).
- Model: Ridge regression with polynomial features.

## Key Takeaways
- Demonstrates how to deploy ML models with FastAPI.
- Shows how APIs can turn predictive models into usable services.
- Includes testing and experiment tracking for reproducibility.

