# Flight Delay Prediction API

## Overview

This project deploys a machine learning model as a REST API capable of predicting flight departure delays for flights departing from Atlanta (ATL).

This project demonstrates an end-to-end machine learning deployment workflow including:

- data preprocessing
- model training
- experiment tracking with MLflow
- API deployment using FastAPI
- automated API testing

The application accepts flight information as input and returns a predicted departure delay in real time.

---

# Swagger Documentation

FastAPI automatically generates interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Features

- FastAPI REST API for live predictions
- Ridge Regression model with polynomial feature engineering
- MLflow experiment tracking
- Automated API testing with pytest
- Docker support for containerized deployment
- Scikit-learn preprocessing pipelines

---

# Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| FastAPI | REST API framework |
| scikit-learn | Machine learning pipeline |
| MLflow | Experiment tracking |
| Pandas / NumPy | Data processing |
| Uvicorn | ASGI server |
| pytest | API testing |
| Docker | Containerized deployment |

---

# Project Structure

```text
flight-delay-prediction-api/
│
├── main.py
├── mlflow_experiment.py
├── requirements.txt
├── Dockerfile
├── model.pkl
├── test_main.py
├── flight_delay_prediction_walkthrough.ipynb
└── README.md
```

---

# API Endpoint

## Predict Flight Delay

**GET** `/predict/delays`

Example request:

```text
http://127.0.0.1:8000/predict/delays?departure_airport=JFK&arrival_airport=LAX&departure_time=14:00&arrival_time=17:30
```

Example response:

```json
{
  "predicted_delay_minutes": 12.5
}
```

---

# Running the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/crystalmford/flight-delay-prediction-api.git
cd flight-delay-prediction-api
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

API available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

# Running with Docker

Build the container:

```bash
docker build -t flight-delay-api .
```

Run the container:

```bash
docker run -p 8000:8000 flight-delay-api
```

---

# Running Tests

```bash
pytest
```

---

# MLflow Experiment Tracking

The project uses MLflow to:

- track model parameters
- log evaluation metrics
- store experiment runs
- manage trained model artifacts

To run the training script:

```bash
python mlflow_experiment.py
```

---

# Dataset Notes

This project was trained using cleaned flight delay data for flights departing from Atlanta (ATL).

The training dataset is not included in the repository.

The deployed API uses the serialized trained model stored in:

```text
model.pkl
```

---

# Key Takeaways

This project demonstrates practical machine learning deployment concepts including:

- serving ML models through APIs
- preprocessing pipelines
- experiment tracking
- automated testing
- containerization with Docker
