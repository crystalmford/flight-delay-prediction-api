# Flight Delay Prediction API

This project serves a machine learning model as a web API to predict departure delays for flights departing from Atlanta (ATL).  
It was developed for **D602 Task 3** in WGU's MSDA program and demonstrates model deployment using **FastAPI**.

---

## Project Overview
- Loads a pre-trained Ridge regression model with polynomial features  
- Accepts JSON input data for flight schedule details  
- Returns predicted departure delays via API endpoint  
- Includes unit tests to validate prediction logic  

---

## Tools & Libraries
- **FastAPI** – web framework for building the REST API  
- **scikit-learn** – model training (Ridge regression)  
- **MLflow** – experiment tracking  
- **Pandas / NumPy** – data preparation  
- **Uvicorn** – lightweight ASGI server for running the API  

---

## File Structure
    ├── app/
    │   ├── main.py          # API entry point (prediction endpoint lives here)
    │   ├── utils.py         # Helper functions for preprocessing & predictions
    │
    ├── data/                # Raw and/or sample datasets
    │
    ├── models/
    │   └── ridge_model.pkl  # Trained Ridge regression model
    │
    ├── notebooks/
    │   └── exploration.ipynb  # Data exploration & feature engineering
    │
    ├── tests/
    │   └── test_api.py      # Unit tests for the API
    │
    ├── requirements.txt     # Dependencies
    ├── README.md            # Project documentation

---

## How to Run
1. Clone the repo:  
       git clone https://github.com/yourusername/flight-delay-prediction.git
       cd flight-delay-prediction

2. Install dependencies:  
       pip install -r requirements.txt

3. Start the API:  
       uvicorn app.main:app --reload

4. Send a sample request:  
       curl -X POST "http://127.0.0.1:8000/predict" \
       -H "Content-Type: application/json" \
       -d '{"month": 6, "day": 15, "hour": 14, "airline": "DL", "destination": "LAX"}'

---

## Example Response
    {
      "predicted_delay": 12.5
    }

---

## Notes
- Built for educational purposes (WGU MSDA D602 Task 3)  
- Example dataset uses flights departing ATL  
- Model is Ridge regression with polynomial features

