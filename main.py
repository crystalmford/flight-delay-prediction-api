import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as model_file:


    model = pickle.load(model_file)

# Define valid airport codes
VALID_AIRPORTS = {"JFK", "LAX", "ORD", "ATL", "DFW"}  # Add more as needed

@app.get("/")
def read_root():
    return {"message": "API is up and running!"}

@app.get("/predict/delays")
def predict_delays(
    departure_airport: str,
    arrival_airport: str,
    departure_time: str,
    arrival_time: str
):
    try:
        # Validate airport codes
        if departure_airport not in VALID_AIRPORTS or arrival_airport not in VALID_AIRPORTS:
            return JSONResponse(status_code=400, content={"error": "Invalid airport code"})

        # Convert time inputs from HH:MM format to float (e.g., "12:30" → 12.5)
        departure_time_float = float(departure_time.replace(":", "."))
        arrival_time_float = float(arrival_time.replace(":", "."))

        # Create input DataFrame with correct format
        input_data = pd.DataFrame(
            [[departure_airport, arrival_airport, departure_time_float, arrival_time_float]],
            columns=["ORIGIN_AIRPORT", "DEST_AIRPORT", "SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL"]
        )

        # Make prediction
        prediction = model.predict(input_data)[0]

        return {"predicted_delay_minutes": prediction}

    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid time format. Use HH:MM (e.g., 12:30)."})
