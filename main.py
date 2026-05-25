import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pickle

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define valid airport codes used for simple input validation
VALID_AIRPORTS = {"JFK", "LAX", "ORD", "ATL", "DFW"}


def convert_time_to_decimal(time_str: str) -> float:
    """Convert HH:MM time format into decimal hours."""
    hours, minutes = time_str.split(":")
    return int(hours) + int(minutes) / 60


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
        if departure_airport not in VALID_AIRPORTS or arrival_airport not in VALID_AIRPORTS:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid airport code"}
            )

        departure_time_float = convert_time_to_decimal(departure_time)
        arrival_time_float = convert_time_to_decimal(arrival_time)

        input_data = pd.DataFrame(
            [[arrival_airport, departure_time_float, arrival_time_float]],
            columns=[
                "DEST_AIRPORT",
                "SCHEDULED_DEPARTURE",
                "SCHEDULED_ARRIVAL"
            ]
        )

        prediction = model.predict(input_data)[0]

        return {"predicted_delay_minutes": round(float(prediction), 2)}

    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid time format. Use HH:MM, such as 12:30."}
        )

        return {"predicted_delay_minutes": prediction}

    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid time format. Use HH:MM (e.g., 12:30)."})
