from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running!"}

def test_valid_prediction():
    response = client.get("/predict/delays", params={
        "departure_airport": "JFK",
        "arrival_airport": "LAX",
        "departure_time": "14:00",
        "arrival_time": "17:30"
    })
    assert response.status_code == 200
    assert "predicted_delay_minutes" in response.json()

def test_invalid_airport():
    response = client.get("/predict/delays", params={
        "departure_airport": "XXX",
        "arrival_airport": "YYY",
        "departure_time": "14:00",
        "arrival_time": "17:30"
    })
    assert response.status_code == 400
    assert "error" in response.json()

def test_invalid_time_format():
    response = client.get("/predict/delays", params={
        "departure_airport": "JFK",
        "arrival_airport": "LAX",
        "departure_time": "abc",  # Invalid format
        "arrival_time": "xyz"
    })
    assert response.status_code == 400
    assert "error" in response.json()

