# Use official Python image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ../../d602-deployment-task-3/flight_delay_prediction_api .

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
