import pandas as pd
import numpy as np
import os
import shutil
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load the cleaned dataset
file_path = r'C:\Users\Crystal\OneDrive\Desktop\d602\task 2\d602-deployment-task-2\data\cleaned_data.csv'
df = pd.read_csv(file_path)

# Define features and target variable
features = ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEST_AIRPORT']  # Use the features from the notebook
target = 'DEPARTURE_DELAY'

X = df[features]
y = df[target]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Apply OneHotEncoding to categorical features and PolynomialFeatures to numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PolynomialFeatures(degree=2, include_bias=False), numeric_cols),  # Polynomial regression
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encoding
    ]
)

# Use Ridge Regression instead of Linear Regression (matches the previous analyst's model)
model = Ridge(alpha=1.0)

# Create the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure all numeric features are in float64 format
X_train = X_train.astype({col: 'float64' for col in X_train.select_dtypes(include='int').columns})

# Define the example input for logging
example_input = pd.DataFrame([X_train.iloc[0].values], columns=X_train.columns)

# Set the experiment dynamically
experiment_name = "Flight Delay Prediction - Polynomial Regression"
mlflow.set_experiment(experiment_name)

# Start an MLFlow run
with mlflow.start_run():
    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Ensure arrays are flattened
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Log parameters and metrics
    mlflow.log_param("model_type", "Polynomial Regression (Ridge)")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("degree", 2)  # Polynomial degree
    mlflow.log_metric("rmse", rmse)

    # Log the model in MLFlow
    mlflow.sklearn.log_model(pipeline, "model", input_example=example_input)

    print("MLFlow experiment completed. Check MLFlow UI for results.")

# Save and register model
model_path = 'models/polynomial_regression_model'

# Remove the existing model directory if it exists
if os.path.exists(model_path):
    shutil.rmtree(model_path)

# Save model locally
mlflow.sklearn.save_model(pipeline, model_path)
print(f"Model saved at {model_path}")

# Register model in MLFlow Model Registry
client = MlflowClient()
model_name = "polynomial_regression_model"

# Check to see if model is already registered
try:
    client.get_registered_model(model_name)
    print(f"Model {model_name} already exists.")
except mlflow.exceptions.MlflowException:
    client.create_registered_model(model_name)
    print(f"Model {model_name} registered successfully.")

# Create a new model version
mlflow.sklearn.log_model(pipeline, "polynomial_regression_model")

print(f"Model version logged in MLFlow.")

