"""
Train and log the Flight Delay Prediction model using MLflow.

This script loads cleaned flight delay data, trains a Ridge regression model
with polynomial numeric features and one-hot encoded categorical features,
logs metrics and parameters to MLflow, and saves the trained model locally.
"""

import os
import shutil

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


# Load cleaned dataset
file_path = os.path.join("data", "cleaned_data.csv")
df = pd.read_csv(file_path)

# Define features and target
features = ["SCHEDULED_DEPARTURE", "SCHEDULED_ARRIVAL", "DEST_AIRPORT"]
target = "DEPARTURE_DELAY"

X = df[features]
y = df[target]

# Identify feature types
categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", PolynomialFeatures(degree=2, include_bias=False), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Model
model = Ridge(alpha=1.0)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Example input for MLflow model signature
example_input = X_train.head(1)

# MLflow experiment
experiment_name = "Flight Delay Prediction - Polynomial Regression"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_param("model_type", "Polynomial Regression with Ridge")
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("polynomial_degree", 2)
    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=example_input,
    )

    print(f"MLflow experiment completed. RMSE: {rmse:.2f}")

# Save model locally
model_path = os.path.join("models", "polynomial_regression_model")

if os.path.exists(model_path):
    shutil.rmtree(model_path)

mlflow.sklearn.save_model(pipeline, model_path)

print(f"Model saved locally at: {model_path}")
