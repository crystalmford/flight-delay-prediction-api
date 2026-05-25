"""
mlflow_cleanup.py

Utility script for inspecting MLflow experiments associated with the
Flight Delay Prediction API project.

This script lists available MLflow experiments and their IDs so the user
can verify where model training runs were logged.
"""

import mlflow


def list_mlflow_experiments():
    """Print all MLflow experiment names and IDs."""
    experiments = mlflow.search_experiments()

    if not experiments:
        print("No MLflow experiments found.")
        return

    for exp in experiments:
        print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")


if __name__ == "__main__":
    list_mlflow_experiments()
