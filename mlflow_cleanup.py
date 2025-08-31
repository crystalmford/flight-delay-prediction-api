# Move all MLFlow experiments saved in Default folder to Flight Delay Prediction folder
import mlflow

# Get a list of all experiments
experiments = mlflow.search_experiments()

# Print experiment names and IDs
for exp in experiments:
    print(f"Experiment Name: {exp.name}, Experiment ID: {exp.experiment_id}")

