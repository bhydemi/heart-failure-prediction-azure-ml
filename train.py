import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--solver', type=str, default='lbfgs',
                        help='Algorithm to use in optimization')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path to the data folder')
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()

    # Log hyperparameters
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("solver", args.solver)

    # Load the heart failure dataset
    data_file = os.path.join(args.data_path, 'heart_failure_clinical_records_dataset.csv')
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
    else:
        # Try current directory
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

    print(f"Loaded data with shape: {df.shape}")

    # Prepare features and target
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train the model
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Log metrics
    mlflow.log_metric("Accuracy", accuracy)

    # Save the model
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/model.joblib'
    joblib.dump(model, model_path)

    # Log the model as an artifact instead of using log_model
    mlflow.log_artifact(model_path)

    mlflow.end_run()

if __name__ == '__main__':
    main()
