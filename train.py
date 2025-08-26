import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

def main():
    X = np.array([[i] for i in range(10)])
    y = np.array([2*i + 1 for i in range(10)])

    model = LinearRegression()
    model.fit(X, y)

    mlflow.set_experiment("My First Project (fast-gate-439411-i4)")
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("r2", model.score(X, y))
        mlflow.sklearn.log_model(model, "model")

    # Save local artifact for the Flask app
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("Saved model to model/model.pkl")

if __name__ == "__main__":
    main()

