import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

def main():
    # Training data
    X = np.array([[i] for i in range(10)])
    y = np.array([2*i + 1 for i in range(10)])

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # MLflow logging
    mlflow.set_experiment("My First Project (fast-gate-439411-i4)")
    with mlflow.start_run():
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("r2", model.score(X, y))

        # Log model with proper name, signature & input example
        mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_regression_model",
            input_example=X[:2],
            signature=infer_signature(X, model.predict(X))
        )

    # Save local artifact for Flask
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    print("Saved model to model/model.pkl")

if __name__ == "__main__":
    main()


