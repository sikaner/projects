import os
import numpy as np
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn


def main():
# Tiny synthetic dataset: y = 2x + 3
X = np.arange(0, 100, dtype=float).reshape(-1, 1)
y = 2 * X.flatten() + 3


model = LinearRegression()
model.fit(X, y)


# Track locally inside the project folder.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("demo-mlflow")


with mlflow.start_run(run_name="linreg-demo"):
mlflow.log_param("coef0", float(model.coef_[0]))
mlflow.log_param("intercept", float(model.intercept_))
mlflow.sklearn.log_model(model, artifact_path="model")


# Also save a copy as an MLflow model directory that the API will load.
os.makedirs("model", exist_ok=True)
mlflow.sklearn.save_model(sk_model=model, path="model")
print("Model saved to ./model (MLflow format)")




if __name__ == "__main__":
main()
