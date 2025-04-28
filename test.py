import mlflow
import mlflow.sklearn

# Set URI to point to the remote/local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Load the latest version of the registered model
model_name = "XGBoost_BestModel"
model_version = "1"  # or "1", "2", etc. if you want a specific version

# You can use pyfunc for general use or sklearn for sklearn-compatible models
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

print(model)