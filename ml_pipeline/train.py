import os
import sys
import joblib
import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from data_preprocessing import preprocess
from optuna.samplers import TPESampler

# Setup base path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Setup Logging
log_dir = os.path.join(base_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("optuna_xgboost_hyperparameter_tuning_new")

# Load and preprocess data
logger.info("Loading and preprocessing data...")
X_train, y_train = preprocess(os.path.join(base_dir, 'dataset_splits/train.csv'))
X_val, y_val = preprocess(os.path.join(base_dir, 'dataset_splits/val.csv'))

logger.info(f"Original training samples: {len(X_train)}")
X_train, y_train = SMOTE(sampling_strategy=0.3, random_state=42).fit_resample(X_train, y_train)
logger.info(f"Resampled training samples after SMOTE: {len(X_train)}")

# Objective function for Optuna hyperparameter tuning
def objective(trial):
    with mlflow.start_run(nested=True):
        # Define hyperparameters to tune
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),  # important!
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        threshold = trial.suggest_float("threshold", 0.005, 0.05)


        logger.info(f"Trial {trial.number+1} - Params: {params}")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predict fraud probabilities
        y_proba = model.predict_proba(X_val)[:, 1]

        # Apply a lower threshold (important!!)
        y_pred = (y_proba > threshold).astype(int)

        # Evaluate model performance
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)

        # Log hyperparameters and metrics with MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgb_model",
            registered_model_name="XGBoost_Tuning_New",
        )

        logger.info(f"Trial {trial.number+1} - Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

        return f1  # ⚡ Focus on maximizing F1 instead of Accuracy

# Run Optuna optimization
logger.info("Starting Optuna hyperparameter tuning")

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=20)

logger.info(f"Best trial: {study.best_trial.number}, Best F1 score: {study.best_value:.4f}")

# Train final model with the best hyperparameters found by Optuna
logger.info("Training final model with best hyperparameters...")

# Extract best threshold separately
best_params = study.best_params.copy()
best_threshold = best_params.pop("threshold")  # Remove threshold from model params

best_model = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric="logloss")
best_model.fit(X_train, y_train)


logging.info(f"Best threshold : {best_threshold}")

# Save the final model and final model 

save_dict = {
    'model' : best_model,
    'threshold' : best_threshold
}

model_path = os.path.join(base_dir, "shared_folder")
os.makedirs(model_path, exist_ok=True)

joblib.dump(save_dict, os.path.join(model_path, "model.pkl"))
logger.info("Saved best model to best_model/model.pkl")

# Register final model with MLflow
with mlflow.start_run(run_name="final_model_new"):

    mlflow.log_params(study.best_params)
    mlflow.log_param("best_threshold", best_threshold)  # Save threshold separately in MLflow

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="xgb_model",
        registered_model_name="XGBoost_BestModel_New",
    )
    logger.info("Final model logged and registered with MLflow.")