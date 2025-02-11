import logging
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from mlflow.models.signature import infer_signature

# scripts/mlops_tracking.py


def log_experiment(model, model_name, X_test, y_test):
    """
    Log experiment details using MLflow.
    """
    with mlflow.start_run():
        # Log model
        if model_name in ["CNN", "RNN", "LSTM"]:
            mlflow.keras.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        # Log metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))