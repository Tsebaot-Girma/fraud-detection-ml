# scripts/model_training.py
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning  # Correct import for UndefinedMetricWarning
import warnings

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate machine learning models.
    """
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Suppress UndefinedMetricWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        # Store results
        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    return results



def train_and_evaluate_deep_learning_models(models, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
    """
    Train and evaluate deep learning models.
    """
    results = {}
    for name, model in models.items():
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

        # Evaluate the model
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_proba = model.predict(X_test)

        # Suppress UndefinedMetricWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Store results
        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

    return results