# scripts/data_preparation.py
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data(df, target_column):
    """
    Prepare data for model training by separating features and target, and splitting into train-test sets.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test