# scripts/encode_categorical_features.py
import pandas as pd

def encode_categorical_features(df, categorical_cols):
    """
    Encode categorical features using One-Hot Encoding.
    """
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df