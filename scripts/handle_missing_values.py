# scripts/handle_missing_values.py
import pandas as pd

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    - Impute numerical columns with median.
    - Drop rows with missing values in categorical columns.
    """
    # Impute numerical columns with median
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Drop rows with missing values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df.dropna(subset=categorical_cols, inplace=True)
    
    return df