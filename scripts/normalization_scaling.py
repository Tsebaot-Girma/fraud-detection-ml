# scripts/normalization_scaling.py
from sklearn.preprocessing import StandardScaler

def normalize_data(df, numerical_cols):
    """
    Normalize numerical features using StandardScaler.
    """
    # Check if the specified columns exist in the DataFrame
    missing_cols = [col for col in numerical_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing: {missing_cols}")

    # Normalize the specified columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df