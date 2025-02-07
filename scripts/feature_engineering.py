# scripts/feature_engineering.py
import pandas as pd

def create_features(df):
    """
    Create new features for fraud detection.
    """
    # Time-Based Features
    if 'purchase_time' in df.columns:
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
    
    # Transaction Frequency and Velocity
    if 'user_id' in df.columns:
        df['transaction_frequency'] = df.groupby('user_id')['purchase_time'].transform('count')
        df['transaction_velocity'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    
    return df