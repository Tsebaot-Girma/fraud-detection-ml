# scripts/data_cleaning.py
import pandas as pd

def clean_data(df):
    """
    Clean the dataset by removing duplicates and correcting data types.
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Correct data types
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'])
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    return df