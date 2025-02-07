# scripts/geolocation_analysis.py
import pandas as pd

def merge_geolocation_data(fraud_df, ip_country_df):
    """
    Merge Fraud_Data.csv with IpAddress_to_Country.csv for geolocation analysis.
    """
    # Ensure the ip_address column is treated as an integer
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)

    # Merge datasets
    merged_df = pd.merge(fraud_df, ip_country_df, how='left', left_on='ip_address', right_on='lower_bound_ip_address')

    return merged_df