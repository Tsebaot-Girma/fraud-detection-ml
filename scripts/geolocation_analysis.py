def merge_geolocation_data(fraud_df, ip_country_df):
    """
    Merge Fraud_Data with IpAddress_to_Country for geolocation analysis.
    """
    # Ensure the ip_address column is treated as an integer
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)

    # Create a temporary DataFrame to hold the results
    fraud_df['country'] = None

    # Iterate through each row in the ip_country DataFrame
    for index, row in ip_country_df.iterrows():
        # Create a mask to find matching IP addresses
        mask = (fraud_df['ip_address'] >= row['lower_bound_ip_address']) & \
               (fraud_df['ip_address'] <= row['upper_bound_ip_address'])
        
        # Assign the country to the matching rows
        fraud_df.loc[mask, 'country'] = row['country']
        fraud_df.loc[mask, 'lower_bound_ip_address'] = row['lower_bound_ip_address']
        fraud_df.loc[mask, 'upper_bound_ip_address'] = row['upper_bound_ip_address']

    return fraud_df