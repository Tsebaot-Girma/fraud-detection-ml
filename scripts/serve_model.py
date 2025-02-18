from flask import Flask, request, jsonify
from joblib import load 
import sys
import logging

# Add the scripts directory to the path
sys.path.append('../scripts')


app = Flask(__name__)

# Load your trained model using joblib
model = load('../models/random_forest_fraud.pkl')  

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Received prediction request')
    data = request.get_json()
    prediction = model.predict([data['features']])
    app.logger.info(f'Prediction: {prediction[0]}')
    return jsonify({'prediction': int(prediction[0])})













# Load fraud data (replace with your dataset)
fraud_data = pd.read_csv('../data/Fraud_Data.csv')
ip_to_country = pd.read_csv('../data/IpAddress_to_Country.csv')

# Merge datasets for geolocation analysis
fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
merged_data = pd.merge(fraud_data, ip_to_country, left_on='ip_address', right_on='lower_bound_ip_address', how='left')

@app.route('/fraud-summary', methods=['GET'])
def fraud_summary():
    # Calculate summary statistics
    total_transactions = len(merged_data)
    fraud_cases = merged_data[merged_data['class'] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100

    # Return summary statistics
    return jsonify({
        'total_transactions': total_transactions,
        'fraud_cases': fraud_cases,
        'fraud_percentage': fraud_percentage
    })

@app.route('/fraud-data', methods=['GET'])
def fraud_data():
    # Return fraud data for visualization
    return merged_data.to_json(orient='records')

if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5000, debug=True)