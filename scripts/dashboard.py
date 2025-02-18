import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import requests

# Initialize Dash app
app = dash.Dash(__name__)

# Fetch data from Flask backend
response = requests.get('http://localhost:5000/fraud-data')
fraud_data = pd.read_json(response.json(), orient='records')

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'textAlign': 'center'}),
    
    # Summary boxes
    html.Div([
        html.Div(id='total-transactions', className='summary-box'),
        html.Div(id='fraud-cases', className='summary-box'),
        html.Div(id='fraud-percentage', className='summary-box')
    ], className='summary-container'),

    # Line chart for fraud cases over time
    dcc.Graph(id='fraud-over-time'),

    # Bar chart for fraud cases by device
    dcc.Graph(id='fraud-by-device'),

    # Bar chart for fraud cases by browser
    dcc.Graph(id='fraud-by-browser'),

    # Geographical map for fraud cases by country
    dcc.Graph(id='fraud-by-country')
])

# Callback to update summary boxes
@app.callback(
    [Output('total-transactions', 'children'),
     Output('fraud-cases', 'children'),
     Output('fraud-percentage', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_summary(n):
    response = requests.get('http://localhost:5000/fraud-summary')
    summary = response.json()
    return [
        f"Total Transactions: {summary['total_transactions']}",
        f"Fraud Cases: {summary['fraud_cases']}",
        f"Fraud Percentage: {summary['fraud_percentage']:.2f}%"
    ]

# Callback to update fraud over time chart
@app.callback(
    Output('fraud-over-time', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_fraud_over_time(n):
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_over_time = fraud_data.groupby(fraud_data['purchase_time'].dt.date)['class'].sum().reset_index()
    fig = px.line(fraud_over_time, x='purchase_time', y='class', title='Fraud Cases Over Time')
    return fig

# Callback to update fraud by device chart
@app.callback(
    Output('fraud-by-device', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_fraud_by_device(n):
    fraud_by_device = fraud_data.groupby('device_id')['class'].sum().reset_index()
    fig = px.bar(fraud_by_device, x='device_id', y='class', title='Fraud Cases by Device')
    return fig

# Callback to update fraud by browser chart
@app.callback(
    Output('fraud-by-browser', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_fraud_by_browser(n):
    fraud_by_browser = fraud_data.groupby('browser')['class'].sum().reset_index()
    fig = px.bar(fraud_by_browser, x='browser', y='class', title='Fraud Cases by Browser')
    return fig

# Callback to update fraud by country chart
@app.callback(
    Output('fraud-by-country', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_fraud_by_country(n):
    fraud_by_country = fraud_data.groupby('country')['class'].sum().reset_index()
    fig = px.choropleth(fraud_by_country, locations='country', locationmode='country names', color='class', title='Fraud Cases by Country')
    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.', port=8050)