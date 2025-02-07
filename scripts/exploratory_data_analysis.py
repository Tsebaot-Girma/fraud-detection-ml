# scripts/exploratory_data_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    """
    # Univariate Analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
    
    # Bivariate Analysis
    if 'class' in df.columns:
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='class', y=col, data=df)
            plt.title(f'{col} vs Fraud (class)')
            plt.show()