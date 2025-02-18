# scripts/shap_explainability.py

import shap
import matplotlib.pyplot as plt

def explain_with_shap(model, X):
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    
    # Calculate SHAP values
    shap_values = explainer(X)

    return shap_values

def plot_shap_summary(shap_values):
    plt.figure()
    shap.summary_plot(shap_values)
    plt.show()

def plot_shap_force(explainer, shap_values, index):
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[index], X.iloc[index])
    plt.show()

def plot_shap_dependence(shap_values, feature_index):
    plt.figure()
    shap.dependence_plot(feature_index, shap_values.values, X)
    plt.show()