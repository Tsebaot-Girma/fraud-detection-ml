# scripts/lime_explainability.py

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def explain_with_lime(model, X, instance, feature_names):
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        class_names=['Non-Fraud', 'Fraud'],
        mode='classification'
    )
    
    # Explain the instance
    exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)

    return exp

def plot_lime_explanation(exp):
    exp.show_in_notebook(show_table=True, show_all=False)

def plot_lime_plot(exp):
    plt.figure()
    exp.as_pyplot_figure()
    plt.show()