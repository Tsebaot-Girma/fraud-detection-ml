# scripts/model_selection.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM

def get_models():
    """
    Define and return a list of models to be used for training.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
    }

    return models

def get_deep_learning_models(input_shape):
    """
    Define and return deep learning models (CNN, RNN, LSTM).
    """
    # CNN Model
    cnn_model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # RNN Model
    rnn_model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])

    # LSTM Model
    lstm_model = Sequential([
        LSTM(50, input_shape=input_shape, return_sequences=True),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])

    return {
        "CNN": cnn_model,
        "RNN": rnn_model,
        "LSTM": lstm_model
    }