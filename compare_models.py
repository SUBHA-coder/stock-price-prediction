import matplotlib.pyplot as plt
from linear_regression import train_linear_regression
from lstm_model import train_lstm_model, reshape_data
from preprocess_data import preprocess_stock_data
import numpy as np

def compare_models(file_path, time_steps=60):
    # Preprocess the stock data
    train_data, test_data, scaler = preprocess_stock_data(file_path)

    # Train Linear Regression
    print("\nTraining Linear Regression Model...")
    linear_model = train_linear_regression(train_data, test_data)

    # Train LSTM
    print("\nTraining LSTM Model...")
    lstm_model = train_lstm_model(train_data, test_data, time_steps=time_steps)

    # Linear Regression Predictions
    X_test_linear = list(range(len(train_data), len(train_data) + len(test_data)))
    linear_predictions = linear_model.predict([[i] for i in X_test_linear])

    # LSTM Predictions
    X_test_lstm, y_test_lstm = reshape_data(test_data, time_steps)
    X_test_lstm_reshaped = np.expand_dims(X_test_lstm, axis=2)  # Ensure the correct shape for LSTM input
    lstm_predictions = lstm_model.predict(X_test_lstm_reshaped)

    # Comparison Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(test_data)), test_data, label="Actual Prices", color="green")
    plt.plot(range(len(test_data)), linear_predictions, label="Linear Regression Predictions", color="blue")
    plt.plot(range(len(y_test_lstm)), lstm_predictions, label="LSTM Predictions", color="red")
    plt.legend()
    plt.title("Comparison of Linear Regression and LSTM Models")
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.show()

# Example usage
if __name__ == "__main__":
    file_path = "AAPL_data.csv"  # Replace with the path to your CSV file
    compare_models(file_path, time_steps=60)
