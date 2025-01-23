import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def reshape_data(data, time_steps=60):
    """
    Reshape the data for LSTM.
    Each input sequence will contain `time_steps` data points.
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def train_lstm_model(train_data, test_data, time_steps=60, epochs=20, batch_size=32):
    # Reshape the training and testing data
    X_train, y_train = reshape_data(train_data, time_steps)
    X_test, y_test = reshape_data(test_data, time_steps)
    
    # Expand dimensions for LSTM input format [samples, time_steps, features]
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"LSTM Mean Squared Error (MSE): {mse}")

    # Plot the actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual Prices", color="green")
    plt.plot(range(len(predictions)), predictions, label="Predicted Prices", color="red")
    plt.legend()
    plt.title("LSTM: Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.show()

    return model

# Example usage
if __name__ == "__main__":
    from preprocess_data import preprocess_stock_data

    file_path = "AAPL_data.csv"  # Replace with the path to your CSV file
    train_data, test_data, scaler = preprocess_stock_data(file_path)

    lstm_model = train_lstm_model(train_data, test_data, time_steps=60, epochs=20, batch_size=32)
