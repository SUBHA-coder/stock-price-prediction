import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_linear_regression(train_data, test_data):
    # Prepare training and testing datasets
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data
    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    y_test = test_data

    # Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(train_data)), y_train, label="Training Data", color="blue")
    plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), y_test, label="Actual Prices", color="green")
    plt.plot(np.arange(len(train_data), len(train_data) + len(test_data)), predictions, label="Predicted Prices", color="red")
    plt.legend()
    plt.title("Linear Regression: Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Normalized Price")
    plt.show()

    return model

# Example usage
if __name__ == "__main__":
    from preprocess_data import preprocess_stock_data

    file_path = "AAPL_data.csv"  # Replace with the path to your CSV file
    train_data, test_data, scaler = preprocess_stock_data(file_path)
    linear_model = train_linear_regression(train_data, test_data)
