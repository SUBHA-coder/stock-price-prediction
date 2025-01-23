from flask import Flask, render_template, request
from linear_regression import train_linear_regression
from lstm_model import train_lstm_model, reshape_data
from preprocess_data import preprocess_stock_data
import numpy as np

app = Flask(__name__)

# Load and preprocess data
FILE_PATH = "AAPL_data.csv"  # Replace with your data file
TIME_STEPS = 60

# Preprocess the data
train_data, test_data, scaler = preprocess_stock_data(FILE_PATH)

# Train the models
print("Training Linear Regression model...")
linear_model = train_linear_regression(train_data, test_data)

print("Training LSTM model...")
lstm_model = train_lstm_model(train_data, test_data, time_steps=TIME_STEPS)

@app.route("/")
def index():
    """Render the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests and render the results."""
    try:
        # Get user input (number of days to predict)
        days = int(request.form.get("days", 1))

        # Linear Regression Predictions
        X_test_linear = list(range(len(train_data), len(train_data) + days))
        linear_predictions = linear_model.predict([[i] for i in X_test_linear])

        # LSTM Predictions
        last_data = test_data[-TIME_STEPS:]  # Get the last `time_steps` data points
        lstm_predictions = []

        for _ in range(days):
            reshaped_data = np.expand_dims(last_data.reshape(1, TIME_STEPS, 1), axis=2)
            pred = lstm_model.predict(reshaped_data)[0, 0]
            lstm_predictions.append(pred)
            last_data = np.append(last_data[1:], pred)  # Update with the new prediction

        # Inverse scale the predictions
        linear_predictions = scaler.inverse_transform(np.array(linear_predictions).reshape(-1, 1)).flatten()
        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

        # Render the results
        return render_template("results.html", 
                               days=days, 
                               linear_predictions=linear_predictions, 
                               lstm_predictions=lstm_predictions)

    except Exception as e:
        return f"An error occurred: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)
