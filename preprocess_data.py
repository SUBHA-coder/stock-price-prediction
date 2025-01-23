import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_stock_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}.")
    
    # Drop rows with non-numeric values in 'Close' (clean the data)
    data = data[pd.to_numeric(data['Close'], errors='coerce').notnull()]
    data['Close'] = data['Close'].astype(float)
    
    # Display the cleaned data
    print(data.head())
    
    # Handle missing values (if any)
    data.dropna(inplace=True)
    
    # Extract the 'Close' prices for prediction
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data to a range of [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Split the data into train and test sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    print(f"Data preprocessed. Train data size: {len(train_data)}, Test data size: {len(test_data)}")
    return train_data, test_data, scaler

# Example usage
if __name__ == "__main__":
    file_path = "AAPL_data.csv"  
    train_data, test_data, scaler = preprocess_stock_data(file_path)
