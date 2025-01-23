import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    # Fetch data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save to CSV
    stock_data.to_csv(f"{ticker}_data.csv")
    print(f"Data saved to {ticker}_data.csv")
    
    # Plot the closing prices
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    ticker_symbol = "AAPL"  
    start_date = "2020-01-01"
    end_date = "2022-01-01"
    fetch_stock_data(ticker_symbol, start_date, end_date)
