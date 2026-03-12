import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker="AAPL", period="2y"):
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume', 'High', 'Low', 'Open']]
    df.dropna(inplace=True)
    print(f"Total rows: {len(df)}")
    print(df.tail())
    return df

if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
