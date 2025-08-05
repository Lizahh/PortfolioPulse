
import yfinance as yf
import pandas as pd

def fetch_data(ticker="AAPL", start="2015-01-01", end="2023-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(f"z{ticker}.csv")
    return df

if __name__ == "__main__":
    fetch_data()