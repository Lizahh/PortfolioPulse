# main.py
from data.fetch_data import fetch_data
from models.feature_selector import select_features
from models.ppo_model import train_model
import pandas as pd

def run_pipeline():
    df = fetch_data("AAPL")
    df = df.dropna()
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    
    X_new, features = select_features(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Target']], 'Target')
    print(f"Selected Features: {features}")
    
    train_model()

if __name__ == "__main__":
    run_pipeline()
