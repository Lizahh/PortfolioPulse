# PortfolioPulse
Portfolio Optimization using Feature Engineering &amp; Reinforcement Learning (PPO)

<img width="1920" height="1481" alt="image" src="https://github.com/user-attachments/assets/c3999208-c8de-4cc1-b2b4-e30eff0a981b" />


# Overview
PortfolioPulse is an intelligent, data-driven financial portfolio optimization framework that leverages cutting-edge machine learning, feature fusion, and reinforcement learning (PPO) to enhance investment strategies, improve decision-making, and maximize risk-adjusted returns.

Designed as a full-stack research-to-deployment pipeline, this project simulates realistic stock market conditions and applies dynamic learning agents for smarter, adaptive portfolio management.

# Key Highlights
## Feature Fusion

* Advanced feature selection using SelectKBest and domain-specific heuristics

* Reduces noise and dimensionality while maximizing predictive power

* Applies across multi-dimensional, multi-ticker datasets

## Reinforcement Refinement

* Implements Proximal Policy Optimization (PPO) via Stable-Baselines3

* Enables agent to learn optimal trading strategies through trial and error

* Trained within a customized Gym environment built for financial markets

## Real Market Data Simulation

* Pulls real-time historical stock data via yfinance

* Custom reward logic based on Sharpe Ratio, Return, and Risk metrics

* Market wrappers simulate real-world volatility and transaction costs

# Modular Architecture

* fetch_data.py: Pulls and structures raw data

* feature_selector.py: Applies filtering and selection

* ppo_model.py: Trains the RL agent

* env.py: Custom RL environment for portfolio dynamics

* main.py: Unified pipeline execution

  # Tech Stack

| Component         | Tech Used                            |
| ----------------- | ------------------------------------ |
| Language          | Python 3.12                          |
| Data              | `yfinance`, `pandas`, `numpy`        |
| Feature Selection | `sklearn`, `scipy`                   |
| RL Algorithm      | `Stable-Baselines3` - PPO            |
| Environment       | `Gym` (will migrate to `Gymnasium`)  |
| Visualization     | `matplotlib`, `seaborn` *(optional)* |

# How It Works (Simplified Flow)

* Fetch Data: Load stock price data (e.g., AAPL) using yfinance

* Preprocessing & Feature Selection: Drop irrelevant columns, engineer features, and select top predictors

* Custom Gym Environment: Set up action space, reward function, transaction costs, slippage, etc.

* Train Agent: Train PPO agent to maximize returns over simulated episodes

* Monitor Training: Visual logs of training loss, KL divergence, entropy, etc.

* Save Model: Export trained policy for inference or simulation

#  Future Improvements

* Migrate to Gymnasium for updated API compliance

*  Add reward visualization over episodes (matplotlib or Streamlit)

*  Add backtesting & benchmarking module

*  Deploy on a minimal Streamlit UI to allow live simulations

*  Allow multiple tickers for portfolio balancing

*  Try advanced RL algorithms (A2C, DDPG, SAC)

# Data Sources

All data is fetched using:

* Yahoo Finance via yfinance

e.g., yfinance.download("AAPL", start="2015-01-01", end="2023-01-01")

You can easily swap the ticker symbol for any supported equity, ETF, crypto, etc.

 Run the Full Pipeline
 ```bash
 python main.py
```

Trained model will be saved as ppo_portfolio_model.zip.

✨ Author
Aliza Mustafa

ML Engineer | Data Science Enthusiast

🔗 [GitHub](https://github.com/Lizahh) | 💼 [LinkedIn](https://www.linkedin.com/in/aliza-mustafa-ml-engineer/)


 

