# Trading Bots Framework

A collection of automated trading bots built with **Python**, including:
- Rule-based bots (SMA/RSI with TP/SL)
- Machine learning bots (joblib models)
- A webhook listener for TradingView alerts
- A trade copier for logging and summaries
- A backtester for performance evaluation
- A Streamlit dashboard for live monitoring

Exchanges are managed using **ccxt** (Bybit, Coinbase, etc).

---

##  Architecture

TradingView Alerts  →  Webhook Listener (Flask)
↓
Core Bot / ML Bot (Python + ccxt)
↓
Exchange
↓
Trade Copier → Dashboard → Backtester

---

## Setup 

1. Clone the repo:
   ```bash
   git clone <your-repo-url>
   cd TradingBots
2. Create a virtual environment:
   python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3.	Install dependencies:
    - pip install -r requirements.txt

4.	Copy .env.example to .env and fill in:
	•	Exchange API key and secret
	•	Telegram bot token and chat ID
	•	Trading settings (symbol, timeframe, DRY_RUN flag)


Bots
	•	tradingview_webhook_listener.py → Flask server that receives TradingView alerts and executes orders.
	•	core_trading_bot.py → Strategy bot using SMA + RSI with TP/SL logic.
	•	ml_signal_bot.py → Loads trained ML models (.joblib) to generate signals.
	•	trade_copier.py → Copies executed trades into CSV files and produces summaries.
	•	dashboard.py → Streamlit dashboard for monitoring balances, trades, and signals.
	•	backtester.py → Runs historical backtests and calculates metrics.

 Machine Learning

Training script (training/train_model.py) allows you to:
	•	Download OHLCV data
	•	Build features (returns, SMA, RSI, volume)
	•	Train models with time-series cross-validation
	•	Export models to models/*.joblib


⚠️ Disclaimer
	•	This project is for educational purposes only.
	•	Always use DRY_RUN=True or exchange testnet before live trading.
	•	Crypto trading is risky — use at your own responsibility

