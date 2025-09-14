# TradingBots

A collection of algorithmic trading bots built in **Python** and **TradingView (Pine Script)**.  
Includes multiple strategies, pre-trained models, backtesting utilities, and a Streamlit dashboard.

---

## 📂 Project Structure
TradingBots/
│
├── bots_python/              # Python trading bots
│   ├── core_trading_bot.py   # Basic trading bot with SMA/RSI logic
│   ├── signal_copier.py      # Copies signals from Telegram and executes trades
│   ├── tradingview_webhook_listener.py # Listens to TradingView alerts via webhook
│   ├── multi_asset_model_bot.py        # Prototype with pre-trained models for BTC, ETH, SOL
│   ├── ml_inference_trading_bot.py     # Loads ML models for prediction-based signals
│   ├── legacy_multi_asset_bot.py       # Older version kept for reference
│   └── .env.example          # Example environment variables (API keys, settings)
│
├── models/                   # Store ML models (ignored by Git)
│
├── results/                  # Store backtest results and logs (ignored by Git)
│
├── strategies_tradingview/   # Pine Script strategies for TradingView
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore secrets, models, and logs
├── LICENSE                   # MIT License
└── README.md                 # Project documentation

---

## ⚙️ Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/daniph04/TradingBots.git
cd TradingBots
pip install -r requirements.txt

- Environment Setup
	1.	Copy the example file:
cp bots_python/.env.example .env

	2.	Open .env and add your exchange API keys and settings:
	•	Exchange (Bybit, Binance, etc.)
	•	API key and secret
	•	Trading symbols
	•	Risk management settings
	•	Optional: Telegram bot token for notifications

⚠️ Never commit your real .env file — it is already ignored in .gitignore.

Running the Bots

Each bot can be run independently. Examples:
# Core trading bot (SMA + RSI strategy)
python bots_python/core_trading_bot.py

# TradingView webhook listener (executes trades from alerts)
python bots_python/tradingview_webhook_listener.py

# Telegram signal copier (executes trades from Telegram signals)
python bots_python/signal_copier.py

# Multi-asset model bot (BTC, ETH, SOL with pre-trained models)
python bots_python/multi_asset_model_bot.py

----

TradingView Strategies

Inside strategies_tradingview/, you’ll find Pine Script strategies such as:
	•	HHLL Long/Short Strategy
	•	NASDAQ RSI/STOCH Strategy

These scripts can be pasted into TradingView Pine Editor, backtested, and connected to the Python bots via webhook alerts.

Dashboard (Optional)

A Streamlit dashboard (dashboard.py) can be used to visualize:
	•	Active trades
	•	Balance and PnL
	•	Strategy performance

Machine Learning

Some bots (e.g., ml_inference_trading_bot.py, multi_asset_model_bot.py) can load pre-trained ML models from /models/.
Models are ignored in Git and should be trained/stored locally.

Disclaimer

This project is for educational purposes only.
Use at your own risk. Trading cryptocurrencies involves high risk, and you are responsible for your own results.

License:
This project is licensed under the MIT License.

---

