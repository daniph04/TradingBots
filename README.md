# TradingBots

A collection of algorithmic trading bots built in **Python** and **TradingView (Pine Script)**.  
Includes multiple strategies, pre-trained models, backtesting utilities, and a Streamlit dashboard.

---

## 📂 Project Structure

### bots_python/
- **core_trading_bot.py** – Basic bot that trades using SMA/RSI conditions. A simple baseline I use for testing.  
- **ml_inference_trading_bot.py** – Loads pre-trained ML models to generate buy/sell signals. No training inside the bot.  
- **multi_asset_model_bot.py** – Extended ML bot that supports BTC, ETH, and SOL models, trading them in parallel.  
- **signal_copier.py** – Listens to external signals (e.g., Telegram) and mirrors them into real exchange trades.  
- **tradingview_webhook_listener.py** – Flask server that listens to TradingView alerts and executes orders automatically.  
- **legacy_multi_asset_bot.py** – Early prototype of multi-asset logic. I keep it for reference.  
- **dashboard.py** – Streamlit dashboard for visualizing trades, balances, and performance.  

### strategies_tradingview/  
Custom Pine Script strategies (long/short logic, alerts) that send signals to the Python bots.  

### models/  
Pre-trained ML models used by `ml_inference_trading_bot.py` and `multi_asset_model_bot.py`. (Ignored by Git).  

### results/  
Backtest results, logs, and evaluation files. (Ignored by Git).  

### Config files  
- **.env.example** – Template for environment variables (API keys, settings).  
- **requirements.txt** – Python dependencies.  
- **.gitignore** – Ignore secrets, models, and logs.  
- **README.md** – Project documentation.  
- **LICENSE** – MIT License.

----

## ⚙️ Installation

Clone this repository and install the required dependencies:


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

📘 Detailed explanations of each bot are available in [Bots Overview](bots_overview.md).

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

