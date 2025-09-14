# Bots Overview

This file explains what each of my trading bots does.  
I keep it simple: each bot has a clear role, either using indicators, pre-trained models, or external signals.

---

## core_trading_bot.py
A simple trading bot based on **SMA and RSI**. It checks market trends with moving averages and momentum with RSI. If conditions are met, it buys; if not, it exits. This is my baseline bot for testing and showing how my risk rules work.

---

## ml_inference_trading_bot.py
This bot uses a **pre-trained ML model** to make trading decisions. It doesn’t train the model itself; it only loads it and predicts whether to buy or exit. I use it for single assets (like BTC) when I want to test machine learning signals in real time.

---

## multi_asset_model_bot.py
A more advanced version of the inference bot. It runs on **multiple assets (BTC, ETH, SOL)** at the same time, each with its own pre-trained model. It decides independently for each coin whether to buy or close positions. This shows how I scale the ML idea to more markets.

---

## signal_copier.py
This bot **copies external signals** (like from Telegram). When it sees a “Buy” or “Sell” message, it executes the same trade on the exchange. It’s useful to automate manual signals and test different strategies from outside sources.

---

## tradingview_webhook_listener.py
A Flask server that listens to **TradingView alerts**. When TradingView sends a signal (for example, from my Pine Script strategies), this bot turns it into a real order on the exchange. It also logs activity and can send Telegram messages to confirm trades.

---

## legacy_multi_asset_bot.py
An **older prototype** of the multi-asset bot. It also loads pre-trained models for BTC, ETH, and SOL, but the structure is less clean. I keep it to show my earlier work and how the multi-asset bot evolved into the better version above.

---

## dashboard.py
A small **Streamlit dashboard** I use to visualize my trades, balances, and performance. It’s not a trading bot, but a tool to review results and make analysis easier.

