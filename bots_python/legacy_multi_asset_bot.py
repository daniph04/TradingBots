"""
Name: legacy_multi_asset_bot.py
Purpose: Early prototype of a multi-asset trading bot that loads PRE-TRAINED models
         (BTC/ETH/SOL) and executes trades based on predictions.

Transparency:
  - No training here, only inference with models saved in /models.
  - Logic preserved from the original script; this file only adds docs and secure env.

Environment (.env):
  API_KEY=...
  API_SECRET=...
  EXCHANGE=bybit              # or binance/kraken/coinbase
  TELEGRAM_TOKEN=...          # optional
  TELEGRAM_CHAT_ID=...        # optional
  TIMEFRAME=1h
  FETCH_LIMIT=200
  TRADE_SIZE=0.001
  DRY_RUN=True
  # Models for each asset you use in this legacy script:
  MODEL_BTC=models/BTC_USDT.joblib
  MODEL_ETH=models/ETH_USDT.joblib
  MODEL_SOL=models/SOL_USDT.joblib
"""

import os
from datetime import datetime

# ===== Secure env (NO hard-coded secrets) =====
API_KEY        = os.getenv("API_KEY")
API_SECRET     = os.getenv("API_SECRET")
EXCHANGE_NAME  = os.getenv("EXCHANGE", "bybit")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
TIMEFRAME      = os.getenv("TIMEFRAME", "1h")
FETCH_LIMIT    = int(os.getenv("FETCH_LIMIT", "200"))
TRADE_SIZE     = float(os.getenv("TRADE_SIZE", "0.001"))
DRY_RUN        = os.getenv("DRY_RUN", "True").lower() == "true"

MODEL_BTC_PATH = os.getenv("MODEL_BTC", "models/BTC_USDT.joblib")
MODEL_ETH_PATH = os.getenv("MODEL_ETH", "models/ETH_USDT.joblib")
MODEL_SOL_PATH = os.getenv("MODEL_SOL", "models/SOL_USDT.joblib")

def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# NOTA: Deja toda tu lógica tal cual está a continuación.
# - Donde uses claves/tokens directos, sustitúyelos por las variables de arriba.
# - Si envías Telegram, envuélvelo con una función que use TELEGRAM_TOKEN/CHAT.
# - No toco tu lógica de features/predicción/órdenes.
