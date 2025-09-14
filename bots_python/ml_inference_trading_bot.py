"""
Name: ml_inference_trading_bot.py
Purpose: Machine Learning *inference* trading bot for Kraken using ccxt.
What it does:
  - Loads a PRE-TRAINED ML model (.joblib) from /models (this script does NOT train).
  - Fetches recent OHLCV data from Kraken, builds the same features used at training time.
  - Gets a prediction (e.g., 1 = buy, 0 = sell/hold) and executes trades accordingly.
  - Sends optional Telegram notifications.

Transparency:
  - This bot performs ONLY inference (prediction). Training must be done separately
    (e.g., training/train_model.py) and the exported model placed in /models.

Security:
  - No hard-coded secrets. All keys/tokens/settings come from environment variables.
  - Supports DRY_RUN to simulate without placing real orders.

Environment variables (set these in your .env):
  API_KEY               = <kraken_api_key>
  API_SECRET            = <kraken_api_secret>
  TELEGRAM_TOKEN        = <telegram_bot_token>              # optional
  TELEGRAM_CHAT_ID      = <telegram_chat_id>                # optional
  SYMBOL                = BTC/USDT                          # default
  TIMEFRAME             = 1h                                # default
  MODEL_PATH            = models/BTC_USDT.joblib            # default
  TRADE_SIZE            = 0.001                             # default base amount for BUY
  DRY_RUN               = True                              # True|False
  FETCH_LIMIT           = 200                               # candles to fetch for features
"""

import os
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import load
import requests

# ======== Secure config from environment (NO hard-coded secrets) ========
API_KEY        = os.getenv("API_KEY")
API_SECRET     = os.getenv("API_SECRET")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL         = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME      = os.getenv("TIMEFRAME", "1h")
MODEL_PATH     = os.getenv("MODEL_PATH", "models/BTC_USDT.joblib")

# Trading controls
TRADE_SIZE     = float(os.getenv("TRADE_SIZE", "0.001"))   # base amount to BUY
DRY_RUN        = os.getenv("DRY_RUN", "True").lower() == "true"
FETCH_LIMIT    = int(os.getenv("FETCH_LIMIT", "200"))

# ======== Exchange client (Kraken) ========
exchange = ccxt.kraken({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})

# ======== Utilities ========
def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def tg(msg: str) -> None:
    """Send Telegram message if configured; otherwise just log skip."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log(f"[TG-SKIP] {msg}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": msg},
            timeout=10,
        )
    except Exception as e:
        log(f"[TG-ERR] {e}")

# ======== Feature engineering (must match training pipeline) ========
def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep this consistent with the feature set used during training.
    Example features: 1-bar return, SMA ratio, RSI(14), normalized volume.
    """
    out = df.copy()

    # 1) Returns
    out["ret_1"] = out["close"].pct_change()

    # 2) SMA ratio (20 vs 50)
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["sma_ratio"] = out["sma_20"] / out["sma_50"] - 1

    # 3) RSI(14)
    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (up / (down.replace(0, np.nan)))))
    out["rsi_14"] = rsi.fillna(50)

    # 4) Normalized volume
    out["vol_norm"] = out["volume"] / out["volume"].rolling(20).mean()

    out = out.dropna().copy()
    return out

# ======== Core bot (inference + execution) ========
def run_once():
    # 1) Load model
    if not os.path.exists(MODEL_PATH):
        log(f"❌ Model not found: {MODEL_PATH}")
        tg(f"❌ Model not found: {MODEL_PATH}")
        return

    try:
        model = load(MODEL_PATH)
        log(f"Loaded model: {MODEL_PATH}")
        tg(f"📂 Loaded model: {MODEL_PATH}")
    except Exception as e:
        log(f"❌ Failed to load model: {e}")
        tg(f"❌ Failed to load model: {e}")
        return

    # 2) Build features on latest data
    try:
        df = fetch_ohlcv_df(SYMBOL, TIMEFRAME, FETCH_LIMIT)
        feat = build_features(df)
        latest_row = feat.iloc[-1:]
        # Keep columns identical to training set
        X = latest_row[["ret_1", "sma_ratio", "rsi_14", "vol_norm"]]
    except Exception as e:
        log(f"❌ Failed to build features: {e}")
        tg(f"❌ Failed to build features: {e}")
        return

    # 3) Predict
    try:
        pred = int(model.predict(X)[0])
        # Optional: proba if supported
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
        log(f"Prediction for {SYMBOL}: {pred} (proba={proba})")
        tg(f"🤖 Prediction {SYMBOL}: {pred}{' | p=' + str(round(proba,3)) if proba is not None else ''}")
    except Exception as e:
        log(f"❌ Prediction failed: {e}")
        tg(f"❌ Prediction failed: {e}")
        return

    # 4) Execute trade (BUY if pred==1; else SELL all base if available)
    base = SYMBOL.split("/")[0]

    try:
        if DRY_RUN:
            if pred == 1:
                log(f"[DRY_RUN] Would BUY {TRADE_SIZE} {base} on {SYMBOL}")
                tg(f"[DRY_RUN] BUY {TRADE_SIZE} {base} {SYMBOL}")
            else:
                log(f"[DRY_RUN] Would SELL full {base} balance on {SYMBOL}")
                tg(f"[DRY_RUN] SELL full {base} {SYMBOL}")
            return

        # Live mode:
        if pred == 1:
            order = exchange.create_market_buy_order(SYMBOL, TRADE_SIZE)
            log(f"✅ BUY executed: {order}")
            tg(f"✅ BUY {TRADE_SIZE} {base} {SYMBOL}")
        else:
            bal = exchange.fetch_balance()
            amount = bal["total"].get(base, 0)
            if amount and amount > 0:
                order = exchange.create_market_sell_order(SYMBOL, amount)
                log(f"✅ SELL executed: {order}")
                tg(f"✅ SELL {amount} {base} {SYMBOL}")
            else:
                log(f"⚠️ No {base} balance to sell.")
                tg(f"⚠️ No {base} balance to sell.")
    except Exception as e:
        log(f"❌ Trade execution failed: {e}")
        tg(f"❌ Trade execution failed: {e}")

if __name__ == "__main__":
    # Single run for clarity; you can schedule externally (cron/systemd) if needed.
    # Keeping it simple and explicit for interviews and reproducibility.
    if not API_KEY or not API_SECRET:
        log("[WARN] API_KEY/API_SECRET not set. Live orders will fail. Use DRY_RUN=True for testing.")
    run_once()
