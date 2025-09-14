"""
Name: multi_asset_model_bot.py
Purpose: Multi-asset trading bot that loads PRE-TRAINED models for inference (no training here).

What it does (transparent):
  - Loads one pre-trained model per asset (e.g., BTC, ETH, SOL) from /models.
  - Fetches recent OHLCV data per asset, builds the SAME features used at training time.
  - Gets a prediction per asset and executes trades via ccxt (BUY fixed size, SELL full balance).
  - Sends optional Telegram notifications.
  - Supports DRY_RUN to simulate without placing real orders.

What it does NOT do:
  - It does NOT train any model. Models must be trained separately and saved to /models.

Environment (.env) expected:
  API_KEY=<exchange_api_key>
  API_SECRET=<exchange_api_secret>
  EXCHANGE=bybit                       # e.g., bybit, binance, kraken, coinbase
  TELEGRAM_TOKEN=<telegram_bot_token>  # optional
  TELEGRAM_CHAT_ID=<chat_id>           # optional
  # Symbols to trade (comma-separated, must match models & feature pipeline):
  SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
  # Map models to symbols (comma-separated same length as SYMBOLS):
  MODEL_PATHS=models/BTC_USDT.joblib,models/ETH_USDT.joblib,models/SOL_USDT.joblib
  TIMEFRAME=1h
  FETCH_LIMIT=200
  TRADE_SIZE=0.001                     # base amount per BUY (per asset)
  DRY_RUN=True                         # True|False

Notes:
  - Logic preserved simple and explicit for interviews/portfolio review.
  - If your original feature set differs, adjust build_features() to match training pipeline.
"""

import os
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from joblib import load
import requests

# =========================
# Config / Environment
# =========================
API_KEY        = os.getenv("API_KEY")
API_SECRET     = os.getenv("API_SECRET")
EXCHANGE_NAME  = os.getenv("EXCHANGE", "bybit")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS_CSV    = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT")
MODEL_PATHS_CSV= os.getenv("MODEL_PATHS", "models/BTC_USDT.joblib,models/ETH_USDT.joblib,models/SOL_USDT.joblib")

TIMEFRAME      = os.getenv("TIMEFRAME", "1h")
FETCH_LIMIT    = int(os.getenv("FETCH_LIMIT", "200"))
TRADE_SIZE     = float(os.getenv("TRADE_SIZE", "0.001"))
DRY_RUN        = os.getenv("DRY_RUN", "True").lower() == "true"

SYMBOLS: List[str]     = [s.strip() for s in SYMBOLS_CSV.split(",") if s.strip()]
MODEL_PATHS: List[str] = [p.strip() for p in MODEL_PATHS_CSV.split(",") if p.strip()]

if len(SYMBOLS) != len(MODEL_PATHS):
    raise ValueError("SYMBOLS and MODEL_PATHS must have the same number of items.")

# =========================
# Utilities
# =========================
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

def get_exchange(name: str):
    """Create a ccxt exchange instance from EXCHANGE env (no hard-coded secrets)."""
    cls = getattr(ccxt, name.lower(), None)
    if cls is None:
        raise ValueError(f"Unsupported exchange: {name}")
    ex = cls({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    return ex

# =========================
# Data / Features
# =========================
def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep this consistent with the training pipeline.
    Example minimal features: ret_1, sma_ratio (20/50), rsi_14, vol_norm.
    """
    out = df.copy()

    # Returns
    out["ret_1"] = out["close"].pct_change()

    # SMA ratio
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["sma_ratio"] = out["sma_20"] / out["sma_50"] - 1

    # RSI(14)
    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (up / (down.replace(0, np.nan)))))
    out["rsi_14"] = rsi.fillna(50)

    # Normalized volume
    out["vol_norm"] = out["volume"] / out["volume"].rolling(20).mean()

    out = out.dropna().copy()
    return out

FEATURE_COLS = ["ret_1", "sma_ratio", "rsi_14", "vol_norm"]

# =========================
# Core (Inference + Execution)
# =========================
def load_models_for_symbols(symbols: List[str], paths: List[str]) -> Dict[str, any]:
    models = {}
    for sym, path in zip(symbols, paths):
        if not os.path.exists(path):
            log(f"❌ Model not found for {sym}: {path}")
            tg(f"❌ Model not found for {sym}: {path}")
            continue
        try:
            models[sym] = load(path)
            log(f"Loaded model for {sym}: {path}")
        except Exception as e:
            log(f"❌ Failed to load model for {sym}: {e}")
            tg(f"❌ Failed to load model for {sym}: {e}")
    return models

def predict_signal(model, X_row: pd.DataFrame) -> Dict[str, float]:
    """
    Returns dict with prediction and optional probability.
    Assumes binary classifier where 1=buy, 0=otherwise (sell/hold per your convention).
    """
    result = {"pred": None, "proba": None}
    try:
        pred = int(model.predict(X_row)[0])
        result["pred"] = pred
        if hasattr(model, "predict_proba"):
            result["proba"] = float(model.predict_proba(X_row)[0, 1])
    except Exception as e:
        log(f"❌ Prediction failed: {e}")
    return result

def execute_trade(ex, symbol: str, pred: int):
    """
    BUY if pred==1 with fixed TRADE_SIZE.
    SELL all base balance if pred!=1 (simple explicit logic).
    Respects DRY_RUN.
    """
    base = symbol.split("/")[0]

    if DRY_RUN:
        if pred == 1:
            log(f"[DRY_RUN] Would BUY {TRADE_SIZE} {base} on {symbol}")
            tg(f"[DRY_RUN] BUY {TRADE_SIZE} {base} {symbol}")
        else:
            log(f"[DRY_RUN] Would SELL full {base} balance on {symbol}")
            tg(f"[DRY_RUN] SELL full {base} {symbol}")
        return

    try:
        if pred == 1:
            order = ex.create_market_buy_order(symbol, TRADE_SIZE)
            log(f"✅ BUY executed {symbol} → {order}")
            tg(f"✅ BUY {TRADE_SIZE} {base} {symbol}")
        else:
            bal = ex.fetch_balance()
            amount = bal["total"].get(base, 0)
            if amount and amount > 0:
                order = ex.create_market_sell_order(symbol, amount)
                log(f"✅ SELL executed {symbol} → {order}")
                tg(f"✅ SELL {amount} {base} {symbol}")
            else:
                log(f"⚠️ No {base} balance to sell for {symbol}")
                tg(f"⚠️ No {base} balance to sell for {symbol}")
    except Exception as e:
        log(f"❌ Trade execution failed for {symbol}: {e}")
        tg(f"❌ Trade execution failed for {symbol}: {e}")

def run_for_symbol(ex, symbol: str, model) -> None:
    # 1) Fetch data & features
    try:
        df = fetch_ohlcv_df(ex, symbol, TIMEFRAME, FETCH_LIMIT)
        feat = build_features(df)
        latest = feat.iloc[-1:]
        X = latest[FEATURE_COLS]
    except Exception as e:
        log(f"❌ Feature build failed for {symbol}: {e}")
        tg(f"❌ Feature build failed for {symbol}: {e}")
        return

    # 2) Predict
    res = predict_signal(model, X)
    pred, proba = res["pred"], res["proba"]
    if pred is None:
        log(f"⚠️ No prediction for {symbol}")
        tg(f"⚠️ No prediction for {symbol}")
        return

    proba_str = f" | p={round(proba,3)}" if proba is not None else ""
    log(f"🤖 {symbol} prediction: {pred}{proba_str}")
    tg(f"🤖 {symbol} prediction: {pred}{proba_str}")

    # 3) Execute
    execute_trade(ex, symbol, pred)

def main():
    if not API_KEY or not API_SECRET:
        log("[WARN] API_KEY/API_SECRET not set. Use DRY_RUN=True for testing; live orders will fail.")

    ex = get_exchange(EXCHANGE_NAME)
    models = load_models_for_symbols(SYMBOLS, MODEL_PATHS)

    # Run once per symbol (explicit & interview-friendly)
    for sym in SYMBOLS:
        model = models.get(sym)
        if model is None:
            log(f"⏭️ Skipping {sym}: no model loaded.")
            continue
        run_for_symbol(ex, sym, model)

if __name__ == "__main__":
    main()
