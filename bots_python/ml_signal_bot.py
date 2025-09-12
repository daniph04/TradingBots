# bots_python/ml_signal_bot.py
# ML-driven signal bot: loads per-symbol models (.joblib), builds features from OHLCV,
# predicts up-move probability, and maps to buy/sell/hold with thresholds.
# Portfolio-ready: .env config, DRY_RUN & TESTNET defaults.

import os, time, traceback
from typing import Dict, List
import joblib
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

load_dotenv()

# ====== ENV ======
EXCHANGE_NAME   = os.getenv("EXCHANGE", "bybit")
TESTNET         = os.getenv("TESTNET", "true").lower() == "true"
API_KEY         = os.getenv("API_KEY", "")
API_SECRET      = os.getenv("API_SECRET", "")

SYMBOLS         = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")]
TIMEFRAME       = os.getenv("TIMEFRAME", "1h")
CANDLES_LIMIT   = int(os.getenv("CANDLES_LIMIT", "400"))

# Models
MODELS_DIR      = os.getenv("MODELS_DIR", "models")
MODEL_SUFFIX    = os.getenv("MODEL_SUFFIX", ".joblib")    # e.g. ".joblib"
# Thresholds
THRESH_BUY      = float(os.getenv("THRESHOLD_BUY", "0.55"))
THRESH_SELL     = float(os.getenv("THRESHOLD_SELL", "0.45"))

# Sizing / runtime
MAX_NOTIONAL_USD = float(os.getenv("MAX_NOTIONAL_USD", "300"))
MIN_NOTIONAL_USD = float(os.getenv("MIN_NOTIONAL_USD", "25"))
DRY_RUN          = os.getenv("DRY_RUN", "true").lower() == "true"
POLL_SECONDS     = int(os.getenv("POLL_SECONDS", "120"))

def get_exchange():
    if EXCHANGE_NAME == "bybit":
        ex = ccxt.bybit({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        ex.set_sandbox_mode(TESTNET)
        return ex
    raise ValueError(f"Exchange '{EXCHANGE_NAME}' not supported in this demo.")

def load_models(symbols: List[str]) -> Dict[str, object]:
    """Load one model per symbol. Expected filename: e.g. BTC_USDT.joblib for BTC/USDT."""
    models = {}
    for sym in symbols:
        fname = sym.replace("/", "_") + MODEL_SUFFIX  # "BTC/USDT" -> "BTC_USDT.joblib"
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[sym] = joblib.load(path)
            print(f"✅ Loaded model for {sym}: {path}")
        else:
            print(f"⚠️ Model not found for {sym}: {path}")
    return models

def fetch_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic price returns
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_6"]  = df["close"].pct_change(6)
    # SMAs
    df["sma10"]  = df["close"].rolling(10).mean()
    df["sma30"]  = df["close"].rolling(30).mean()
    df["sma_ratio"] = df["sma10"] / df["sma30"]
    # RSI (simple)
    delta = df["close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi"] = 100 - (100 / (1 + rs))
    # Volume features
    df["v_sma20"] = df["volume"].rolling(20).mean()
    df["v_ratio"] = df["volume"] / (df["v_sma20"] + 1e-9)
    return df

FEATURES = ["ret_1","ret_3","ret_6","sma10","sma30","sma_ratio","rsi","v_ratio"]

def prob_up(model, row: pd.Series) -> float:
    X = np.array([row[FEATURES].values], dtype=float)
    # Works for sklearn, XGBoost (sklearn API) etc.
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        # assume binary cls with proba of class 1 = up
        return float(p[1])
    # Fallback: decision_function or predict -> convert to pseudo-prob
    y = model.predict(X)[0]
    return float(max(0.0, min(1.0, y)))  # crude fallback

def account_notional_usd(ex) -> float:
    bal = ex.fetch_balance()
    usdt = float(bal["total"].get("USDT", 0) or 0)
    usd  = float(bal["total"].get("USD", 0) or 0)
    return usdt if usdt > 0 else usd

def calc_amount(ex, symbol: str, price: float) -> float:
    notional = min(account_notional_usd(ex) * 0.02, MAX_NOTIONAL_USD)
    if notional < MIN_NOTIONAL_USD:
        return 0.0
    qty = notional / price
    mkt = ex.load_markets()[symbol]
    prec = mkt.get("precision", {}).get("amount", 6)
    return float(round(qty, int(prec)))

def place_order(ex, symbol: str, side: str, amount: float):
    if DRY_RUN:
        print(f"[DRY-RUN] {side.upper()} {symbol} amount={amount}")
        return {"dry_run": True, "side": side, "symbol": symbol, "amount": amount}
    if amount <= 0:
        raise ValueError("Amount <= 0")
    return ex.create_order(symbol, "market", side, amount)

def step_symbol(ex, model, symbol: str):
    df = fetch_df(ex, symbol, TIMEFRAME, CANDLES_LIMIT)
    df = add_features(df)
    row = df.dropna().iloc[-1]  # ensure features valid
    p_up = prob_up(model, row)

    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"])
    ts = row["time"]

    if p_up >= THRESH_BUY:
        amt = calc_amount(ex, symbol, price)
        res = place_order(ex, symbol, "buy", amt)
        print(f"[{ts}] {symbol} p_up={p_up:.3f} → BUY | price~{price} | res={res}")
    elif p_up <= THRESH_SELL:
        # Try to sell base balance (spot)
        base = symbol.split("/")[0]
        bal = ex.fetch_balance()
        base_amt = float(bal["total"].get(base, 0) or 0)
        if base_amt > 0:
            res = place_order(ex, symbol, "sell", base_amt)
            print(f"[{ts}] {symbol} p_up={p_up:.3f} → SELL | price~{price} | amt={base_amt} | res={res}")
        else:
            print(f"[{ts}] {symbol} p_up={p_up:.3f} → SELL signal but base balance=0")
    else:
        print(f"[{ts}] {symbol} p_up={p_up:.3f} → HOLD")

def run():
    ex = get_exchange()
    models = load_models(SYMBOLS)
    print(f"✅ ML bot started (TESTNET={TESTNET}, DRY_RUN={DRY_RUN}) with symbols: {SYMBOLS}")
    while True:
        try:
            for sym in SYMBOLS:
                model = models.get(sym)
                if model is None:
                    print(f"⏭️  Skipping {sym}: no model loaded.")
                    continue
                step_symbol(ex, model, sym)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"❌ Loop error: {e}\n{traceback.format_exc(limit=1)}")
            time.sleep(5)

if __name__ == "__main__":
    run()
