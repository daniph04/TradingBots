# bots_python/backtester.py
# Simple backtester for SMA/RSI rules or ML model signals (if model available)
# Saves results (equity/trades/metrics) under results/

import os, math, json
import numpy as np
import pandas as pd
import ccxt
from datetime import datetime
from dotenv import load_dotenv

try:
    import joblib
except Exception:
    joblib = None  # ML optional

load_dotenv()

EXCHANGE   = os.getenv("EXCHANGE", "bybit")
SYMBOL     = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME  = os.getenv("TIMEFRAME", "1h")
LIMIT      = int(os.getenv("LIMIT", "3000"))

USE_RSI    = os.getenv("USE_RSI", "true").lower() == "true"
SMA_FAST   = int(os.getenv("SMA_FAST", "10"))
SMA_SLOW   = int(os.getenv("SMA_SLOW", "30"))
RSI_LEN    = int(os.getenv("RSI_LENGTH", "14"))
RSI_OB     = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OS     = float(os.getenv("RSI_OVERSOLD", "30"))

# ML
MODELS_DIR   = os.getenv("MODELS_DIR", "models")
MODEL_SUFFIX = os.getenv("MODEL_SUFFIX", ".joblib")
THRESH_BUY   = float(os.getenv("THRESHOLD_BUY", "0.55"))
THRESH_SELL  = float(os.getenv("THRESHOLD_SELL", "0.45"))

# Backtest params
INITIAL_EQUITY = float(os.getenv("BT_INITIAL_EQUITY", "10000"))
FEE_BPS        = float(os.getenv("BT_FEE_BPS", "10"))  # 10 bps = 0.10% per trade
RESULTS_DIR    = os.getenv("RESULTS_DIR", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def fetch_df():
    ex = getattr(ccxt, EXCHANGE)({"enableRateLimit": True, "options": {"defaultType":"spot"}})
    ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_fast"] = df["close"].rolling(SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(SMA_SLOW).mean()
    if USE_RSI:
        delta = df["close"].diff()
        gain = (delta.clip(lower=0)).rolling(RSI_LEN).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_LEN).mean()
        rs = gain / (loss.replace(0, 1e-9))
        df["rsi"] = 100 - (100 / (1 + rs))
    return df

FEATURES = ["ret_1","ret_3","ret_6","sma10","sma30","sma_ratio","rsi","v_ratio"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma30"] = df["close"].rolling(30).mean()
    df["sma_ratio"] = df["sma10"] / df["sma30"]
    delta = df["close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["v_sma20"] = df["volume"].rolling(20).mean()
    df["v_ratio"] = df["volume"] / (df["v_sma20"] + 1e-9)
    return df

def load_model_for_symbol(symbol: str):
    if joblib is None:
        return None
    path = os.path.join(MODELS_DIR, symbol.replace("/","_") + MODEL_SUFFIX)
    return joblib.load(path) if os.path.exists(path) else None

def signal_rule(row, prev):
    # SMA crossover + RSI filter
    if pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]) or pd.isna(prev["sma_fast"]) or pd.isna(prev["sma_slow"]):
        return "hold"
    golden = prev["sma_fast"] <= prev["sma_slow"] and row["sma_fast"] > row["sma_slow"]
    death  = prev["sma_fast"] >= prev["sma_slow"] and row["sma_fast"] < row["sma_slow"]
    if golden:
        if (not USE_RSI) or row.get("rsi", 50) <= RSI_OB:
            return "buy"
    if death:
        if (not USE_RSI) or row.get("rsi", 50) >= RSI_OS:
            return "sell"
    return "hold"

def prob_up_ml(model, row):
    if model is None:
        return None
    X = np.array([row[FEATURES].values], dtype=float)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])
    y = float(model.predict(X)[0])
    return max(0.0, min(1.0, y))

def metrics_from_equity(eq: pd.Series, freq_per_year=365*24):  # para 1h: 8760
    rets = eq.pct_change().fillna(0)
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    # CAGR aproximado
    years = len(eq) / freq_per_year
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/max(years,1e-9)) - 1 if years > 0 else np.nan
    # Sharpe (simple, sin rf)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(freq_per_year) if rets.std() > 0 else np.nan
    # Max Drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max - 1)
    mdd = dd.min()
    return {
        "total_return": float(total_return),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(mdd)
    }

def backtest(df: pd.DataFrame, use_ml=True):
    df = df.copy()
    df = add_indicators(df)
    if use_ml:
        df = add_features(df)

    model = load_model_for_symbol(SYMBOL) if use_ml else None
    use_ml = use_ml and model is not None

    equity = INITIAL_EQUITY
    base_qty = 0.0
    fee = FEE_BPS / 10000.0
    eq_curve = []
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row["close"]

        # Calc current equity assuming mark-to-market
        eq_now = equity + base_qty * price
        eq_curve.append([row["time"], eq_now])

        # Generate signal
        sig = "hold"
        if use_ml:
            pr = prob_up_ml(model, row)
            if pr is not None:
                if pr >= THRESH_BUY:  sig = "buy"
                elif pr <= THRESH_SELL: sig = "sell"
        else:
            sig = signal_rule(row, prev)

        # Execute (simple: all-in/out on signals)
        if sig == "buy" and equity > 0:
            qty = (equity * (1 - fee)) / price
            base_qty += qty
            trades.append({"time": row["time"], "side": "buy", "price": float(price), "qty": float(qty)})
            equity = 0

        elif sig == "sell" and base_qty > 0:
            proceeds = base_qty * price * (1 - fee)
            equity += proceeds
            trades.append({"time": row["time"], "side": "sell", "price": float(price), "qty": float(base_qty)})
            base_qty = 0.0

    # Close position at the end
    if base_qty > 0:
        price = df.iloc[-1]["close"]
        proceeds = base_qty * price * (1 - fee)
        equity += proceeds
        trades.append({"time": df.iloc[-1]["time"], "side": "sell", "price": float(price), "qty": float(base_qty)})
        base_qty = 0.0
        eq_curve.append([df.iloc[-1]["time"], equity])

    eq_df = pd.DataFrame(eq_curve, columns=["time","equity"]).dropna()
    # Winrate simple
    wins = [t for t in trades if t["side"]=="sell"]  # cada venta cierra una operación
    winrate = None
    if len(wins) > 1:
        # reconstruye PnL por trade simple (aprox)
        pnl = []
        buy_px = None
        for t in trades:
            if t["side"]=="buy": buy_px = t["price"]
            if t["side"]=="sell" and buy_px:
                pnl.append((t["price"]-buy_px)/buy_px)
                buy_px = None
        if pnl:
            winrate = sum(1 for x in pnl if x>0)/len(pnl)

    m = metrics_from_equity(eq_df["equity"], freq_per_year=8760)  # 1h
    m["winrate"] = float(winrate) if winrate is not None else None
    return eq_df, trades, m

def main():
    df = fetch_df()
    eq_rule, trades_rule, metrics_rule = backtest(df, use_ml=False)
    eq_ml, trades_ml, metrics_ml = backtest(df, use_ml=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{SYMBOL.replace('/','_')}_{TIMEFRAME}_{ts}"

    # Save
    eq_rule.to_csv(os.path.join(RESULTS_DIR, f"{base}_equity_rule.csv"), index=False)
    pd.DataFrame(trades_rule).to_csv(os.path.join(RESULTS_DIR, f"{base}_trades_rule.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{base}_metrics_rule.json"), "w") as f:
        json.dump(metrics_rule, f, indent=2)

    eq_ml.to_csv(os.path.join(RESULTS_DIR, f"{base}_equity_ml.csv"), index=False)
    pd.DataFrame(trades_ml).to_csv(os.path.join(RESULTS_DIR, f"{base}_trades_ml.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, f"{base}_metrics_ml.json"), "w") as f:
        json.dump(metrics_ml, f, indent=2)

    print("✅ Saved results under", RESULTS_DIR)
    print("Rule-based metrics:", metrics_rule)
    print("ML-based metrics:", metrics_ml)

if __name__ == "__main__":
    main()
