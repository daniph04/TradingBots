# bots_python/codigofinal.py
# Core trading bot (polling mode) using ccxt + pandas
# Strategy: SMA crossover with optional RSI filter
# Clean for portfolio: .env-based config, DRY_RUN and TESTNET defaults

import os
import time
import traceback
from typing import List

import ccxt
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

# ====== ENV CONFIG ======
EXCHANGE_NAME      = os.getenv("EXCHANGE", "bybit")        # e.g. bybit
TESTNET            = os.getenv("TESTNET", "true").lower() == "true"
API_KEY            = os.getenv("API_KEY", "")
API_SECRET         = os.getenv("API_SECRET", "")
SYMBOLS            = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT").split(",")]
TIMEFRAME          = os.getenv("TIMEFRAME", "1h")
CANDLES_LIMIT      = int(os.getenv("CANDLES_LIMIT", "300"))

SMA_FAST           = int(os.getenv("SMA_FAST", "10"))
SMA_SLOW           = int(os.getenv("SMA_SLOW", "30"))

USE_RSI            = os.getenv("USE_RSI", "true").lower() == "true"
RSI_LENGTH         = int(os.getenv("RSI_LENGTH", "14"))
RSI_OVERBOUGHT     = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD       = float(os.getenv("RSI_OVERSOLD", "30"))

# Risk / sizing (spot demo, not financial advice)
ACCOUNT_CURRENCY   = os.getenv("ACCOUNT_CURRENCY", "USDT")
RISK_PCT_PER_TRADE = float(os.getenv("RISK_PCT_PER_TRADE", "0.02"))   # 2% per trade (demo)
MAX_NOTIONAL_USD   = float(os.getenv("MAX_NOTIONAL_USD", "300"))      # cap per trade (demo)
MIN_NOTIONAL_USD   = float(os.getenv("MIN_NOTIONAL_USD", "25"))       # avoid dust trades

POLL_SECONDS       = int(os.getenv("POLL_SECONDS", "60"))
DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"

# Telegram (optional)
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT      = os.getenv("TELEGRAM_CHAT", "")


# ====== UTIL ======
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT, "text": text}, timeout=10)
    except Exception:
        pass


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


def fetch_ohlcv_df(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_fast"] = df["close"].rolling(SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(SMA_SLOW).mean()
    if USE_RSI:
        # Simple RSI implementation (Wilder's can be added later)
        delta = df["close"].diff()
        gain = (delta.clip(lower=0)).rolling(RSI_LENGTH).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_LENGTH).mean()
        rs = gain / loss.replace(0, 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
    else:
        df["rsi"] = None
    return df


def last_signal(df: pd.DataFrame):
    """Return 'buy' | 'sell' | 'hold' with a basic SMA crossover + optional RSI filter."""
    row = df.iloc[-1]
    prev = df.iloc[-2]

    # No signals if indicators not ready
    if pd.isna(row["sma_fast"]) or pd.isna(row["sma_slow"]) or pd.isna(prev["sma_fast"]) or pd.isna(prev["sma_slow"]):
        return "hold"

    golden_cross = prev["sma_fast"] <= prev["sma_slow"] and row["sma_fast"] > row["sma_slow"]
    death_cross  = prev["sma_fast"] >= prev["sma_slow"] and row["sma_fast"] < row["sma_slow"]

    if golden_cross:
        if not USE_RSI or (row["rsi"] is not None and row["rsi"] <= RSI_OVERBOUGHT):
            return "buy"
    if death_cross:
        if not USE_RSI or (row["rsi"] is not None and row["rsi"] >= RSI_OVERSOLD):
            return "sell"
    return "hold"


def account_notional_usd(ex) -> float:
    bal = ex.fetch_balance()
    # Try USDT first, fallback to USD
    usdt = bal["total"].get("USDT", 0)
    usd  = bal["total"].get("USD", 0)
    return float(usdt or usd or 0)


def quote_precision_from_market(market: dict) -> int:
    # Helps avoid precision errors; keep it simple here
    amount_prec = market.get("precision", {}).get("amount", 6)
    return int(amount_prec)


def calc_order_size(ex, symbol: str, price: float) -> float:
    # Risk-based sizing capped by MAX_NOTIONAL_USD
    acct_usd = account_notional_usd(ex)
    if acct_usd <= 0:
        return 0.0
    target = min(acct_usd * RISK_PCT_PER_TRADE, MAX_NOTIONAL_USD)
    if target < MIN_NOTIONAL_USD:
        return 0.0
    qty = target / price
    mkt = ex.load_markets()[symbol]
    decimals = quote_precision_from_market(mkt)
    return float(round(qty, decimals))


def place_market_order(ex, symbol: str, side: str, amount: float):
    if DRY_RUN:
        msg = f"[DRY-RUN] {side.upper()} {symbol} amount={amount}"
        print(msg); send_telegram(msg)
        return {"dry_run": True, "side": side, "symbol": symbol, "amount": amount}

    if amount <= 0:
        raise ValueError("Amount <= 0; sizing failed or insufficient balance.")

    if side == "buy":
        return ex.create_order(symbol, "market", "buy", amount)
    elif side == "sell":
        return ex.create_order(symbol, "market", "sell", amount)
    else:
        raise ValueError("Unsupported side")


def step_for_symbol(ex, symbol: str):
    df = fetch_ohlcv_df(ex, symbol, TIMEFRAME, CANDLES_LIMIT)
    df = add_indicators(df)
    sig = last_signal(df)
    ts  = df["time"].iloc[-1]

    if sig == "hold":
        print(f"[{ts}] {symbol}: HOLD")
        return

    ticker = ex.fetch_ticker(symbol)
    price = float(ticker["last"])

    if sig == "buy":
        amount = calc_order_size(ex, symbol, price)
        res = place_market_order(ex, symbol, "buy", amount)
        print(f"[{ts}] BUY {symbol} @ ~{price} | res={res}")
        send_telegram(f"✅ BUY {symbol} ~{price} | amt={amount}")
        return

    if sig == "sell":
        # Simplified: sell up to the quote we might hold; in real bot, track positions/wallet of base asset
        # Here we try to sell the base asset balance (spot demo)
        base = symbol.split("/")[0]
        bal = ex.fetch_balance()
        base_amt = float(bal["total"].get(base, 0))
        if base_amt <= 0:
            print(f"[{ts}] {symbol}: SELL signal but no base balance.")
            return
        res = place_market_order(ex, symbol, "sell", base_amt)
        print(f"[{ts}] SELL {symbol} @ ~{price} | res={res}")
        send_telegram(f"✅ SELL {symbol} ~{price} | amt={base_amt}")
        return


def run(symbols: List[str]):
    ex = get_exchange()
    print(f"✅ Core bot started (TESTNET={TESTNET}, DRY_RUN={DRY_RUN}) on {EXCHANGE_NAME}")
    send_telegram("✅ Core trading bot started.")
    while True:
        try:
            for sym in symbols:
                step_for_symbol(ex, sym)
            time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            err = f"❌ Loop error: {e}\n{traceback.format_exc(limit=1)}"
            print(err)
            send_telegram(err)
            time.sleep(5)


if __name__ == "__main__":
    run(SYMBOLS)
