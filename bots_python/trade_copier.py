# bots_python/trade_copier.py
# Trade Copier Bot
# Listens to signals/trades from another source (file/webhook) and replicates them on exchange
# Clean version for portfolio: .env-based config, DRY_RUN/testnet-ready

import os
import time
import json
import requests
import ccxt
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIG ======
EXCHANGE_NAME = os.getenv("EXCHANGE", "bybit")
TESTNET       = os.getenv("TESTNET", "true").lower() == "true"
API_KEY       = os.getenv("API_KEY", "")
API_SECRET    = os.getenv("API_SECRET", "")

# Source of trades: can be a local file or an API endpoint
SIGNAL_FILE   = os.getenv("SIGNAL_FILE", "signals.json")
POLL_SECONDS  = int(os.getenv("POLL_SECONDS", "30"))
DRY_RUN       = os.getenv("DRY_RUN", "true").lower() == "true"

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

def read_signals() -> list:
    """Read signals from JSON file (or switch to API if needed)."""
    if not os.path.exists(SIGNAL_FILE):
        return []
    try:
        with open(SIGNAL_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def place_order(ex, symbol: str, side: str, amount: float):
    if DRY_RUN:
        print(f"[DRY-RUN] {side.upper()} {symbol} amount={amount}")
        return {"dry_run": True, "symbol": symbol, "side": side, "amount": amount}

    if side == "buy":
        return ex.create_order(symbol, "market", "buy", amount)
    elif side == "sell":
        return ex.create_order(symbol, "market", "sell", amount)
    else:
        raise ValueError("Unsupported side")

def run():
    ex = get_exchange()
    print(f"✅ Trade Copier started (TESTNET={TESTNET}, DRY_RUN={DRY_RUN})")
    while True:
        signals = read_signals()
        for sig in signals:
            symbol = sig.get("symbol", "BTC/USDT")
            side   = sig.get("side", "").lower()
            amount = float(sig.get("amount", 0))

            if side not in ["buy", "sell"] or amount <= 0:
                continue

            try:
                res = place_order(ex, symbol, side, amount)
                print(f"Executed {side.upper()} {symbol} amount={amount} | res={res}")
            except Exception as e:
                print(f"❌ Error executing {side} {symbol}: {e}")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run()
