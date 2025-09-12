# bots_python/listener_tradingview.py
# Minimal TradingView -> Exchange listener using Flask + ccxt (testnet friendly)
# Clean for portfolio: uses .env, no hardcoded secrets, dry-run by default.

import os, json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import ccxt
import requests

load_dotenv()

# ====== Config (from .env) ======
EXCHANGE     = os.getenv("EXCHANGE", "bybit")       # e.g., bybit
TESTNET      = os.getenv("TESTNET", "true").lower() == "true"
API_KEY      = os.getenv("API_KEY", "")
API_SECRET   = os.getenv("API_SECRET", "")
SYMBOL_DEF   = os.getenv("SYMBOL", "BTC/USDT")
ORDER_SIZE   = float(os.getenv("ORDER_SIZE", "0.001"))  # example size
DRY_RUN      = os.getenv("DRY_RUN", "true").lower() == "true"

# Telegram (optional)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT", "")

def send_telegram(msg: str):
    """Send a Telegram message if token/chat are configured."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT, "text": msg}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")

def get_exchange():
    """Init ccxt exchange (spot demo)."""
    if EXCHANGE == "bybit":
        ex = ccxt.bybit({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        ex.set_sandbox_mode(TESTNET)
        return ex
    raise ValueError(f"Exchange '{EXCHANGE}' not supported in this demo.")

def normalize_symbol(sym: str) -> str:
    """Accepts 'BINANCE:BTCUSDT', 'BYBIT:BTCUSDT', 'BTCUSDT', or 'BTC/USDT' -> returns 'BTC/USDT'."""
    if not sym:
        return SYMBOL_DEF
    s = sym.replace("BINANCE:", "").replace("BYBIT:", "").upper()
    if "/" not in s and s.endswith("USDT"):
        s = s[:-4] + "/USDT"
    return s

app = Flask(__name__)

@app.post("/webhook")
def webhook():
    """
    Expected JSON from TradingView alertcondition:
      {"action":"open|close","side":"long|short","symbol":"BTC/USDT"}
    """
    try:
        data = request.get_json(force=True)
        if not data:
            data = json.loads(request.data.decode("utf-8"))
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    action = str(data.get("action", "")).lower()
    side   = str(data.get("side", "")).lower()
    symbol = normalize_symbol(data.get("symbol", SYMBOL_DEF))

    if action not in {"open", "close"} or side not in {"long", "short"}:
        return jsonify({"ok": False, "error": "Invalid action/side"}), 400

    # Spot demo: implement only long open/close as market BUY/SELL.
    # Shorts typically require derivatives/margin; we just acknowledge them.
    if DRY_RUN:
        msg = f"[DRY-RUN] {action.upper()} {side.upper()} {symbol} size={ORDER_SIZE}"
        print(msg); send_telegram(msg)
        return jsonify({"ok": True, "dry_run": True, "action": action, "side": side, "symbol": symbol})

    try:
        ex = get_exchange()
        if action == "open" and side == "long":
            order = ex.create_order(symbol, "market", "buy", ORDER_SIZE)
            msg = f"✅ OPEN LONG {symbol} size={ORDER_SIZE}\n{order}"
            print(msg); send_telegram(msg)
            return jsonify({"ok": True, "order": order})

        if action == "close" and side == "long":
            order = ex.create_order(symbol, "market", "sell", ORDER_SIZE)
            msg = f"✅ CLOSE LONG {symbol} size={ORDER_SIZE}\n{order}"
            print(msg); send_telegram(msg)
            return jsonify({"ok": True, "order": order})

        # Acknowledge shorts in spot demo
        msg = f"ℹ️ Received {action} {side} for {symbol} (short not implemented in spot demo)"
        print(msg); send_telegram(msg)
        return jsonify({"ok": True, "note": "short_not_implemented_in_spot_demo"})

    except Exception as e:
        err = f"❌ Error executing {action} {side} for {symbol}: {e}"
        print(err); send_telegram(err)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"ok": True, "status": "alive"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"✅ Listener running on 0.0.0.0:{port} (TESTNET={TESTNET}, DRY_RUN={DRY_RUN})")
    send_telegram("✅ TradingView listener started.")
    app.run(host="0.0.0.0", port=port)
