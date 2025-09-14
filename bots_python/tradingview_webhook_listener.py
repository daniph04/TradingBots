"""
Name: tradingview_webhook_listener.py
Purpose: Receive TradingView alerts (JSON) and execute market orders via ccxt (Bybit unified).
Inputs: HTTP POST /webhook with JSON: {"action": "buy"|"sell", "symbol": "BTC/USDT" or "BTC"}
Outputs: Places market orders and sends Telegram notifications. Includes a 2h heartbeat.
Security: API keys and tokens are read from environment variables (no hard-coded secrets).
Notes: Logic intentionally preserved from the original script (fixed $400 buy, sell full balance).
"""

import os
import ccxt
from flask import Flask, request, jsonify
import requests
import threading
import time
from datetime import datetime

# ========= Secure config from environment (NO hard-coded secrets) =========
API_KEY        = os.getenv("API_KEY")                 # Bybit API key
API_SECRET     = os.getenv("API_SECRET")              # Bybit API secret
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")          # Telegram bot token
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")        # Telegram chat id (string)

# Basic safety checks (do not change trading logic)
if not API_KEY or not API_SECRET:
    print("[WARN] API_KEY/API_SECRET not set. The exchange client will fail on live calls.")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
    print("[WARN] TELEGRAM_TOKEN/TELEGRAM_CHAT_ID not set. Telegram notifications will be skipped.")

app = Flask(__name__)

# Exchange client (Bybit unified). Logic preserved.
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'unified'
    }
})

def send_telegram_message(message: str):
    """Send a text message to Telegram chat. Silently skip if not configured."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        # No tokens configured → skip without raising.
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] [TG-SKIP] {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {'chat_id': TELEGRAM_CHAT, 'text': message}
    try:
        response = requests.post(url, data=data, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

def log_local_message(message: str):
    """Local stdout logger with UTC timestamp."""
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def heartbeat():
    """Send a heartbeat to Telegram every 2 hours (logic preserved)."""
    while True:
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        msg = f"🟢 Bot active | Last check: {now}"
        send_telegram_message(msg)
        log_local_message(msg)
        time.sleep(7200)  # 2 hours

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for TradingView alerts."""
    # Parse JSON payload robustly (logic preserved)
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            import json
            data = json.loads(request.data.decode('utf-8'))
        log_local_message(f"Received webhook data: {data}")
    except Exception as e:
        log_local_message(f"Error parsing webhook data: {e}")
        return jsonify({'error': 'Invalid JSON'}), 400

    # Validate required fields
    if not data or 'action' not in data or 'symbol' not in data:
        log_local_message("Invalid data received in webhook.")
        return jsonify({'error': 'Invalid data'}), 400

    action = str(data['action']).lower()
    symbol = str(data['symbol']).upper()

    # If the symbol doesn't include a quote, default to USDT (logic preserved)
    if '/' not in symbol:
        symbol += '/USDT'

    try:
        if action == 'buy':
            # === LOGIC PRESERVED: require >= $400 USD available and buy market for $400 ===
            balance = exchange.fetch_balance()
            usd_balance = balance['total'].get('USD', 0)
            if usd_balance < 400:
                msg = f"❌ Not enough USD to buy {symbol}. Needed: 400 USD | Available: {usd_balance} USD"
                send_telegram_message(msg)
                log_local_message(msg)
                return jsonify({'error': msg}), 400

            amount_usd = 400
            log_local_message(f"Placing BUY market for {symbol} with notional ${amount_usd}")
            # Note: original logic used create_market_buy_order(symbol, amount) with 'amount' as 400.
            # On some exchanges, ccxt expects 'amount' in base units. We keep the call identical.
            order = exchange.create_market_buy_order(symbol, amount_usd)
            msg = f"✅ BUY executed: {symbol} for 400 USD.\n🧾 Details: {order}"
            send_telegram_message(msg)
            log_local_message(msg)
            return jsonify({'status': 'buy order placed', 'order': order}), 200

        elif action == 'sell':
            # === LOGIC PRESERVED: sell full available balance of base asset ===
            balance = exchange.fetch_balance()
            base_asset = symbol.split('/')[0]
            base_balance = balance['total'].get(base_asset, 0)
            if base_balance > 0:
                log_local_message(f"Placing SELL market for {symbol} amount={base_balance}")
                order = exchange.create_market_sell_order(symbol, base_balance)
                msg = f"✅ SELL executed: {base_balance} {base_asset} ({symbol}).\n🧾 Details: {order}"
                send_telegram_message(msg)
                log_local_message(msg)
                return jsonify({'status': 'sell order placed', 'order': order}), 200
            else:
                msg = f"⚠️ No balance available in {base_asset} to sell."
                log_local_message(msg)
                send_telegram_message(msg)
                return jsonify({'error': msg}), 400

        else:
            msg = f"❓ Unknown action: {action}"
            log_local_message(msg)
            send_telegram_message(msg)
            return jsonify({'error': 'Unknown action'}), 400

    except Exception as e:
        error_msg = f"❌ Error executing {action.upper()} for {symbol}:\n{str(e)}"
        send_telegram_message(error_msg)
        log_local_message(error_msg)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start heartbeat thread and boot the Flask app (logic preserved)
    threading.Thread(target=heartbeat, daemon=True).start()
    msg = "✅ Webhook listener started and ready to receive alerts."
    send_telegram_message(msg)
    log_local_message(msg)
    app.run(host='0.0.0.0', port=5000)
