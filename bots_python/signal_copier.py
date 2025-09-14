"""
Name: signal_copier.py
Purpose: Listen to Telegram trading signals and mirror them as market orders on Alpaca.
Scope:
  - Uses Telethon to read messages from a specific Telegram channel or chat.
  - Parses simple "buy/sell" signals and places corresponding MARKET orders via Alpaca.
  - Logs all parsed signals and executions to a CSV file for later reporting.
  - Supports DRY_RUN mode to simulate without sending real orders.

⚠️ Transparency:
  - This script DOES NOT change your trading logic. It only sanitizes secrets (env vars),
    adds clear English comments, and wraps I/O safely.
  - Adjust the `parse_signal()` function if your signal format is different.

Environment (.env):
  TG_API_ID=123456
  TG_API_HASH=your_telegram_api_hash
  TG_SESSION=signal_copier_session
  TG_CHANNEL=your_channel_username_or_id   # e.g., @signals_channel (string) or numeric ID
  ALPACA_API_KEY=your_alpaca_key
  ALPACA_SECRET_KEY=your_alpaca_secret
  ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
  DRY_RUN=True
  DEFAULT_QTY=1
  DEFAULT_SIDE=buy                         # fallback if side not in text (rare)
  ALLOWED_SIDES=buy,sell
  LOG_PATH=logs/signal_copier_trades.csv
"""

import os
import re
import csv
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from telethon import TelegramClient, events
from telethon.errors import RPCError

from alpaca_trade_api.rest import REST, TimeFrame, APIError


# =========================
# Secure configuration
# =========================
TG_API_ID   = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH = os.getenv("TG_API_HASH")
TG_SESSION  = os.getenv("TG_SESSION", "signal_copier_session")
TG_CHANNEL  = os.getenv("TG_CHANNEL")  # username (e.g., @channel) or numeric ID as string

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

DRY_RUN       = os.getenv("DRY_RUN", "True").lower() == "true"
DEFAULT_QTY   = float(os.getenv("DEFAULT_QTY", "1"))
DEFAULT_SIDE  = os.getenv("DEFAULT_SIDE", "buy").lower()
ALLOWED_SIDES = set(s.strip().lower() for s in os.getenv("ALLOWED_SIDES", "buy,sell").split(","))
LOG_PATH      = os.getenv("LOG_PATH", "logs/signal_copier_trades.csv")

# Ensure log directory exists
Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)


# =========================
# Utilities
# =========================
def log(msg: str) -> None:
    print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def write_csv_row(path: str, row: Dict[str, Any]) -> None:
    """Append a dict row to CSV (creates header if file doesn't exist)."""
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# =========================
# Signal parsing
# =========================
SIGNAL_REGEXES = [
    # Examples:
    # "BUY TSLA 2", "Sell NVDA 5", "buy AAPL", "sell META 1.5"
    re.compile(r"\b(?P<side>buy|sell)\b\s+(?P<symbol>[A-Z]{1,6})(?:\s+(?P<qty>\d+(\.\d+)?))?", re.I),
    # "Long TSLA 2" / "Short BTC 0.01" (mapped to buy/sell spot by convention)
    re.compile(r"\b(?P<side>long|short)\b\s+(?P<symbol>[A-Z]{1,6})(?:\s+(?P<qty>\d+(\.\d+)?))?", re.I),
]

def parse_signal(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a Telegram message into a normalized signal:
      {"side": "buy"|"sell", "symbol": "TSLA", "qty": float}
    Falls back to DEFAULT_QTY and DEFAULT_SIDE if needed.
    """
    if not text:
        return None

    for rx in SIGNAL_REGEXES:
        m = rx.search(text)
        if m:
            side = m.group("side").lower()
            symbol = m.group("symbol").upper() if m.group("symbol") else None
            qty = m.group("qty")
            qty = float(qty) if qty else DEFAULT_QTY

            # Normalize side: map long->buy, short->sell
            if side == "long":
                side = "buy"
            elif side == "short":
                side = "sell"

            if side not in ALLOWED_SIDES:
                return None
            if not symbol:
                return None

            return {"side": side, "symbol": symbol, "qty": qty}

    # If nothing matched, return None
    return None


# =========================
# Alpaca client
# =========================
def get_alpaca() -> REST:
    if not (ALPACA_KEY and ALPACA_SECRET):
        log("[WARN] Alpaca credentials not set. Use DRY_RUN=True.")
    return REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE)


async def execute_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a parsed signal on Alpaca (MARKET DAY order).
    In DRY_RUN mode, do not place real orders—just log.
    """
    side = signal["side"]
    symbol = signal["symbol"]
    qty = float(signal["qty"])

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    result: Dict[str, Any] = {
        "timestamp_utc": ts,
        "side": side,
        "symbol": symbol,
        "qty": qty,
        "status": "parsed",
        "order_id": "",
        "error": "",
        "dry_run": DRY_RUN,
        "raw_text": signal.get("raw_text", ""),
    }

    try:
        if DRY_RUN:
            log(f"[DRY_RUN] Would {side.upper()} {qty} {symbol}")
            result["status"] = "simulated"
            write_csv_row(LOG_PATH, result)
            return result

        api = get_alpaca()
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )
        result["status"] = "filled_or_submitted"
        result["order_id"] = getattr(order, "id", "") or getattr(order, "client_order_id", "")
        log(f"✅ {side.upper()} {qty} {symbol} | order_id={result['order_id']}")
        write_csv_row(LOG_PATH, result)
        return result

    except APIError as e:
        result["status"] = "error"
        result["error"] = str(e)
        log(f"❌ Alpaca API error: {e}")
        write_csv_row(LOG_PATH, result)
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        log(f"❌ Execution error: {e}")
        write_csv_row(LOG_PATH, result)
        return result


# =========================
# Telegram listener
# =========================
async def main() -> None:
    if not TG_API_ID or not TG_API_HASH or not TG_CHANNEL:
        log("❌ Telegram credentials or channel not configured. Check .env")
        return

    client = TelegramClient(TG_SESSION, TG_API_ID, TG_API_HASH)
    await client.start()
    log("🔌 Telegram client started (signal copier).")
    log(f"Listening on channel/chat: {TG_CHANNEL}")

    @client.on(events.NewMessage(chats=TG_CHANNEL))
    async def handler(event):
        try:
            text = event.raw_text or ""
            log(f"📨 New message: {text}")

            parsed = parse_signal(text)
            if not parsed:
                log("↩️ No valid signal found in message. Skipping.")
                return

            # Attach raw text for logging transparency
            parsed["raw_text"] = text

            # Execute on Alpaca (or simulate in DRY_RUN)
            await execute_signal(parsed)

        except RPCError as e:
            log(f"❌ Telegram RPC error: {e}")
        except Exception as e:
            log(f"❌ Handler error: {e}")

    # Keep the client running forever
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("👋 Stopped by user.")
