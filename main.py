#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Seller (Alpaca) — partial profit-taking + dust cleanup + safe rounding

Features
- Partial sells by profit rung (TP1/TP2/TP3) to "lock profits and keep the rest running"
- Full close if a position is tiny (value < DUST_CLOSE_VALUE_USD, default $5)
- Fixes 403 "insufficient qty available" via floor rounding + clamp
- Optional Google Sheets logging (service account)
- DRY_RUN by default

Env (most useful)
  BROKER=ALPACA
  APCA_API_KEY_ID / ALPACA_KEY_ID
  APCA_API_SECRET_KEY / ALPACA_SECRET_KEY
  APCA_API_BASE_URL="https://paper-api.alpaca.markets"
  ALPACA_DATA_URL="https://data.alpaca.markets"
  ALPACA_DATA_FEED="iex"  # or "sip" if you have it
  DRY_RUN=1

  # Profit ladder (defaults shown)
  TP1_PCT=8         TP1_SIZE=0.4
  TP2_PCT=15        TP2_SIZE=0.3
  TP3_PCT=25        TP3_SIZE=0.3

  # Other behavior
  DUST_CLOSE_VALUE_USD=5.0        # close full if position value < this
  SELL_SHARE_DECIMALS=6           # broker precision for fractional shares
  MAX_POSITIONS_PER_RUN=999       # safety cap if desired

  # Sheets logging (optional)
  USE_SHEETS_LOG=1
  SHEET_NAME="Trading Log"
  STOCK_LOG_TAB="log"             # columns: Timestamp | Action | Symbol | NotionalUSD | Qty | OrderID | Status | Note
  GOOGLE_CREDS_JSON='{"type":"service_account",...}'
"""
import os, json, time
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import requests

# ========= Config & helpers =========
BROKER = os.getenv("BROKER", "ALPACA").upper()

ALPACA_KEY = (
    os.getenv("ALPACA_KEY_ID")
    or os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY_ID")
    or os.getenv("ALPACA_KEY")
    or ""
)
ALPACA_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_API_SECRET_KEY")
    or os.getenv("ALPACA_SECRET")
    or ""
)
ALPACA_BASE_URL = (
    os.getenv("ALPACA_BASE_URL")
    or os.getenv("APCA_API_BASE_URL")
    or "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL  = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")  # "iex" for free plan, "sip" if you have it

# Behavior
DRY_RUN = os.getenv("DRY_RUN", "1").lower() in ("1","true","yes")
TP1_PCT  = float(os.getenv("TP1_PCT",  "8"))
TP1_SIZE = float(os.getenv("TP1_SIZE", "0.4"))
TP2_PCT  = float(os.getenv("TP2_PCT",  "15"))
TP2_SIZE = float(os.getenv("TP2_SIZE", "0.3"))
TP3_PCT  = float(os.getenv("TP3_PCT",  "25"))
TP3_SIZE = float(os.getenv("TP3_SIZE", "0.3"))

DUST_CLOSE_VALUE_USD = float(os.getenv("DUST_CLOSE_VALUE_USD", "5.0"))
SHARE_DECIMALS       = int(os.getenv("SELL_SHARE_DECIMALS", "6"))
MAX_POS_PER_RUN      = int(os.getenv("MAX_POSITIONS_PER_RUN", "999"))

# Sheets logging
USE_SHEETS_LOG = os.getenv("USE_SHEETS_LOG", "1").lower() in ("1","true","yes")
SHEET_NAME     = os.getenv("SHEET_NAME", "Trading Log")
LOG_TAB        = os.getenv("STOCK_LOG_TAB", "log")
GOOGLE_CREDS_JSON = os.getenv("GOOGLE_CREDS_JSON", "")

getcontext().prec = 28  # high precision for Decimal

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def d(x) -> Decimal:
    return Decimal(str(x))

def round_down_shares(qty: Decimal, decimals: int = SHARE_DECIMALS) -> Decimal:
    """
    Floor to broker's fractional share precision.
    """
    step = Decimal(1).scaleb(-decimals)  # 10^-decimals
    return (qty // step) * step

def clamp_sell_qty(requested_qty: Decimal, available_qty: Decimal, decimals: int = SHARE_DECIMALS) -> Decimal:
    rq = round_down_shares(requested_qty, decimals)
    av = round_down_shares(available_qty, decimals)
    if rq > av:
        rq = av
    return rq

def vlog(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ========= Alpaca HTTP =========
def alp_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def http_get(url: str, params: Dict[str, Any] = None, timeout: int = 20):
    r = requests.get(url, headers=alp_headers(), params=params or {}, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()

def http_post(url: str, data: Dict[str, Any], timeout: int = 20):
    r = requests.post(url, headers=alp_headers(), data=json.dumps(data), timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r.json()

def http_delete(url: str, timeout: int = 20):
    r = requests.delete(url, headers=alp_headers(), timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"DELETE {url} -> {r.status_code} {r.text}")
    return r.json() if r.text else {"status": "ok"}

# ========= Broker data/actions =========
def fetch_positions() -> List[Dict[str, Any]]:
    """
    Alpaca: GET /v2/positions
    Returns list with: symbol, qty, avg_entry_price, market_value, side
    Only long equity positions are considered here.
    """
    url = f"{ALPACA_BASE_URL}/v2/positions"
    try:
        j = http_get(url)
    except Exception as e:
        raise RuntimeError(f"positions fetch failed: {e}")

    out = []
    for p in j:
        try:
            if str(p.get("side", "")).lower() != "long":
                continue
            symbol = p["symbol"]
            qty = float(p["qty"])
            if qty <= 0:
                continue
            avg = float(p["avg_entry_price"])
            mval = float(p.get("market_value", 0) or 0)
            out.append({
                "symbol": symbol,
                "qty": qty,
                "avg_cost": avg,
                "market_value": mval,
            })
        except Exception:
            continue
    return out

def fetch_last_price(symbol: str) -> float:
    """
    Alpaca Data v2: latest trade price
    """
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": ALPACA_DATA_FEED}
    j = http_get(url, params=params)
    return float(j["trade"]["p"])

def close_position(symbol: str) -> Tuple[str, str]:
    """
    Alpaca close-position endpoint handles full liquidation & rounding.
    """
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    if DRY_RUN:
        return "DRYRUN", "dry-run"
    try:
        j = http_delete(url)
        return (j.get("id", "closed"), j.get("status", "closed"))
    except Exception as e:
        return ("", f"error: {e}")

def sell_market(symbol: str, qty: Decimal) -> Tuple[str, str]:
    """
    Submit a market sell with safe fractional rounding.
    """
    safe_qty = clamp_sell_qty(qty, qty)  # just in case caller didn't already floor
    if safe_qty <= Decimal("0"):
        return "", "qty<=0"

    url = f"{ALPACA_BASE_URL}/v2/orders"
    data = {
        "symbol": symbol,
        "qty": str(safe_qty),   # send as string to preserve decimals
        "side": "sell",
        "type": "market",
        "time_in_force": "day",
    }
    if DRY_RUN:
        return "DRYRUN", "dry-run"
    try:
        j = http_post(url, data)
        return (j.get("id",""), j.get("status","submitted"))
    except Exception as e:
        # One more try shaving one tick if it's an "insufficient qty" style error
        msg = str(e).lower()
        if "insufficient" in msg or "403" in msg:
            step = Decimal(1).scaleb(-SHARE_DECIMALS)
            retry_qty = safe_qty - step
            if retry_qty > Decimal("0"):
                data["qty"] = str(retry_qty)
                j = http_post(url, data)
                return (j.get("id",""), j.get("status","submitted"))
        return "", f"error: {e}"

# ========= Strategy (profit rungs + dust cleanup) =========
def decide_partial_qty(symbol: str, qty: float, avg_cost: float, last: float) -> Tuple[str, Decimal, str]:
    """
    Returns (action, qty_to_sell_decimal, note)
      - If position value < DUST_CLOSE_VALUE_USD => CLOSE_ALL (use close_position)
      - Else compute profit% and sell ONE rung per run (TP3 > TP2 > TP1).
      - If no rung hit => HOLD
    """
    if last <= 0 or qty <= 0:
        return "HOLD", Decimal("0"), "invalid_px_or_qty"

    pos_value = last * qty
    if pos_value < DUST_CLOSE_VALUE_USD:
        return "CLOSE_ALL", Decimal(str(qty)), f"dust_cleanup(value=${pos_value:.2f}<{DUST_CLOSE_VALUE_USD})"

    profit_pct = (last - avg_cost) / avg_cost * 100.0
    # Check higher rung first so we don't sell multiple rungs in one run
    if profit_pct >= TP3_PCT and TP3_SIZE > 0:
        target = Decimal(str(qty)) * Decimal(str(TP3_SIZE))
        return "SELL_PART", target, f"tp3({profit_pct:.2f}%≥{TP3_PCT}%) size={TP3_SIZE}"
    if profit_pct >= TP2_PCT and TP2_SIZE > 0:
        target = Decimal(str(qty)) * Decimal(str(TP2_SIZE))
        return "SELL_PART", target, f"tp2({profit_pct:.2f}%≥{TP2_PCT}%) size={TP2_SIZE}"
    if profit_pct >= TP1_PCT and TP1_SIZE > 0:
        target = Decimal(str(qty)) * Decimal(str(TP1_SIZE))
        return "SELL_PART", target, f"tp1({profit_pct:.2f}%≥{TP1_PCT}%) size={TP1_SIZE}"

    return "HOLD", Decimal("0"), f"hold(profit={profit_pct:.2f}%)"

# ========= Optional Sheets logging =========
def sheets_append(rows: List[List[str]]):
    if not USE_SHEETS_LOG:
        return
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        info = json.loads(GOOGLE_CREDS_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        ws = gc.open(SHEET_NAME).worksheet(LOG_TAB)
        ws.append_rows(rows, value_input_option="USER_ENTERED")
    except Exception as e:
        vlog(f"⚠️ Sheets logging failed: {e}")

def log_row(action: str, symbol: str, notional: float, qty: Decimal, order_id: str, status: str, note: str) -> List[str]:
    return [
        now_iso(),
        action,
        symbol,
        f"{notional:.2f}",
        str(qty),
        order_id or "",
        status or "",
        note or "",
    ]

# ========= Main =========
def main():
    print("🏁 stock-seller starting")
    print(f"ENV: BROKER={BROKER} DRY_RUN={DRY_RUN} BASE_URL={ALPACA_BASE_URL} FEED={ALPACA_DATA_FEED}")

    if BROKER != "ALPACA":
        raise RuntimeError("Only BROKER=ALPACA supported in this version.")

    positions = fetch_positions()
    if not positions:
        print("ℹ️ No long US equity positions. Nothing to do.")
        return

    sell_count = 0
    hold_count = 0
    rows_for_sheet: List[List[str]] = []

    for i, pos in enumerate(positions):
        if i >= MAX_POS_PER_RUN:
            vlog(f"Hit MAX_POSITIONS_PER_RUN={MAX_POS_PER_RUN}, stopping.")
            break

        symbol = pos["symbol"]
        qty    = float(pos["qty"])
        avg    = float(pos["avg_cost"])

        try:
            last = fetch_last_price(symbol)
        except Exception as e:
            vlog(f"⚠️ {symbol}: failed to fetch last price: {e}")
            hold_count += 1
            continue

        action, target_qty, note = decide_partial_qty(symbol, qty, avg, last)

        # Compute notional using last; round display only
        notional = float(target_qty) * last

        if action == "CLOSE_ALL":
            if DRY_RUN:
                order_id, status = "DRYRUN", "dry-run"
            else:
                order_id, status = close_position(symbol)
            vlog(f"[{symbol}] CLOSE_ALL qty={qty:.6f} last=${last:.2f} note={note} -> {status}")
            rows_for_sheet.append(log_row("SELL", symbol, qty*last, Decimal(str(qty)), order_id, status, note))
            sell_count += 1
            continue

        if action == "SELL_PART":
            # Safe rounding & clamp to available
            available = d(qty)
            target    = clamp_sell_qty(target_qty, available)
            # If due to rounding target==available and you prefer to avoid accidental full exit,
            # you could shave 1 tick. Here, we allow it (user wanted to sometimes fully exit via ladder).
            if target <= Decimal("0"):
                vlog(f"[{symbol}] SELL_PART computed qty<=0 after clamp; skipping. note={note}")
                hold_count += 1
                continue

            order_id, status = sell_market(symbol, target)
            vlog(f"[{symbol}] SELL_PART qty={str(target)} last=${last:.2f} notional=${notional:.2f} note={note} -> {status}")
            rows_for_sheet.append(log_row("SELL", symbol, float(target)*last, target, order_id, status, note))
            sell_count += 1
            continue

        # HOLD
        vlog(f"[{symbol}] HOLD last=${last:.2f} note={note}")
        hold_count += 1

    if rows_for_sheet:
        sheets_append(rows_for_sheet)

    print(f"🧾 Summary: SELL={sell_count} HOLD={hold_count} {'(dry-run)' if DRY_RUN else ''}")
    print("✅ stock-seller finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
