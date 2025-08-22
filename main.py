#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Seller — "sell only on profit & RSI" with safe rounding (Alpaca)

Rules
- SELL only if:
    profit_pct >= PROFIT_THRESHOLD_PCT  AND  RSI >= RSI_SELL_MIN
- If SELL criteria true and position value >= DUST_SELL_CUTOFF_USD:
    sell one ladder step (SELL_STEP_SIZE of shares) this run
- If SELL criteria true and position value <  DUST_SELL_CUTOFF_USD:
    sell the entire position (no ladder)
- Otherwise HOLD (even if position < $5)

Other features
- Fractional share rounding & clamping to avoid "insufficient qty" (403)
- Optional Sheets logging
- DRY_RUN by default

Env
  APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
  ALPACA_DATA_URL="https://data.alpaca.markets"
  ALPACA_DATA_FEED="iex"   # or "sip" if you have it

  DRY_RUN=1
  PROFIT_THRESHOLD_PCT=5
  RSI_LEN=14
  RSI_SELL_MIN=70
  RSI_TIMEFRAME=1Day      # 1Min, 5Min, 15Min, 1Hour, 1Day
  SELL_STEP_SIZE=0.33     # fraction of shares to sell per run when laddering
  DUST_SELL_CUTOFF_USD=5  # if criteria are true & value < cutoff -> sell ALL
  SELL_SHARE_DECIMALS=6   # broker fractional precision for equities

  # Optional safety limit
  MAX_POSITIONS_PER_RUN=999

  # Optional Sheets logging
  USE_SHEETS_LOG=0
  SHEET_NAME="Trading Log"
  STOCK_LOG_TAB="log"
  GOOGLE_CREDS_JSON='{"type":"service_account", ... }'
"""
import os, json, time
from decimal import Decimal, getcontext
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import requests

# ===== Config =====
ALPACA_KEY    = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
ALPACA_BASE   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE     = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
DATA_FEED     = os.getenv("ALPACA_DATA_FEED", "iex")

DRY_RUN = os.getenv("DRY_RUN", "1").lower() in ("1","true","yes")

PROFIT_THRESHOLD_PCT = float(os.getenv("PROFIT_THRESHOLD_PCT", "5"))
RSI_LEN              = int(os.getenv("RSI_LEN", "14"))
RSI_SELL_MIN         = float(os.getenv("RSI_SELL_MIN", "70"))
RSI_TIMEFRAME        = os.getenv("RSI_TIMEFRAME", "1Day")

SELL_STEP_SIZE       = float(os.getenv("SELL_STEP_SIZE", "0.33"))  # 33% per run
DUST_SELL_CUTOFF_USD = float(os.getenv("DUST_SELL_CUTOFF_USD", "5"))
SHARE_DECIMALS       = int(os.getenv("SELL_SHARE_DECIMALS", "6"))

MAX_POS_PER_RUN      = int(os.getenv("MAX_POSITIONS_PER_RUN", "999"))

# Sheets logging
USE_SHEETS_LOG   = os.getenv("USE_SHEETS_LOG", "0").lower() in ("1","true","yes")
SHEET_NAME       = os.getenv("SHEET_NAME", "Trading Log")
STOCK_LOG_TAB    = os.getenv("STOCK_LOG_TAB", "log")
GOOGLE_CREDS_JSON= os.getenv("GOOGLE_CREDS_JSON", "")

getcontext().prec = 28

# ===== Utilities =====
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def vlog(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def alp_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def http_get(url: str, params: Dict[str, Any]=None, timeout:int=20):
    r = requests.get(url, headers=alp_headers(), params=params or {}, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"GET {url} -> {r.status_code} {r.text}")
    return r.json()

def http_post(url: str, data: Dict[str, Any], timeout:int=20):
    r = requests.post(url, headers=alp_headers(), data=json.dumps(data), timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"POST {url} -> {r.status_code} {r.text}")
    return r.json()

def http_delete(url: str, timeout:int=20):
    r = requests.delete(url, headers=alp_headers(), timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"DELETE {url} -> {r.status_code} {r.text}")
    return r.json() if r.text else {"status":"ok"}

# ===== Fractional rounding & clamp =====
from decimal import Decimal as D
def round_down_shares(qty: D, decimals:int=SHARE_DECIMALS) -> D:
    step = D(1).scaleb(-decimals)  # 10^-decimals
    return (qty // step) * step

def clamp_sell_qty(requested_qty: D, available_qty: D, decimals:int=SHARE_DECIMALS) -> D:
    rq = round_down_shares(requested_qty, decimals)
    av = round_down_shares(available_qty, decimals)
    return rq if rq <= av else av

# ===== RSI (Wilder) =====
def rsi_wilder(closes: List[float], length:int=14) -> float:
    if len(closes) < length + 1:
        return float("nan")
    gains = 0.0
    losses = 0.0
    # seed
    for i in range(1, length+1):
        ch = closes[i] - closes[i-1]
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    avg_gain = gains / length
    avg_loss = losses / length
    # roll
    for i in range(length+1, len(closes)):
        ch = closes[i] - closes[i-1]
        up = max(ch, 0.0)
        dn = max(-ch, 0.0)
        avg_gain = (avg_gain*(length-1) + up) / length
        avg_loss = (avg_loss*(length-1) + dn) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ===== Broker IO =====
def fetch_positions() -> List[Dict[str, Any]]:
    url = f"{ALPACA_BASE}/v2/positions"
    j = http_get(url)
    out = []
    for p in j:
        try:
            if str(p.get("side","")).lower() != "long":
                continue
            qty = float(p["qty"])
            if qty <= 0: continue
            out.append({
                "symbol": p["symbol"],
                "qty": qty,
                "avg_cost": float(p["avg_entry_price"]),
                "market_value": float(p.get("market_value", 0) or 0),
            })
        except Exception:
            pass
    return out

def fetch_last_price(symbol: str) -> float:
    url = f"{DATA_BASE}/v2/stocks/{symbol}/trades/latest"
    params = {"feed": DATA_FEED}
    j = http_get(url, params=params)
    return float(j["trade"]["p"])

def fetch_closes_for_rsi(symbol: str, limit:int=200, timeframe:str=RSI_TIMEFRAME) -> List[float]:
    url = f"{DATA_BASE}/v2/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": limit, "feed": DATA_FEED, "adjustment":"raw"}
    j = http_get(url, params=params)
    bars = j.get("bars", [])
    return [float(b["c"]) for b in bars]

def close_position(symbol: str) -> Tuple[str,str]:
    if DRY_RUN:
        return "DRYRUN","dry-run"
    url = f"{ALPACA_BASE}/v2/positions/{symbol}"
    j = http_delete(url)
    return j.get("id","closed"), j.get("status","closed")

def sell_market(symbol: str, qty: D) -> Tuple[str,str]:
    safe_qty = clamp_sell_qty(qty, qty)
    if safe_qty <= D("0"):
        return "","qty<=0"
    if DRY_RUN:
        return "DRYRUN","dry-run"
    url = f"{ALPACA_BASE}/v2/orders"
    data = {"symbol":symbol, "qty":str(safe_qty), "side":"sell", "type":"market", "time_in_force":"day"}
    try:
        j = http_post(url, data)
        return j.get("id",""), j.get("status","submitted")
    except Exception as e:
        msg = str(e).lower()
        if "insufficient" in msg or "403" in msg:
            step = D(1).scaleb(-SHARE_DECIMALS)
            retry_qty = max(D("0"), safe_qty - step)
            if retry_qty > 0:
                data["qty"] = str(retry_qty)
                j = http_post(url, data)
                return j.get("id",""), j.get("status","submitted")
        return "","error:"+str(e)

# ===== Logging (Sheets optional) =====
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
        ws = gc.open(SHEET_NAME).worksheet(STOCK_LOG_TAB)
        ws.append_rows(rows, value_input_option="USER_ENTERED")
    except Exception as e:
        vlog(f"⚠️ Sheets logging failed: {e}")

def make_row(action: str, symbol: str, notional: float, qty: D, order_id: str, status: str, note: str) -> List[str]:
    return [now_iso(), action, symbol, f"{notional:.2f}", str(qty), order_id or "", status or "", note or ""]

# ===== Decision =====
def should_sell(profit_pct: float, rsi_value: float) -> bool:
    return (profit_pct >= PROFIT_THRESHOLD_PCT) and (rsi_value >= RSI_SELL_MIN)

def main():
    print("🏁 stock-seller starting")
    print(f"ENV DRY_RUN={DRY_RUN} PROFIT_THRESHOLD_PCT={PROFIT_THRESHOLD_PCT} RSI_SELL_MIN={RSI_SELL_MIN} "
          f"SELL_STEP_SIZE={SELL_STEP_SIZE} DUST_SELL_CUTOFF_USD={DUST_SELL_CUTOFF_USD}")

    positions = fetch_positions()
    if not positions:
        print("ℹ️ No long positions found.")
        return

    rows_for_sheet: List[List[str]] = []
    sells = holds = 0

    for i, pos in enumerate(positions):
        if i >= MAX_POS_PER_RUN:
            vlog(f"Hit MAX_POSITIONS_PER_RUN={MAX_POS_PER_RUN}; stopping.")
            break

        symbol = pos["symbol"]
        qty    = float(pos["qty"])
        avg    = float(pos["avg_cost"])

        # Latest price
        try:
            last = fetch_last_price(symbol)
        except Exception as e:
            vlog(f"⚠️ {symbol}: last price failed: {e}")
            holds += 1
            continue

        # RSI
        try:
            closes = fetch_closes_for_rsi(symbol, limit=max(200, RSI_LEN+20))
            rsi_val = rsi_wilder(closes, RSI_LEN)
        except Exception as e:
            vlog(f"⚠️ {symbol}: RSI fetch/compute failed: {e}")
            holds += 1
            continue

        if not closes or last <= 0 or qty <= 0:
            holds += 1
            continue

        profit_pct = (last - avg) / avg * 100.0
        pos_value  = last * qty

        note_prefix = f"last=${last:.2f} avg=${avg:.2f} profit={profit_pct:.2f}% rsi={rsi_val:.2f} value=${pos_value:.2f}"

        if should_sell(profit_pct, rsi_val):
            # SELL criteria met
            if pos_value < DUST_SELL_CUTOFF_USD:
                # Small position -> sell ALL
                order_id, status = close_position(symbol)
                vlog(f"[{symbol}] CLOSE_ALL (dust under ${DUST_SELL_CUTOFF_USD:.2f}) | {note_prefix} -> {status}")
                rows_for_sheet.append(make_row("SELL", symbol, pos_value, D(str(qty)), order_id, status, "dust_sell_all"))
                sells += 1
                continue
            else:
                # Ladder step: sell a fraction of current shares
                step_qty = D(str(qty)) * D(str(SELL_STEP_SIZE))
                step_qty = clamp_sell_qty(step_qty, D(str(qty)))
                if step_qty <= D("0"):
                    vlog(f"[{symbol}] step_qty<=0 after clamp; HOLD | {note_prefix}")
                    holds += 1
                    continue
                order_id, status = sell_market(symbol, step_qty)
                notional = float(step_qty) * last
                vlog(f"[{symbol}] SELL_PART qty={str(step_qty)} notional=${notional:.2f} | {note_prefix} -> {status}")
                rows_for_sheet.append(make_row("SELL", symbol, notional, step_qty, order_id, status, "ladder_step"))
                sells += 1
                continue
        else:
            # HOLD (criteria not met), even if value < $5
            vlog(f"[{symbol}] HOLD | {note_prefix}")
            holds += 1

    if rows_for_sheet:
        sheets_append(rows_for_sheet)

    print(f"🧾 Summary: SELL={sells} HOLD={holds} {'(dry-run)' if DRY_RUN else ''}")
    print("✅ stock-seller finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
