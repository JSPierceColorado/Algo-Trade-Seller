import os, json, time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import numpy as np

# ============= Config (env) =============
# Broker / execution
BROKER = os.getenv("BROKER", "ALPACA").upper()   # "ALPACA" (supported) or ""

# Accept several common env var names for Alpaca keys (more deploy-proof)
ALPACA_KEY = (
    os.getenv("ALPACA_KEY_ID")
    or os.getenv("APCA_API_KEY_ID")
    or os.getenv("ALPACA_API_KEY")
    or os.getenv("ALPACA_KEY")
    or ""
)
ALPACA_SECRET = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("ALPACA_API_SECRET")
    or os.getenv("ALPACA_SECRET")
    or ""
)

ALPACA_BASE_URL = (
    os.getenv("ALPACA_BASE_URL")
    or os.getenv("APCA_API_BASE_URL")
    or "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL   = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")  # "iex" for free plans; "sip" if you have it

DRY_RUN               = os.getenv("DRY_RUN", "1").lower() in ("1","true","yes")

# Google Sheets logging
USE_SHEETS_LOG        = os.getenv("USE_SHEETS_LOG", "1").lower() in ("1","true","yes")
SHEET_NAME            = os.getenv("SHEET_NAME", "Trading Log")
LOG_TAB               = os.getenv("STOCK_LOG_TAB", "log")  # headers: Timestamp | Action | Symbol | NotionalUSD | Qty | OrderID | Status | Note

# Verbosity (console)
VERBOSE               = os.getenv("VERBOSE", "1").lower() in ("1","true","yes")

# Strategy params
RSI_WINDOW            = int(os.getenv("RSI_WINDOW", "14"))
RSI_OVERBOUGHT        = int(os.getenv("RSI_OVERBOUGHT", "70"))       # RSI threshold for "overbought"
TAKE_PROFIT_OB_PCT    = float(os.getenv("TAKE_PROFIT_OVERBOUGHT_PCT", "5"))   # profit% if RSI overbought
TAKE_PROFIT_PCT       = float(os.getenv("TAKE_PROFIT_PCT", "10"))             # unconditional take-profit %

# ========================================
# Utility
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def vlog_print(msg: str):
    if VERBOSE:
        print(msg)

# ========================================
# Positions (Portfolio-native from Alpaca)
def fetch_alpaca_positions() -> List[Dict[str, Any]]:
    """
    Returns list of long equity positions:
    [{"ticker": "AAPL", "qty": 12.5, "avg_cost": 188.34}]
    """
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca credentials missing (ALPACA_KEY_ID / ALPACA_SECRET_KEY)")

    url = f"{ALPACA_BASE_URL}/v2/positions"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Alpaca positions error {r.status_code}: {r.text}")
    data = r.json() or []

    out = []
    for p in data:
        try:
            if str(p.get("side","")).lower() != "long":
                continue
            if p.get("asset_class","").lower() not in ("us_equity", "us_equity/etp", "us_equity/adr"):
                continue
            symbol = (p.get("symbol") or "").upper()
            qty = float(p.get("qty", "0") or 0)
            avg = float(p.get("avg_entry_price", "0") or 0)
            if symbol and qty > 0:
                out.append({"ticker": symbol, "qty": qty, "avg_cost": avg, "entry_dt": None})
        except Exception:
            continue
    return out

# ========================================
# Market data (Alpaca Data API) — robust with feed + pagination + start date
def fetch_history(ticker: str, max_days: int = 450) -> pd.DataFrame:
    """
    Robust: uses Alpaca v2 bars with explicit feed (env), start date (~800d), and pagination.
    Returns DataFrame with Date index and columns: Open, High, Low, Close, Volume.
    """
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca credentials missing (ALPACA_KEY_ID / ALPACA_SECRET_KEY)")

    base_url = f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"

    def _get_bars(feed: str) -> pd.DataFrame:
        start_dt = (datetime.utcnow() - pd.Timedelta(days=800)).strftime("%Y-%m-%dT00:00:00Z")
        params = {
            "timeframe": "1Day",
            "start": start_dt,
            "adjustment": "raw",   # or "all"
            "limit": "10000",
            "feed": feed,
        }
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET,
        }

        all_bars = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                r = requests.get(base_url, headers=headers, params=params, timeout=20)
            except Exception as e:
                vlog_print(f"⚠️ {ticker} {feed} request error: {e}")
                break
            if r.status_code >= 300:
                vlog_print(f"⚠️ {ticker} {feed} HTTP {r.status_code}: {r.text[:120]}")
                break
            j = r.json() or {}
            bars = j.get("bars", [])
            if not bars:
                break
            all_bars.extend(bars)
            page_token = j.get("next_page_token")
            if not page_token or len(all_bars) >= max_days + 50:
                break
            time.sleep(0.1)

        if not all_bars:
            return pd.DataFrame()

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
        df = df.rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        df = df.set_index("Date").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df.tail(max_days).copy()

    # Try preferred feed first, then fallback to SIP
    df = _get_bars(ALPACA_DATA_FEED)
    if df.empty and ALPACA_DATA_FEED != "sip":
        vlog_print(f"ℹ️ {ticker}: retrying with SIP feed fallback")
        df = _get_bars("sip")

    if df.empty:
        vlog_print(f"⚠️ No bars for {ticker} even after feed fallback")
    else:
        vlog_print(f"✅ {ticker}: {len(df)} daily bars fetched (last close ${float(df['Close'].iloc[-1]):.2f})")

    return df

# ========================================
# Indicators — hardened RSI (no NaNs)
def rsi(s: pd.Series, w: int = 14) -> pd.Series:
    if len(s) < 2:
        return pd.Series([50.0]*len(s), index=s.index)  # neutral if not enough data
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)

    gain = up.ewm(alpha=1/w, adjust=False).mean()
    loss = down.ewm(alpha=1/w, adjust=False).mean()

    # RS with zero-protection
    rs = pd.Series(np.divide(gain, loss.replace(0, np.nan)), index=s.index)
    rsi_raw = 100 - (100 / (1 + rs))

    # Deterministic fills for edge cases
    rsi_filled = rsi_raw.copy()
    rsi_filled[(loss == 0) & (gain > 0)] = 100.0   # only gains
    rsi_filled[(gain == 0) & (loss > 0)] = 0.0     # only losses
    rsi_filled = rsi_filled.fillna(50.0)           # flat/warm-up
    return rsi_filled

# ========================================
# Decision engine (YOUR RULES)
def decide_sell(ticker: str, qty: float, avg_cost: float, df: pd.DataFrame,
                entry_dt: Optional[pd.Timestamp]) -> Tuple[str, List[str], float]:
    """
    Returns (decision, reasons, last_price)
      - SELL if RSI >= RSI_OVERBOUGHT AND profit >= TAKE_PROFIT_OB_PCT
      - SELL if profit >= TAKE_PROFIT_PCT
      - else HOLD
    """
    if df is None or df.empty or "Close" not in df.columns:
        return "HOLD", ["no_data"], 0.0

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # Compute profit %
    if avg_cost and avg_cost > 0:
        profit_pct = (last - avg_cost) / avg_cost * 100.0
    else:
        profit_pct = 0.0

    rsi_series = rsi(close, RSI_WINDOW)
    rsi_now = float(rsi_series.iloc[-1]) if len(rsi_series) else 50.0
    if np.isnan(rsi_now):
        rsi_now = 50.0  # neutral fallback

    reasons: List[str] = []
    decision = "HOLD"

    # Rule 1: RSI overbought AND profit >= TAKE_PROFIT_OB_PCT
    if rsi_now >= RSI_OVERBOUGHT and profit_pct >= TAKE_PROFIT_OB_PCT:
        decision = "SELL"
        reasons.append(f"rsi_overbought_and_profit(RSI≥{RSI_OVERBOUGHT}, profit≥{TAKE_PROFIT_OB_PCT:.1f}%, now={profit_pct:.2f}%)")

    # Rule 2: Profit take at TAKE_PROFIT_PCT
    if decision != "SELL" and profit_pct >= TAKE_PROFIT_PCT:
        decision = "SELL"
        reasons.append(f"take_profit(profit≥{TAKE_PROFIT_PCT:.1f}%, now={profit_pct:.2f}%)")

    if not reasons:
        reasons = [f"hold(profit={profit_pct:.2f}%, rsi={rsi_now:.2f})"]

    return decision, reasons, last

# ========================================
# Optional order placement (Alpaca)
def alpaca_sell_market(ticker: str, qty: float) -> Tuple[str, str]:
    """
    Returns (order_id, status). DRY_RUN is handled by caller.
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }
    data = {
        "symbol": ticker,
        "qty": str(round(qty, 6)),
        "side": "sell",
        "type": "market",
        "time_in_force": "day"
    }
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)
    if r.status_code >= 300:
        return ("", f"error {r.status_code}: {r.text}")
    j = r.json()
    return (j.get("id",""), j.get("status","submitted"))

# ========================================
# Google Sheets logging (log tab only)
try:
    import gspread
    def get_gc():
        raw = os.getenv("GOOGLE_CREDS_JSON")
        if not raw:
            raise RuntimeError("Missing GOOGLE_CREDS_JSON")
        return gspread.service_account_from_dict(json.loads(raw))

    def ensure_log_tab(gc):
        sh = None
        try:
            sh = gc.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)

        headers = ["Timestamp","Action","Symbol","NotionalUSD","Qty","OrderID","Status","Note"]

        try:
            ws = sh.worksheet(LOG_TAB)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=LOG_TAB, rows="2000", cols="50")

        end = chr(ord('A') + len(headers) - 1) + "1"
        vals = ws.get_values(f"A1:{end}")
        if not vals or vals[0] != headers:
            ws.update([headers], f"A1:{end}")  # values first, then range_name
        try:
            ws.freeze(rows=1)
        except Exception:
            pass
        return ws

    def append_rows(ws, rows: List[List[Any]]):
        if not rows: return
        for i in range(0, len(rows), 100):
            ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")
except Exception:
    gspread = None
    def get_gc(): raise RuntimeError("gspread not available")
    def ensure_log_tab(gc): return None
    def append_rows(ws, rows): return

# ========================================
# Main
def main():
    print("🏁 stock-seller (portfolio-native) starting")
    print(f"ENV: BROKER={BROKER} DRY_RUN={DRY_RUN} BASE_URL={ALPACA_BASE_URL} FEED={ALPACA_DATA_FEED}")

    # 1) Fetch live positions from broker
    if BROKER != "ALPACA":
        raise RuntimeError("Only BROKER=ALPACA is supported in this portfolio-native version right now.")
    positions = fetch_alpaca_positions()
    if not positions:
        print("ℹ️ No long US equity positions from Alpaca. Nothing to do.")
        return

    print(f"📈 Found {len(positions)} long position(s) to evaluate")

    # 2) Optionally set up log sheet
    ws_log = None
    if USE_SHEETS_LOG and gspread is not None:
        try:
            gc = get_gc()
            ws_log = ensure_log_tab(gc)
        except Exception as e:
            print(f"⚠️ Sheets logging disabled: {e}")

    log_rows: List[List[Any]] = []

    # Helper to add a row in LOG format + console message
    def vlog(action: str, symbol: str, notional: float, qty: float, order_id: str, status: str, note: str):
        ts = now_iso()
        msg = f"[{symbol}] {action} qty={qty:.4f} notional=${notional:.2f} status={status} note={note}"
        vlog_print(msg)
        log_rows.append([ts, action, symbol, f"{notional:.2f}", f"{qty:.4f}", order_id, status, note])

    sell_count = 0
    hold_count = 0
    dryrun_proceeds = 0.0

    # 3) Evaluate positions
    for pos in positions:
        tkr = pos["ticker"]
        qty = float(pos["qty"])
        avg = float(pos["avg_cost"])

        df = fetch_history(tkr, max_days=450)
        decision, reasons, last = decide_sell(tkr, qty, avg, df, pos.get("entry_dt"))

        notional = last * qty if last > 0 else 0.0
        notes = "; ".join(reasons)

        if decision == "SELL":
            if DRY_RUN:
                order_id, status = "DRYRUN", "dry-run"
            else:
                order_id, status = alpaca_sell_market(tkr, qty)
            vlog("STOCK-SELL", tkr, notional, qty, order_id, status, notes)
            sell_count += 1
            dryrun_proceeds += notional
        else:
            vlog("STOCK-SELL-HOLD", tkr, notional, qty, "", "HOLD", notes)
            hold_count += 1

        time.sleep(0.15)  # polite throttle

    # 4) Write logs (optional)
    if ws_log is not None and log_rows:
        # pad to 8 cols as needed for your exact header set
        fixed = []
        for r in log_rows:
            if len(r) < 8: r += [""] * (8 - len(r))
            elif len(r) > 8: r = r[:8]
            fixed.append(r)
        append_rows(ws_log, fixed)

    print(
        f"🧾 Summary: SELL={sell_count} HOLD={hold_count} "
        f"{'(dry-run, est. proceeds $' + f'{dryrun_proceeds:.2f}' + ')' if DRY_RUN else ''}"
    )
    print("✅ stock-seller finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
