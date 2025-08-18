import os, json, time, math, warnings
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
# Alpaca market data base URL
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

DRY_RUN               = os.getenv("DRY_RUN", "1").lower() in ("1","true","yes")
MIN_NOTIONAL_USD      = float(os.getenv("MIN_NOTIONAL_USD", "25"))

# Optional: Google Sheets logging (positions are NOT read from Sheets)
USE_SHEETS_LOG        = os.getenv("USE_SHEETS_LOG", "1").lower() in ("1","true","yes")
SHEET_NAME            = os.getenv("SHEET_NAME", "Trading Log")
SIGNALS_TAB           = os.getenv("STOCK_SIGNALS_TAB", "stocks_sell_signals")
LOG_TAB               = os.getenv("STOCK_LOG_TAB", "stocks_log")

# Exit criteria toggles
USE_TREND_BREAK       = os.getenv("USE_TREND_BREAK", "1").lower() in ("1","true","yes")
USE_ATR_STOP          = os.getenv("USE_ATR_STOP", "1").lower() in ("1","true","yes")
USE_TRAILING_DROP     = os.getenv("USE_TRAILING_DROP", "1").lower() in ("1","true","yes")
USE_MOMENTUM_FADE     = os.getenv("USE_MOMENTUM_FADE", "0").lower() in ("1","true","yes")
USE_EARNINGS_EXIT     = os.getenv("USE_EARNINGS_EXIT", "0").lower() in ("1","true","yes")
USE_TIME_STOP         = os.getenv("USE_TIME_STOP", "0").lower() in ("1","true","yes")

# Thresholds / params
SMA_SHORT             = int(os.getenv("SMA_SHORT", "50"))
SMA_LONG              = int(os.getenv("SMA_LONG", "200"))
TREND_MIN_DAYS        = int(os.getenv("TREND_MIN_DAYS", "3"))       # days below long SMA to confirm break
ATR_WINDOW            = int(os.getenv("ATR_WINDOW", "14"))
ATR_MULT              = float(os.getenv("ATR_MULT", "2.5"))         # EMA20 - k*ATR stop
TRAIL_LOOKBACK_D      = int(os.getenv("TRAIL_LOOKBACK_D", "100"))
TRAIL_DROP_PCT        = float(os.getenv("TRAIL_DROP_PCT", "12"))    # % drop from rolling high
RSI_WINDOW            = int(os.getenv("RSI_WINDOW", "14"))
RSI_FLOOR             = int(os.getenv("RSI_FLOOR", "40"))
TIME_STOP_DAYS        = int(os.getenv("TIME_STOP_DAYS", "60"))
UNDERPERF_SPY_PCT     = float(os.getenv("UNDERPERF_SPY_PCT", "8"))  # vs SPY over TIME_STOP_DAYS

# ========================================
# Utility
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
            if p.get("asset_class","{}").lower() not in ("us_equity", "us_equity/etp", "us_equity/adr"):
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
# Market data (Alpaca Data API)
def fetch_history(ticker: str, max_days: int = 450) -> pd.DataFrame:
    """
    Fetch daily bars from Alpaca. Returns DataFrame with index=datetime and
    columns: Open, High, Low, Close, Volume.
    """
    if not (ALPACA_KEY and ALPACA_SECRET):
        raise RuntimeError("Alpaca credentials missing (ALPACA_KEY_ID / ALPACA_SECRET_KEY)")

    limit = min(max_days + 20, 10000)

    url = f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
    params = {
        "timeframe": "1Day",
        "limit": str(limit),
        "adjustment": "raw",  # or "all" if you want splits/divs adjusted
        # "feed": "sip",      # optional if you have SIP; omit if not
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code >= 300:
            return pd.DataFrame()
        j = r.json() or {}
        bars = j.get("bars", [])
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        # 't' is RFC3339 timestamp
        df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert(None)
        df = df.rename(columns={
            "t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
        }).set_index("Date")
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[cols].sort_index().dropna().tail(max_days).copy()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_next_earnings_days(ticker: str) -> Optional[pd.Timestamp]:
    """Stub: implement via Polygon/other if USE_EARNINGS_EXIT is enabled."""
    return None

# ========================================
# Indicators
def sma(s: pd.Series, w: int) -> pd.Series: return s.rolling(w).mean()
def ema(s: pd.Series, w: int) -> pd.Series: return s.ewm(span=w, adjust=False).mean()


def atr(df: pd.DataFrame, w: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window=w, min_periods=w).mean()


def rsi(s: pd.Series, w: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    gain = up.ewm(alpha=1/w, adjust=False).mean()
    loss = down.ewm(alpha=1/w, adjust=False).mean().replace(0, np.nan)
    rs = gain / loss
    return 100 - (100/(1+rs))


def macd(s: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(s, fast); es = ema(s, slow)
    line = ef - es; sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

# ========================================
# Decision engine
def decide_sell(ticker: str, qty: float, avg_cost: float, df: pd.DataFrame,
                entry_dt: Optional[pd.Timestamp]) -> Tuple[str, List[str], float]:
    """
    Returns (decision, reasons, last_price)
    decision: "SELL" or "HOLD"
    """
    if df is None or df.empty or "Close" not in df.columns:
        return "HOLD", ["no_data"], 0.0

    close = df["Close"].astype(float)
    last = float(close.iloc[-1])

    # indicators
    sma_s = sma(close, SMA_SHORT)
    sma_l = sma(close, SMA_LONG)
    ema20 = ema(close, 20)
    atr14 = atr(df, ATR_WINDOW)
    rsi14 = rsi(close, RSI_WINDOW)
    macd_line, macd_sig, macd_hist = macd(close)

    reasons = []
    votes = 0

    # Trend break
    if USE_TREND_BREAK and len(sma_l.dropna()) > TREND_MIN_DAYS:
        below_long = (close < sma_l).tail(TREND_MIN_DAYS).all()
        cross_down = (sma_s.iloc[-1] < sma_l.iloc[-1]) and (last < sma_s.iloc[-1] < sma_l.iloc[-1])
        if below_long or cross_down:
            votes += 1; reasons.append(f"trend_break(SMA{SMA_SHORT}/{SMA_LONG})")

    # ATR stop
    if USE_ATR_STOP and not atr14.isna().iloc[-1]:
        stop_lvl = float(ema20.iloc[-1]) - ATR_MULT * float(atr14.iloc[-1])
        if last < stop_lvl:
            votes += 1; reasons.append(f"atr_stop(EMA20-{ATR_MULT}*ATR)")

    # Trailing drop since entry (or last N days)
    if USE_TRAILING_DROP:
        if entry_dt is not None and entry_dt in df.index:
            df_entry = df.loc[entry_dt:]
        else:
            df_entry = df.tail(TRAIL_LOOKBACK_D)
        rolling_high = float(df_entry["Close"].max())
        if rolling_high > 0:
            drop_pct = (rolling_high - last) / rolling_high * 100.0
            if drop_pct >= TRAIL_DROP_PCT:
                votes += 1; reasons.append(f"trail_drop({drop_pct:.1f}%≥{TRAIL_DROP_PCT}%)")

    # Momentum fade (optional)
    if USE_MOMENTUM_FADE:
        if rsi14.iloc[-1] < RSI_FLOOR and macd_line.iloc[-1] < macd_sig.iloc[-1]:
            votes += 1; reasons.append(f"momentum_fade(RSI<{RSI_FLOOR}, MACD<Signal)")

    # Earnings exit (optional)
    if USE_EARNINGS_EXIT:
        ed = fetch_next_earnings_days(ticker)
        if ed is not None:
            days_to = (ed.date() - datetime.utcnow().date()).days
            if days_to <= 1:
                votes += 1; reasons.append("pre_earnings_exit")

    # Time stop (optional): stagnation + underperform SPY
    if USE_TIME_STOP and len(close) >= TIME_STOP_DAYS + 5:
        spy = fetch_history("SPY", max_days=TIME_STOP_DAYS + 250)
        if not spy.empty:
            spy_close = spy["Close"].astype(float)
            common_close, spy_aligned = close.align(spy_close, join="inner")
            common_close = common_close.tail(TIME_STOP_DAYS+1)
            spy_aligned  = spy_aligned.tail(TIME_STOP_DAYS+1)
            if len(common_close) > TIME_STOP_DAYS and len(spy_aligned) > TIME_STOP_DAYS:
                start = float(common_close.iloc[0]); end = float(common_close.iloc[-1])
                s_start = float(spy_aligned.iloc[0]); s_end = float(spy_aligned.iloc[-1])
                ret = (end/start - 1.0) * 100.0 if start>0 else 0.0
                spy_ret = (s_end/s_start - 1.0) * 100.0 if s_start>0 else 0.0
                if ret < 0 and (spy_ret - ret) >= UNDERPERF_SPY_PCT:
                    votes += 1; reasons.append(f"time_stop({TIME_STOP_DAYS}d underperf by {spy_ret-ret:.1f}%)")

    decision = "SELL" if (votes >= 1) else "HOLD"
    if not reasons:
        reasons = ["no_trigger"]
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
# Optional Sheets logging (signals & actions) — positions are NOT read from Sheets
try:
    import gspread
    def get_gc():
        raw = os.getenv("GOOGLE_CREDS_JSON")
        if not raw:
            raise RuntimeError("Missing GOOGLE_CREDS_JSON")
        return gspread.service_account_from_dict(json.loads(raw))
    def ensure_tabs(gc):
        sh = None
        try: sh = gc.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sh = gc.create(SHEET_NAME)

        def _ensure(title, headers, clear_first=False):
            try:
                ws = sh.worksheet(title)
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=title, rows="2000", cols="50")
            if clear_first:
                ws.clear()
            end = chr(ord('A') + len(headers) - 1) + "1"
            vals = ws.get_values(f"A1:{end}")
            if not vals or vals[0] != headers:
                # Fix deprecation: pass values first, then range_name
                ws.update([headers], f"A1:{end}")
            try: ws.freeze(rows=1)
            except Exception: pass
            return ws

        signals_headers = ["Timestamp","Ticker","Decision","Qty","Last","Notional","Reasons","Notes"]
        log_headers     = ["Timestamp","Action","Ticker","ProceedsUSD","Qty","OrderID","Status","Note"]

        ws_sig = _ensure(SIGNALS_TAB, signals_headers, clear_first=True)
        ws_log = _ensure(LOG_TAB,     log_headers,     clear_first=False)
        return ws_sig, ws_log

    def append_rows(ws, rows: List[List[Any]]):
        if not rows: return
        for i in range(0, len(rows), 100):
            ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")
except Exception:
    gspread = None
    def get_gc(): raise RuntimeError("gspread not available")
    def ensure_tabs(gc): return None, None
    def append_rows(ws, rows): return

# ========================================
# Main
def main():
    print("🏁 stock-seller (portfolio-native) starting")

    # 1) Fetch live positions from broker
    if BROKER != "ALPACA":
        raise RuntimeError("Only BROKER=ALPACA is supported in this portfolio-native version right now.")
    positions = fetch_alpaca_positions()
    if not positions:
        print("ℹ️ No long equity positions from Alpaca.")
        return

    # 2) Optionally set up logging tabs (signals/log). Positions are not read from Sheets.
    ws_sig = ws_log = None
    if USE_SHEETS_LOG and gspread is not None:
        try:
            gc = get_gc()
            ws_sig, ws_log = ensure_tabs(gc)
        except Exception as e:
            print(f"⚠️ Sheets logging disabled: {e}")

    signals_rows: List[List[Any]] = []
    log_rows: List[List[Any]] = []

    for pos in positions:
        tkr = pos["ticker"]
        qty = float(pos["qty"])
        avg = float(pos["avg_cost"])

        df = fetch_history(tkr, max_days=max(450, SMA_LONG + 50))
        decision, reasons, last = decide_sell(tkr, qty, avg, df, pos.get("entry_dt"))

        notional = last * qty if last > 0 else 0.0
        notes = f"avg ${avg:.2f}"
        signals_rows.append([now_iso(), tkr, decision, f"{qty:.4f}", f"{last:.2f}", f"{notional:.2f}", ", ".join(reasons), notes])

        if decision == "SELL" and notional >= MIN_NOTIONAL_USD:
            if DRY_RUN:
                order_id, status = "DRYRUN", "dry-run"
            else:
                order_id, status = alpaca_sell_market(tkr, qty)
            log_rows.append([now_iso(), "STOCK-SELL", tkr, f"{notional:.2f}", f"{qty:.4f}", order_id, status, "; ".join(reasons)])
        else:
            log_rows.append([now_iso(), "STOCK-SELL-HOLD", tkr, f"{notional:.2f}", f"{qty:.4f}", "", "HOLD", "; ".join(reasons)])

        time.sleep(0.2)  # polite throttle

    # 3) Write logs (optional)
    if ws_sig is not None:
        append_rows(ws_sig, signals_rows)
    if ws_log is not None:
        # pad to 8 cols as needed
        fixed = []
        for r in log_rows:
            if len(r) < 8: r += [""] * (8 - len(r))
            elif len(r) > 8: r = r[:8]
            fixed.append(r)
        append_rows(ws_log, fixed)

    print("✅ stock-seller finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
