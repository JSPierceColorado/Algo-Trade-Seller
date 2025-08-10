import os
import json
import time
from datetime import datetime, timezone

import gspread
from alpaca_trade_api.rest import REST


# =========================
# Config (env or defaults)
# =========================
SHEET_NAME    = os.getenv("SHEET_NAME", "Trading Log")
LOG_TAB       = os.getenv("LOG_TAB", "log")

ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL  = os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets")  # live by default

# Target profit to trigger sell (percent)
TARGET_GAIN_PCT    = float(os.getenv("TARGET_GAIN_PCT", "5.0"))  # e.g., 5.0 = 5%
SLEEP_BETWEEN_ORDERS_SEC = float(os.getenv("SLEEP_BETWEEN_ORDERS_SEC", "0.5"))
EXTENDED_HOURS = os.getenv("EXTENDED_HOURS", "false").lower() in ("1", "true", "yes")


# =========================
# Helpers
# =========================
def now_iso_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_google_client():
    raw = os.getenv("GOOGLE_CREDS_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_CREDS_JSON env var.")
    creds = json.loads(raw)
    return gspread.service_account_from_dict(creds)


def _get_ws(gc, sheet_name, tab):
    sh = gc.open(sheet_name)
    try:
        return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows="2000", cols="50")


def append_logs(ws, rows):
    """
    rows: List[List[Any]] to append to LOG_TAB.
    """
    if not rows:
        return
    existing = ws.get_all_values()
    if not existing:
        ws.append_row(["Timestamp", "Action", "Symbol", "NotionalUSD", "Qty", "OrderID", "Status", "Note"])
    for i in range(0, len(rows), 100):
        ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")


def make_alpaca():
    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY.")
    return REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=APCA_API_BASE_URL)


def pct_gain(cost_basis: float, market_value: float) -> float:
    if cost_basis <= 0:
        return float("-inf")
    return (market_value - cost_basis) / cost_basis


def sell_all(api: REST, symbol: str, qty_str: str, extended: bool):
    """
    Market sell for the full quantity in the position.
    qty_str should be the position.qty string from Alpaca (works for fractional).
    """
    order = api.submit_order(
        symbol=symbol,
        side="sell",
        type="market",
        time_in_force="day",
        qty=qty_str,
        extended_hours=extended,
    )
    return order


# =========================
# Main
# =========================
def main():
    print("🏁 Sell bot starting")

    gc = get_google_client()
    api = make_alpaca()
    log_ws = _get_ws(gc, SHEET_NAME, LOG_TAB)

    target = TARGET_GAIN_PCT / 100.0
    logs = []

    try:
        positions = api.list_positions()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch positions: {e}")

    if not positions:
        print("ℹ️ No positions found.")
        return

    for pos in positions:
        try:
            # Only act on long positions
            if getattr(pos, "side", "").lower() != "long":
                continue

            symbol = pos.symbol
            qty_str = pos.qty  # keep as string for Alpaca (supports fractional)
            qty = float(qty_str)
            if qty <= 0:
                continue

            # cost_basis and market_value are USD strings like "1234.56"
            try:
                cost_basis = float(pos.cost_basis)
                market_value = float(pos.market_value)
            except Exception:
                # Fallback via avg_entry_price * qty *last? But Alpaca provides both; if missing, skip safely.
                note = "Missing/invalid cost_basis or market_value"
                print(f"⚠️ {symbol} {note}")
                logs.append([now_iso_utc(), "SELL-SKIP", symbol, "", qty_str, "", "SKIPPED", note])
                continue

            gain = pct_gain(cost_basis, market_value)

            if gain >= target:
                order = sell_all(api, symbol, qty_str, EXTENDED_HOURS)
                oid = getattr(order, "id", "")
                status = getattr(order, "status", "submitted")
                est_notional = f"{market_value:.2f}"
                print(f"✅ Submitted SELL {symbol} qty {qty_str} (~${est_notional}) @ gain {gain*100:.2f}% (order {oid})")
                logs.append([now_iso_utc(), "SELL", symbol, est_notional, qty_str, oid, status, f"Gain {gain*100:.2f}%"])
                time.sleep(SLEEP_BETWEEN_ORDERS_SEC)
            else:
                note = f"Gain {gain*100:.2f}% below target {TARGET_GAIN_PCT:.2f}%"
                logs.append([now_iso_utc(), "SELL-SKIP", symbol, f"{market_value:.2f}", qty_str, "", "SKIPPED", note])

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"❌ {pos.symbol if 'pos' in locals() else ''} {msg}")
            logs.append([now_iso_utc(), "SELL-ERROR", getattr(pos, 'symbol', ''), "", getattr(pos, 'qty', ''), "", "ERROR", msg])

    append_logs(log_ws, logs)
    print("✅ Sell cycle complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
