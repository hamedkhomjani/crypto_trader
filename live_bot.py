"""
Live Bitcoin Golden Cross Trading Bot (paper-trading by default).

Fetches real market data from an exchange via ccxt, computes 7/30 period moving
averages, and trades the Golden Cross / Death Cross strategy. By default it runs
in PAPER mode: fills are simulated locally and no real orders are placed.

Modes (env MODE):
    paper  - real market data, simulated fills, no API keys required. (default)
    live   - places real market orders. Requires API keys and MODE=live.

Usage:
    python3 live_bot.py            # run continuously, checking on an interval
    python3 live_bot.py --once     # run a single check and exit (good for cron)

Reads configuration from a .env file (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv

from btc_golden_cross import calculate_moving_averages


load_dotenv()

MODE = os.getenv("MODE", "paper").lower()
EXCHANGE_ID = os.getenv("EXCHANGE", "binance")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1d")
SHORT_WINDOW = int(os.getenv("SHORT_WINDOW", "7"))
LONG_WINDOW = int(os.getenv("LONG_WINDOW", "30"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "3600"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")

log = logging.getLogger("live_bot")


def setup_logging() -> None:
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    log.addHandler(console)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)


@dataclass
class State:
    """Persisted account + bookkeeping state."""

    position: str = "CASH"  # CASH or BTC
    usd_cash: float = INITIAL_CAPITAL
    btc_held: float = 0.0
    last_candle_ts: int = 0
    trades: list[dict] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return self.usd_cash

    def save(self, path: str = STATE_FILE) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str = STATE_FILE) -> "State":
        if os.path.exists(path):
            with open(path) as fh:
                data = json.load(fh)
            known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
            return cls(**known)
        return cls()


def build_exchange() -> ccxt.Exchange:
    """Create a ccxt exchange. Public data only unless MODE=live."""
    if not hasattr(ccxt, EXCHANGE_ID):
        log.error("Unknown exchange '%s'.", EXCHANGE_ID)
        sys.exit(1)

    params: dict = {"enableRateLimit": True}
    if MODE == "live":
        params["apiKey"] = os.getenv(f"{EXCHANGE_ID.upper()}_API_KEY")
        params["secret"] = os.getenv(f"{EXCHANGE_ID.upper()}_API_SECRET")
        if not params["apiKey"] or not params["secret"]:
            log.error("MODE=live requires %s_API_KEY and %s_API_SECRET.", EXCHANGE_ID.upper(), EXCHANGE_ID.upper())
            sys.exit(1)

    exchange = getattr(ccxt, EXCHANGE_ID)(params)
    exchange.load_markets()
    log.info("Connected to %s | mode=%s | %s %s", EXCHANGE_ID, MODE.upper(), SYMBOL, TIMEFRAME)
    return exchange


def fetch_closed_candles(exchange: ccxt.Exchange) -> pd.DataFrame:
    """Fetch OHLCV candles and drop the still-forming (unclosed) last candle."""
    limit = LONG_WINDOW + 10
    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    # The final candle is still forming and would "repaint"; use only closed candles.
    df = df.iloc[:-1].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.rename(columns={"close": "Price"})
    return df


def detect_signal(df: pd.DataFrame) -> tuple[str, float, float]:
    """
    Return (signal, short_ma, long_ma) for the latest closed candle.

    signal is one of: GOLDEN_CROSS, DEATH_CROSS, NONE.
    """
    short_col, long_col = f"MA{SHORT_WINDOW}", f"MA{LONG_WINDOW}"
    df = calculate_moving_averages(df, short_window=SHORT_WINDOW, long_window=LONG_WINDOW)

    prev, curr = df.iloc[-2], df.iloc[-1]
    if pd.isna(prev[short_col]) or pd.isna(prev[long_col]) or pd.isna(curr[short_col]) or pd.isna(curr[long_col]):
        return "NONE", curr[short_col], curr[long_col]

    if prev[short_col] <= prev[long_col] and curr[short_col] > curr[long_col]:
        return "GOLDEN_CROSS", curr[short_col], curr[long_col]
    if prev[short_col] >= prev[long_col] and curr[short_col] < curr[long_col]:
        return "DEATH_CROSS", curr[short_col], curr[long_col]
    return "NONE", curr[short_col], curr[long_col]


def record_trade(state: State, side: str, amount: float, price: float) -> None:
    state.trades.append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "amount_btc": round(amount, 8),
            "price": price,
        }
    )


def execute_paper(state: State, signal: str, price: float) -> str:
    """Simulate a fill locally. Returns a human-readable action string."""
    if signal == "GOLDEN_CROSS" and state.position == "CASH" and state.usd_cash > 0:
        state.btc_held = state.usd_cash / price
        state.usd_cash = 0.0
        state.position = "BTC"
        record_trade(state, "BUY", state.btc_held, price)
        return f"BUY {state.btc_held:.8f} BTC @ ${price:,.2f}"

    if signal == "DEATH_CROSS" and state.position == "BTC" and state.btc_held > 0:
        state.usd_cash = state.btc_held * price
        amount = state.btc_held
        state.btc_held = 0.0
        state.position = "CASH"
        record_trade(state, "SELL", amount, price)
        return f"SELL {amount:.8f} BTC @ ${price:,.2f}"

    return "NONE"


def execute_live(exchange: ccxt.Exchange, state: State, signal: str, price: float) -> str:
    """Place a real market order. Untested - validate on testnet before real funds."""
    if signal == "GOLDEN_CROSS" and state.position == "CASH":
        balance = exchange.fetch_balance()
        quote = SYMBOL.split("/")[1]
        free_quote = balance["free"].get(quote, 0.0)
        if free_quote <= 0:
            return "NONE"
        amount = float(exchange.amount_to_precision(SYMBOL, free_quote / price))
        if amount <= 0:
            return "NONE"
        exchange.create_market_buy_order(SYMBOL, amount)
        state.btc_held = amount
        state.position = "BTC"
        record_trade(state, "BUY", amount, price)
        return f"LIVE BUY {amount:.8f} BTC @ ~${price:,.2f}"

    if signal == "DEATH_CROSS" and state.position == "BTC" and state.btc_held > 0:
        amount = float(exchange.amount_to_precision(SYMBOL, state.btc_held))
        if amount > 0:
            exchange.create_market_sell_order(SYMBOL, amount)
            record_trade(state, "SELL", amount, price)
        state.btc_held = 0.0
        state.position = "CASH"
        return f"LIVE SELL {amount:.8f} BTC @ ~${price:,.2f}"

    return "NONE"


def run_once(exchange: ccxt.Exchange, state: State) -> None:
    df = fetch_closed_candles(exchange)
    latest = df.iloc[-1]
    candle_ts = int(latest["ts"])
    price = float(latest["Price"])
    candle_dt = latest["Date"].strftime("%Y-%m-%d %H:%M UTC")

    if candle_ts <= state.last_candle_ts:
        log.info(f"No new closed candle since last check (latest {candle_dt}). Price ${price:,.2f}")
        return

    signal, short_ma, long_ma = detect_signal(df)
    ma_short_col, ma_long_col = f"MA{SHORT_WINDOW}", f"MA{LONG_WINDOW}"

    log.info(
        f"Candle {candle_dt} | Price ${price:,.2f} | {ma_short_col} ${short_ma:,.2f} | "
        f"{ma_long_col} ${long_ma:,.2f} | Signal: {signal}"
    )

    if signal == "NONE":
        state.last_candle_ts = candle_ts
        state.save()
        return

    if MODE == "live":
        action = execute_live(exchange, state, signal, price)
    else:
        action = execute_paper(state, signal, price)

    if action != "NONE":
        log.info("TRADE EXECUTED: %s", action)
    else:
        log.info("Signal %s ignored (no position change needed).", signal)

    state.last_candle_ts = candle_ts
    state.save()
    log.info(
        f"Portfolio: position={state.position} | cash=${state.usd_cash:,.2f} | "
        f"btc={state.btc_held:.8f} | value=${state.usd_cash + state.btc_held * price:,.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Golden Cross trading bot")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    args = parser.parse_args()

    setup_logging()
    if MODE == "live":
        log.warning("LIVE MODE: this will place REAL orders with REAL funds.")
    else:
        log.info("Paper mode: simulated fills only. No real orders will be placed.")

    exchange = build_exchange()
    state = State.load()
    log.info(
        f"Starting state: position={state.position} | "
        f"cash=${state.usd_cash:,.2f} | btc={state.btc_held:.8f}"
    )

    if args.once:
        run_once(exchange, state)
        return

    log.info("Entering loop. Checking every %d seconds. Ctrl+C to stop.", POLL_INTERVAL)
    while True:
        try:
            run_once(exchange, state)
        except ccxt.NetworkError as exc:
            log.warning("Network error, will retry next cycle: %s", exc)
        except ccxt.ExchangeError as exc:
            log.error("Exchange error: %s", exc)
        except KeyboardInterrupt:
            log.info("Stopped by user. State saved to %s.", STATE_FILE)
            break
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
