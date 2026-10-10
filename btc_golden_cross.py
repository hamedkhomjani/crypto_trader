"""
Bitcoin Price Simulation & Golden Cross Trading Strategy Simulator

This script simulates 60 days of Bitcoin price data using Geometric Brownian Motion (GBM),
calculates 7-day and 30-day Simple Moving Averages (SMA) using pandas,
executes a Golden Cross trading strategy, and outputs a daily ledger and performance report.
"""

import numpy as np
import pandas as pd


def generate_btc_prices(
    days: int = 60,
    start_price: float = 60000.0,
    annual_drift: float = 0.15,
    annual_volatility: float = 0.65,
    seed: int | None = 42,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Simulates daily BTC prices using Geometric Brownian Motion (GBM).

    Parameters:
        days: Number of days to simulate.
        start_price: Initial BTC price in USD.
        annual_drift: Expected annual return rate (mu).
        annual_volatility: Expected annual volatility (sigma).
        seed: Random seed for reproducibility (None for random).
        start_date: Starting date string for the dataset.

    Returns:
        DataFrame with 'Date' and 'Price' columns.
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert annual parameters to daily scale
    dt = 1 / 365.0
    daily_drift = (annual_drift - 0.5 * annual_volatility**2) * dt
    daily_vol = annual_volatility * np.sqrt(dt)

    # Generate random shocks
    shocks = np.random.normal(0, 1, days)

    # Calculate price series
    price_multipliers = np.exp(daily_drift + daily_vol * shocks)
    price_series = np.zeros(days)
    price_series[0] = start_price

    for t in range(1, days):
        price_series[t] = price_series[t - 1] * price_multipliers[t]

    date_range = pd.date_range(start=start_date, periods=days, freq="D")

    df = pd.DataFrame(
        {
            "Date": date_range,
            "Price": np.round(price_series, 2),
        }
    )

    return df


def calculate_moving_averages(
    df: pd.DataFrame, short_window: int = 7, long_window: int = 30
) -> pd.DataFrame:
    """
    Calculates short-term and long-term simple moving averages using pandas.

    Parameters:
        df: DataFrame containing a 'Price' column.
        short_window: Window size for short-term MA (default 7 days).
        long_window: Window size for long-term MA (default 30 days).

    Returns:
        DataFrame with added 'MA_short' and 'MA_long' columns.
    """
    df = df.copy()
    df[f"MA{short_window}"] = df["Price"].rolling(window=short_window).mean().round(2)
    df[f"MA{long_window}"] = df["Price"].rolling(window=long_window).mean().round(2)
    return df


def run_golden_cross_strategy(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    short_col: str = "MA7",
    long_col: str = "MA30",
) -> pd.DataFrame:
    """
    Simulates Golden Cross / Death Cross trading strategy.

    - Golden Cross: Short MA crosses ABOVE Long MA -> BUY (if not holding)
    - Death Cross: Short MA crosses BELOW Long MA -> SELL (if holding)

    Parameters:
        df: DataFrame with 'Price', short MA, and long MA columns.
        initial_capital: Starting USD cash balance.
        short_col: Column name for short-term MA.
        long_col: Column name for long-term MA.

    Returns:
        DataFrame with daily trading ledger data.
    """
    results = df.copy()

    usd_cash = initial_capital
    btc_held = 0.0
    position = "CASH"  # CASH or BTC

    signals = []
    actions = []
    usd_balances = []
    btc_balances = []
    portfolio_values = []

    for i in range(len(results)):
        current_price = results.loc[i, "Price"]
        ma_short = results.loc[i, short_col]
        ma_long = results.loc[i, long_col]

        signal = "HOLD"
        action = "NONE"

        # Crossover detection requires previous day MAs to be non-NaN
        if i > 0 and pd.notna(results.loc[i - 1, short_col]) and pd.notna(results.loc[i - 1, long_col]):
            prev_short = results.loc[i - 1, short_col]
            prev_long = results.loc[i - 1, long_col]

            # Golden Cross: Short MA crosses above Long MA
            if prev_short <= prev_long and ma_short > ma_long:
                signal = "GOLDEN CROSS (BUY)"
                if position == "CASH" and usd_cash > 0:
                    btc_held = usd_cash / current_price
                    usd_cash = 0.0
                    position = "BTC"
                    action = f"BUY {btc_held:.4f} BTC @ ${current_price:,.2f}"

            # Death Cross: Short MA crosses below Long MA
            elif prev_short >= prev_long and ma_short < ma_long:
                signal = "DEATH CROSS (SELL)"
                if position == "BTC" and btc_held > 0:
                    usd_cash = btc_held * current_price
                    action = f"SELL {btc_held:.4f} BTC @ ${current_price:,.2f}"
                    btc_held = 0.0
                    position = "CASH"

        total_value = usd_cash + (btc_held * current_price)

        signals.append(signal)
        actions.append(action)
        usd_balances.append(round(usd_cash, 2))
        btc_balances.append(round(btc_held, 6))
        portfolio_values.append(round(total_value, 2))

    results["Signal"] = signals
    results["Action"] = actions
    results["USD_Balance"] = usd_balances
    results["BTC_Balance"] = btc_balances
    results["Portfolio_Value"] = portfolio_values

    return results


def print_daily_ledger(df_results: pd.DataFrame) -> None:
    """Prints a clean daily ledger of trades and portfolio state."""
    print("\n" + "=" * 115)
    print(" DAILY TRADING LEDGER (60-DAY BITCOIN GOLDEN CROSS SIMULATION)".center(115))
    print("=" * 115)
    header = (
        f"{'Day':<5} | {'Date':<10} | {'Price ($)':<10} | {'7-Day MA':<10} | "
        f"{'30-Day MA':<10} | {'Signal':<20} | {'USD Balance':<12} | "
        f"{'BTC Held':<10} | {'Total Value':<12}"
    )
    print(header)
    print("-" * 115)

    for idx, row in df_results.iterrows():
        day_num = idx + 1
        date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        price_str = f"${row['Price']:,.2f}"
        ma7_str = f"${row['MA7']:,.2f}" if pd.notna(row["MA7"]) else "N/A"
        ma30_str = f"${row['MA30']:,.2f}" if pd.notna(row["MA30"]) else "N/A"
        signal_str = row["Signal"]
        usd_str = f"${row['USD_Balance']:,.2f}"
        btc_str = f"{row['BTC_Balance']:.4f}"
        val_str = f"${row['Portfolio_Value']:,.2f}"

        trade_highlight = " ***" if row["Action"] != "NONE" else ""

        print(
            f"{day_num:<5} | {date_str:<10} | {price_str:<10} | {ma7_str:<10} | "
            f"{ma30_str:<10} | {signal_str:<20} | {usd_str:<12} | "
            f"{btc_str:<10} | {val_str:<12}{trade_highlight}"
        )

        if row["Action"] != "NONE":
            print(f"      ---> EXECUTED TRADE: {row['Action']}")

    print("=" * 115)


def print_portfolio_performance(
    df_results: pd.DataFrame, initial_capital: float = 10000.0
) -> None:
    """Prints portfolio performance summary vs Buy & Hold strategy."""
    start_price = df_results.iloc[0]["Price"]
    final_price = df_results.iloc[-1]["Price"]
    final_value = df_results.iloc[-1]["Portfolio_Value"]

    total_return = ((final_value - initial_capital) / initial_capital) * 100
    profit_loss = final_value - initial_capital

    # Buy & Hold performance comparison
    buy_hold_btc = initial_capital / start_price
    buy_hold_final_value = buy_hold_btc * final_price
    buy_hold_return = ((buy_hold_final_value - initial_capital) / initial_capital) * 100

    print("\n" + "=" * 65)
    print(" PORTFOLIO PERFORMANCE SUMMARY".center(65))
    print("=" * 65)
    print(f" Starting BTC Price:          ${start_price:,.2f}")
    print(f" Ending BTC Price:            ${final_price:,.2f}")
    print(f" Price Change:                {((final_price - start_price)/start_price)*100:+.2f}%")
    print("-" * 65)
    print(f" Initial Portfolio Value:     ${initial_capital:,.2f}")
    print(f" Final Portfolio Value:       ${final_value:,.2f}")
    print(f" Strategy Net Profit / Loss:  ${profit_loss:+,.2f}")
    print(f" Strategy Total Return:       {total_return:+.2f}%")
    print("-" * 65)
    print(f" Buy & Hold Strategy Return:  {buy_hold_return:+.2f}% (${buy_hold_final_value:,.2f})")
    print(
        f" Strategy Outperformance:     {total_return - buy_hold_return:+.2f}% "
        f"(${final_value - buy_hold_final_value:+,.2f})"
    )
    print("=" * 65 + "\n")


def main():
    # 1. Simulate 60 days of BTC price data (seed=35 generates realistic crossovers)
    df = generate_btc_prices(days=60, start_price=60000.0, seed=35)

    # 2. Calculate 7-day and 30-day Moving Averages using pandas
    df = calculate_moving_averages(df, short_window=7, long_window=30)

    # 3. Implement Golden Cross trading algorithm
    df_results = run_golden_cross_strategy(df, initial_capital=10000.0)

    # 4. Print daily ledger and portfolio performance
    print_daily_ledger(df_results)
    print_portfolio_performance(df_results, initial_capital=10000.0)


if __name__ == "__main__":
    main()
