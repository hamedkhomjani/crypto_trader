"""
Unit tests for Bitcoin Price Simulation & Golden Cross Trading Strategy Simulator.
"""

import numpy as np
import pandas as pd
import pytest

from btc_golden_cross import (
    calculate_moving_averages,
    generate_btc_prices,
    run_golden_cross_strategy,
)


def test_generate_btc_prices_shape_and_columns():
    days = 60
    start_price = 50000.0
    df = generate_btc_prices(days=days, start_price=start_price, seed=123)

    assert len(df) == days
    assert list(df.columns) == ["Date", "Price"]
    assert df.iloc[0]["Price"] == start_price
    assert (df["Price"] > 0).all()


def test_generate_btc_prices_reproducibility():
    df1 = generate_btc_prices(days=30, seed=42)
    df2 = generate_btc_prices(days=30, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_calculate_moving_averages():
    # Simple test series
    prices = [10.0] * 10 + [20.0] * 10 + [30.0] * 10
    dates = pd.date_range("2025-01-01", periods=30)
    df = pd.DataFrame({"Date": dates, "Price": prices})

    df_ma = calculate_moving_averages(df, short_window=3, long_window=5)

    assert "MA3" in df_ma.columns
    assert "MA5" in df_ma.columns

    # First few rows should be NaN for rolling window
    assert pd.isna(df_ma.loc[0, "MA3"])
    assert pd.isna(df_ma.loc[1, "MA3"])
    assert df_ma.loc[2, "MA3"] == 10.0

    assert pd.isna(df_ma.loc[3, "MA5"])
    assert df_ma.loc[4, "MA5"] == 10.0


def test_golden_cross_buy_signal():
    # Construct synthetic data where short MA crosses above long MA on day 5
    # Day 0-3: MA7 <= MA30
    # Day 4: MA7 crosses above MA30 -> GOLDEN CROSS (BUY)
    dates = pd.date_range("2025-01-01", periods=6)
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    ma7 = [10.0, 10.0, 10.0, 10.0, 20.0, 25.0]
    ma30 = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

    df = pd.DataFrame(
        {
            "Date": dates,
            "Price": prices,
            "MA7": ma7,
            "MA30": ma30,
        }
    )

    df_res = run_golden_cross_strategy(df, initial_capital=1000.0)

    # Day 4 (index 4): MA7 was 10 (<= 15) on day 3, now 20 (> 15) on day 4.
    assert df_res.loc[4, "Signal"] == "GOLDEN CROSS (BUY)"
    assert "BUY" in df_res.loc[4, "Action"]
    assert df_res.loc[4, "USD_Balance"] == 0.0
    assert df_res.loc[4, "BTC_Balance"] == 10.0  # $1000 / $100 = 10 BTC
    assert df_res.loc[4, "Portfolio_Value"] == 1000.0


def test_death_cross_sell_signal():
    # Construct scenario: Starts holding BTC, then Death Cross occurs
    dates = pd.date_range("2025-01-01", periods=6)
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 120.0]
    ma7 = [10.0, 20.0, 25.0, 25.0, 10.0, 5.0]
    ma30 = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

    df = pd.DataFrame(
        {
            "Date": dates,
            "Price": prices,
            "MA7": ma7,
            "MA30": ma30,
        }
    )

    df_res = run_golden_cross_strategy(df, initial_capital=1000.0)

    # Index 1: Golden Cross -> Buy 10 BTC @ $100
    assert df_res.loc[1, "Signal"] == "GOLDEN CROSS (BUY)"
    assert df_res.loc[1, "BTC_Balance"] == 10.0

    # Index 4: Death Cross (MA7 was 25 >= 15, now 10 < 15) -> Sell 10 BTC @ $100 = $1000 cash
    assert df_res.loc[4, "Signal"] == "DEATH CROSS (SELL)"
    assert "SELL" in df_res.loc[4, "Action"]
    assert df_res.loc[4, "USD_Balance"] == 1000.0
    assert df_res.loc[4, "BTC_Balance"] == 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
