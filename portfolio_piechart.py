#!/usr/bin/env python3
"""
Portfolio allocation pie chart from tickers + share counts (and optional cash).

- Uses yfinance to fetch latest available close/last price for tickers.
- Treats CASH as a dollar amount (not shares).
- Produces a pie chart of portfolio % by market value.

Install:
  pip install yfinance pandas matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# Configuration / Conventions
# -----------------------------
CASH_SYMBOLS = {"CASH", "USD", "$", "CASH_USD"}  # accepted aliases


@dataclass(frozen=True)
class Position:
    symbol: str          # e.g., "VOO", "VXUS", "CASH"
    quantity: float      # shares for tickers, dollars for cash


def _normalize_symbol(sym: str) -> str:
    s = sym.strip().upper()
    return "CASH" if s in CASH_SYMBOLS else s


def fetch_latest_prices(
    tickers: Iterable[str],
) -> pd.Series:
    """
    Fetch latest available price for each ticker.

    Priority:
      1) fast_info["last_price"] if present
      2) last row of 1d history Close
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.Series(dtype=float)

    # yfinance supports multi-ticker download; faster than looping.
    # We pull 5 days to be resilient around weekends/holidays.
    hist = yf.download(
        tickers=tickers,
        period="5d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    prices: Dict[str, float] = {}
    for t in tickers:
        # If only one ticker, yfinance returns columns without ticker level.
        if isinstance(hist.columns, pd.MultiIndex):
            close_series = hist[(t, "Close")].dropna()
        else:
            close_series = hist["Close"].dropna()

        if not close_series.empty:
            prices[t] = float(close_series.iloc[-1])
        else:
            prices[t] = float("nan")

    return pd.Series(prices, name="price")


def portfolio_values(
    positions: Iterable[Position],
) -> pd.DataFrame:
    """
    Returns a DataFrame with symbol, quantity, price, value, pct.
    """
    pos = [Position(_normalize_symbol(p.symbol), float(p.quantity)) for p in positions]

    # Separate cash from tickers
    cash_value = sum(p.quantity for p in pos if p.symbol == "CASH")
    tickers = [p.symbol for p in pos if p.symbol != "CASH"]

    prices = fetch_latest_prices(tickers)
    rows = []

    # Ticker positions
    for p in pos:
        if p.symbol == "CASH":
            continue
        px = float(prices.get(p.symbol, float("nan")))
        val = p.quantity * px if pd.notna(px) else float("nan")
        rows.append({"symbol": p.symbol, "quantity": p.quantity, "price": px, "value": val})

    # Cash position (aggregate)
    if cash_value != 0:
        rows.append({"symbol": "CASH", "quantity": cash_value, "price": 1.0, "value": cash_value})

    df = pd.DataFrame(rows)

    # Basic validation
    if df.empty:
        raise ValueError("No positions provided.")
    if df["value"].isna().any():
        missing = df.loc[df["value"].isna(), "symbol"].tolist()
        raise RuntimeError(f"Missing price data for: {missing}. Check tickers / market data availability.")

    total = float(df["value"].sum())
    if total <= 0:
        raise ValueError(f"Total portfolio value is non-positive: {total}")

    df["pct"] = 100.0 * df["value"] / total
    df = df.sort_values("value", ascending=False).reset_index(drop=True)
    return df


def plot_allocation_pie(
    df: pd.DataFrame,
    title: str = "Portfolio Allocation",
    min_label_pct: float = 1.0,
) -> None:
    """
    Pie chart of allocation by value.
    Labels only shown if >= min_label_pct to reduce clutter.
    """
    labels = [
        f"{sym} ({pct:.1f}%)" if pct >= min_label_pct else ""
        for sym, pct in zip(df["symbol"], df["pct"])
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.pie(
        df["value"],
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p >= min_label_pct else "",
        startangle=90,
        counterclock=False,
    )
    ax.set_title(title)
    ax.axis("equal")  # keep it circular
    plt.tight_layout()
    plt.show()


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Define portfolio: {ticker: shares}, and cash in dollars.
    # Cash is treated as dollars, not shares.
    positions = [
        Position("SPYM", 44.92),
        Position("VUG", 2.30),
        Position("VXUS", 34.41),
        Position("DXJ", 12.17),
        Position("BRK-B", 3.68),
        Position("AMPX", 3640),
        Position("GOOGL", 100.3),
        Position("META", 2.93),
        Position("AAPL", 12.64),
        Position("AMZN", 5.45),
        Position("NVDA", 9.46),
        Position("MSFT", 5.03),
        Position("ASML", 3.06),
        Position("RDDT", 20),
        Position("INTC", 75),
        Position("TSM", 2.03),
        Position("RIVN", 150.73),
        Position("BYDDY", 115),
        Position("HOOD", 25),
        Position("MOH", 26),
        Position("UPS", 25.43),
        Position("CHTR", 17),
        Position("NUE", 23.33),
        Position("IREN", 100),
        Position("SEZL", 25),
        Position("CEG", 5.22),
        Position("QS", 30),
        Position("IAUM", 95.18),
        Position("SIVR", 9.58),
        Position("VCIT", 80.41),
        Position("VGLT", 142.99),
        Position("SGOV", 62.99),
        Position("CASH", 790.9),
    ]

    df = portfolio_values(positions)
    print(df[["symbol", "quantity", "price", "value", "pct"]].to_string(index=False))
    plot_allocation_pie(df, title="My Portfolio Allocation", min_label_pct=1.0)
