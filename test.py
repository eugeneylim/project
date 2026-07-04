"""
Standalone strategy runner — no IBKR needed.
Pulls free market data via yfinance, runs a technical signal backtest,
and simulates fills. Swap the DataFeed/Broker classes for IBKR later.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# ---------- Data layer (swap for IBKR reqHistoricalData later) ----------
class DataFeed:
    def history(self, symbol: str, period="1y", interval="1d") -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        # yfinance may return MultiIndex columns (field, ticker) — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        return df.dropna()


# ---------- Simple paper broker (swap for IB() later) ----------
class PaperBroker:
    def __init__(self, cash=100_000.0):
        self.cash = cash
        self.position = 0
        self.trades = []

    def order(self, symbol, qty, price, side):
        cost = qty * price
        if side == "BUY":
            self.cash -= cost
            self.position += qty
        else:
            self.cash += cost
            self.position -= qty
        self.trades.append((symbol, side, qty, round(price, 2)))

    def equity(self, price):
        return self.cash + self.position * price


# ---------- Technical signals ----------
def add_sma_signal(df, fast=20, slow=50):
    df["fast_sma"] = df["close"].rolling(fast).mean()
    df["slow_sma"] = df["close"].rolling(slow).mean()
    df["signal"] = df["fast_sma"] > df["slow_sma"]
    return df, f"{fast}/{slow} SMA crossover", ["fast_sma", "slow_sma"]


def add_ema_signal(df, fast=12, slow=26):
    df["fast_ema"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["slow_ema"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = df["fast_ema"] > df["slow_ema"]
    return df, f"{fast}/{slow} EMA crossover", ["fast_ema", "slow_ema"]


def add_rsi_signal(df, period=14, oversold=30, overbought=70):
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss

    df["rsi"] = 100 - (100 / (1 + rs))
    df["signal"] = False
    in_trade = False

    for idx, row in df.iterrows():
        if not in_trade and row["rsi"] < oversold:
            in_trade = True
        elif in_trade and row["rsi"] > overbought:
            in_trade = False
        df.at[idx, "signal"] = in_trade

    return df, f"{period}-period RSI ({oversold}/{overbought})", []


def add_macd_signal(df, fast=12, slow=26, signal_period=9):
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd"] = fast_ema - slow_ema
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["signal"] = df["macd"] > df["macd_signal"]
    return df, f"MACD ({fast}, {slow}, {signal_period})", []


def add_bollinger_signal(df, period=20, std_dev=2):
    middle = df["close"].rolling(period).mean()
    band_width = df["close"].rolling(period).std() * std_dev

    df["bb_middle"] = middle
    df["bb_upper"] = middle + band_width
    df["bb_lower"] = middle - band_width
    df["signal"] = False
    in_trade = False

    for idx, row in df.iterrows():
        if not in_trade and row["close"] < row["bb_lower"]:
            in_trade = True
        elif in_trade and row["close"] > row["bb_middle"]:
            in_trade = False
        df.at[idx, "signal"] = in_trade

    return df, f"{period}-period Bollinger Bands ({std_dev} std)", [
        "bb_upper",
        "bb_middle",
        "bb_lower",
    ]


def add_donchian_signal(df, fast=20, slow=50):
    """Classic CTA/Turtle-style trend following: breakout entry, wider channel exit."""
    entry_window = slow
    exit_window = fast

    df["entry_high"] = df["high"].rolling(entry_window).max().shift(1)
    df["exit_low"] = df["low"].rolling(exit_window).min().shift(1)
    df["signal"] = False
    in_trade = False

    for idx, row in df.iterrows():
        if not in_trade and row["close"] > row["entry_high"]:
            in_trade = True
        elif in_trade and row["close"] < row["exit_low"]:
            in_trade = False
        df.at[idx, "signal"] = in_trade

    return df, f"Donchian breakout ({slow}-day entry / {fast}-day exit)", [
        "entry_high",
        "exit_low",
    ]


def apply_signal(df, strategy, fast, slow):
    strategies = {
        "sma": lambda data: add_sma_signal(data, fast=fast, slow=slow),
        "ema": lambda data: add_ema_signal(data, fast=fast, slow=slow),
        "rsi": add_rsi_signal,
        "macd": add_macd_signal,
        "bollinger": add_bollinger_signal,
        "donchian": lambda data: add_donchian_signal(data, fast=fast, slow=slow),
    }
    if strategy not in strategies:
        choices = ", ".join(strategies)
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {choices}")
    return strategies[strategy](df)


# ---------- Backtest runner ----------
def backtest(symbol="SPY", strategy="sma", fast=20, slow=50, qty=100):
    feed = DataFeed()
    broker = PaperBroker()

    df = feed.history(symbol)
    df, strategy_name, overlay_columns = apply_signal(df, strategy, fast, slow)
    df = df.dropna()

    equity_curve = []
    markers = []  # (timestamp, price, side)
    prev = None
    for ts, row in df.iterrows():
        px = float(row["close"])
        signal = "long" if row["signal"] else "flat"

        if signal == "long" and prev != "long" and broker.position == 0:
            broker.order(symbol, qty, px, "BUY")
            markers.append((ts, px, "BUY"))
        elif signal == "flat" and prev == "long" and broker.position > 0:
            broker.order(symbol, broker.position, px, "SELL")
            markers.append((ts, px, "SELL"))
        prev = signal
        equity_curve.append((ts, broker.equity(px)))

    last_px = float(df["close"].iloc[-1])
    if broker.position:  # close out at end
        broker.order(symbol, broker.position, last_px, "SELL")
        markers.append((df.index[-1], last_px, "SELL"))
        equity_curve[-1] = (df.index[-1], broker.equity(last_px))

    return broker, df, strategy_name, overlay_columns, markers, equity_curve


def run(symbol="SPY", strategy="sma", fast=20, slow=50, qty=100, plot=True):
    broker, df, strategy_name, overlay_columns, markers, equity_curve = backtest(
        symbol=symbol, strategy=strategy, fast=fast, slow=slow, qty=qty
    )
    last_px = float(df["close"].iloc[-1])

    print(f"\n{symbol} | {strategy_name}")
    print(f"Trades executed: {len(broker.trades)}")
    for t in broker.trades:
        print("  ", t)
    print(f"Final equity: ${broker.equity(last_px):,.2f}")
    print(f"Return: {(broker.equity(last_px)/100_000 - 1)*100:.2f}%")

    if plot:
        plot_results(symbol, strategy_name, df, markers, equity_curve, overlay_columns)


def plot_results(symbol, strategy_name, df, markers, equity_curve, overlay_columns, show=True):
    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1)
    colors = ["tab:blue", "tab:orange", "tab:cyan"]
    for column, color in zip(overlay_columns, colors):
        label = column.replace("_", " ").title()
        ax_price.plot(df.index, df[column], label=label, color=color, linewidth=1)

    buys = [(ts, px) for ts, px, side in markers if side == "BUY"]
    sells = [(ts, px) for ts, px, side in markers if side == "SELL"]
    if buys:
        ax_price.scatter(*zip(*buys), marker="^", color="green", s=100, label="Buy", zorder=5)
    if sells:
        ax_price.scatter(*zip(*sells), marker="v", color="red", s=100, label="Sell", zorder=5)

    ax_price.set_title(f"{symbol} - {strategy_name}")
    ax_price.set_ylabel("Price")
    ax_price.legend()

    eq_ts, eq_val = zip(*equity_curve)
    ax_equity.plot(eq_ts, eq_val, color="tab:purple")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_xlabel("Date")

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest technical signals with yfinance data.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol to backtest.")
    parser.add_argument(
        "--strategy",
        default="sma",
        choices=["sma", "ema", "rsi", "macd", "bollinger", "donchian"],
        help="Technical signal to backtest.",
    )
    parser.add_argument("--fast", type=int, default=20, help="Fast window for SMA/EMA.")
    parser.add_argument("--slow", type=int, default=50, help="Slow window for SMA/EMA.")
    parser.add_argument("--qty", type=int, default=100, help="Shares per entry.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the result chart.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        symbol=args.symbol,
        strategy=args.strategy,
        fast=args.fast,
        slow=args.slow,
        qty=args.qty,
        plot=not args.no_plot,
    )
