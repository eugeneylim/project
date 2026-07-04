"""
Cross-sectional momentum long/short equity backtest.

The most common systematic implementation of long/short equity, the oldest
and largest hedge fund strategy category (Alfred Jones, 1949): rank a
universe of stocks by trailing momentum, go long the strongest names and
short the weakest, rebalance periodically. Dollar-neutral by construction,
so returns come from the spread between winners and losers rather than
broad market direction.

Standalone from scratch: test.py/cta.py model a single symbol that's either
long or flat, which doesn't fit concurrent multi-symbol long AND short legs,
so position accounting here is new.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "JNJ", "PFE", "UNH", "MRK",
    "PG", "KO", "PEP", "WMT", "HD",
    "DIS", "NFLX", "CSCO", "INTC", "ORCL", "IBM",
    "GE", "CAT", "BA", "MMM",
]

TRADING_DAYS_PER_MONTH = 21
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1  # exclude the most recent month to avoid short-term reversal
TOP_N = 6  # long leg size
BOTTOM_N = 6  # short leg size
NOTIONAL_PER_LEG = 100_000.0


def fetch_prices(universe, period="5y"):
    data = yf.download(universe, period=period, auto_adjust=True, progress=False)["Close"]
    return data.dropna(how="all").ffill().dropna(axis=1)


def rebalance_indices(prices, lookback):
    month_ends = prices.resample("ME").last().index
    positions = sorted({prices.index.get_indexer([d], method="pad")[0] for d in month_ends})
    return [i for i in positions if i >= lookback]


def momentum_scores(prices, start_i, lookback, skip):
    window = prices.iloc[start_i - lookback : start_i - skip + 1]
    return (window.iloc[-1] / window.iloc[0] - 1).dropna().sort_values(ascending=False)


def backtest():
    prices = fetch_prices(UNIVERSE)
    lookback = LOOKBACK_MONTHS * TRADING_DAYS_PER_MONTH
    skip = SKIP_MONTHS * TRADING_DAYS_PER_MONTH
    rebal_idx = rebalance_indices(prices, lookback)

    base_equity = 2 * NOTIONAL_PER_LEG
    equity_curve = []
    holdings_log = []

    for k, start_i in enumerate(rebal_idx):
        end_i = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(prices)

        scores = momentum_scores(prices, start_i, lookback, skip)
        longs, shorts = list(scores.index[:TOP_N]), list(scores.index[-BOTTOM_N:])
        holdings_log.append((prices.index[start_i], longs, shorts))

        entry_px = prices.iloc[start_i]
        long_qty = {s: (NOTIONAL_PER_LEG / len(longs)) / entry_px[s] for s in longs}
        short_qty = {s: (NOTIONAL_PER_LEG / len(shorts)) / entry_px[s] for s in shorts}

        for i in range(start_i, end_i):
            px = prices.iloc[i]
            long_pnl = sum(q * (px[s] - entry_px[s]) for s, q in long_qty.items())
            short_pnl = sum(q * (entry_px[s] - px[s]) for s, q in short_qty.items())
            equity_curve.append((prices.index[i], base_equity + long_pnl + short_pnl))

        base_equity = equity_curve[-1][1]

    equity = pd.Series(dict(equity_curve))
    return equity, holdings_log, prices


def equal_weighted_benchmark(prices, start_date):
    """Equal-weighted, long-only index of the same universe, indexed to 100 at start_date."""
    window = prices.loc[start_date:]
    daily_ret = window.pct_change().fillna(0).mean(axis=1)
    index = (1 + daily_ret).cumprod()
    return index / index.iloc[0] * 100


def performance_metrics(equity):
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    daily_ret = equity.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else float("nan")
    max_dd = (equity / equity.cummax() - 1).min()
    return total_return, ann_vol, sharpe, max_dd


def plot_equity(equity, benchmark, show=True):
    strategy_indexed = equity / equity.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_indexed.index, strategy_indexed.values, color="tab:purple", label="Long/Short Strategy")
    ax.plot(benchmark.index, benchmark.values, color="tab:gray", label="Equal-Weighted Universe (benchmark)")
    ax.set_title("Momentum Long/Short Equity vs. Universe Benchmark")
    ax.set_ylabel("Indexed value (start = 100)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def run(plot=True):
    print(
        f"Cross-sectional momentum long/short equity — "
        f"{LOOKBACK_MONTHS}-{SKIP_MONTHS} month momentum, "
        f"long top {TOP_N} / short bottom {BOTTOM_N} of {len(UNIVERSE)}, monthly rebalance\n"
    )

    equity, holdings_log, prices = backtest()
    benchmark = equal_weighted_benchmark(prices, equity.index[0])

    strat_return, strat_vol, strat_sharpe, strat_dd = performance_metrics(equity)
    bench_return, bench_vol, bench_sharpe, bench_dd = performance_metrics(benchmark)
    common = equity.pct_change().dropna().index.intersection(benchmark.pct_change().dropna().index)
    correlation = equity.pct_change()[common].corr(benchmark.pct_change()[common])

    latest_date, latest_longs, latest_shorts = holdings_log[-1]
    print(f"Rebalances: {len(holdings_log)}")
    print(f"Current holdings (as of {latest_date.date()}):")
    print(f"  Long:  {', '.join(latest_longs)}")
    print(f"  Short: {', '.join(latest_shorts)}\n")

    print(f"{'':20s}{'Strategy':>12s}{'Benchmark':>14s}")
    print(f"{'Total return:':20s}{strat_return*100:>+11.2f}%{bench_return*100:>+13.2f}%")
    print(f"{'Annualized vol:':20s}{strat_vol*100:>11.2f}%{bench_vol*100:>13.2f}%")
    print(f"{'Sharpe ratio:':20s}{strat_sharpe:>12.2f}{bench_sharpe:>14.2f}")
    print(f"{'Max drawdown:':20s}{strat_dd*100:>11.2f}%{bench_dd*100:>13.2f}%")
    print(f"\nDaily return correlation (strategy vs. benchmark): {correlation:+.2f}")

    if plot:
        plot_equity(equity, benchmark)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest a cross-sectional momentum long/short equity strategy.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the equity curve chart.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(plot=not args.no_plot)
