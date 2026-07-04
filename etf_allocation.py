"""
Risk parity / "All Weather" ETF allocation strategy.

Ray Dalio/Bridgewater-style buy-and-hold: hold a small basket of ETFs
spanning distinct asset classes (equities, long and intermediate
treasuries, gold, commodities) so the portfolio can hold up across
different economic regimes (growth, recession, inflation, deflation).
Instead of fixed dollar weights, each ETF is weighted by inverse trailing
volatility so every asset contributes roughly equal RISK to the portfolio,
not equal dollars -- the defining trait of risk parity. Rebalanced
quarterly; no market timing, no shorting, no active trading.

Standalone from test.py/cta.py/ls_equity.py: this holds a static long
basket that drifts between quarterly rebalances rather than reacting to
any entry/exit signal, so it doesn't fit those scripts' models either.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ETFS = {
    "SPY": "US Equities",
    "TLT": "Long-Term Treasuries",
    "IEF": "Intermediate Treasuries",
    "GLD": "Gold",
    "DBC": "Commodities",
}

VOL_LOOKBACK_DAYS = 63  # ~3 months of daily returns
STARTING_CAPITAL = 100_000.0
BENCHMARK_SYMBOL = "SPY"


def fetch_prices(symbols, period="10y"):
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
    return data.dropna(how="all").ffill().dropna()


def rebalance_indices(prices, lookback):
    quarter_ends = prices.resample("QE").last().index
    positions = sorted({prices.index.get_indexer([d], method="pad")[0] for d in quarter_ends})
    return [i for i in positions if i >= lookback]


def inverse_vol_weights(prices, as_of_i, lookback):
    window = prices.iloc[as_of_i - lookback : as_of_i]
    ann_vol = window.pct_change().dropna().std() * np.sqrt(252)
    inv_vol = 1 / ann_vol
    return inv_vol / inv_vol.sum(), ann_vol


def backtest():
    prices = fetch_prices(list(ETFS))
    rebal_idx = rebalance_indices(prices, VOL_LOOKBACK_DAYS)

    equity_curve = []
    weights_log = []
    shares = {s: 0.0 for s in ETFS}

    for k, start_i in enumerate(rebal_idx):
        end_i = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(prices)

        weights, ann_vol = inverse_vol_weights(prices, start_i, VOL_LOOKBACK_DAYS)
        weights_log.append((prices.index[start_i], weights, ann_vol))

        px = prices.iloc[start_i]
        portfolio_value = sum(shares[s] * px[s] for s in ETFS) or STARTING_CAPITAL
        shares = {s: (portfolio_value * weights[s]) / px[s] for s in ETFS}

        for i in range(start_i, end_i):
            px_i = prices.iloc[i]
            equity_curve.append((prices.index[i], sum(shares[s] * px_i[s] for s in ETFS)))

    equity = pd.Series(dict(equity_curve))
    return equity, weights_log, prices


def weights_history_frame(weights_log):
    return pd.DataFrame({date: weights for date, weights, _ in weights_log}).T


def buy_and_hold_benchmark(prices, symbol, start_date):
    window = prices.loc[start_date:, symbol]
    return window / window.iloc[0] * 100


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
    ax.plot(strategy_indexed.index, strategy_indexed.values, color="tab:purple", label="Risk Parity ETF Allocation")
    ax.plot(benchmark.index, benchmark.values, color="tab:gray", label=f"{BENCHMARK_SYMBOL} Buy & Hold")
    ax.set_title("Risk Parity ETF Allocation vs. Buy & Hold")
    ax.set_ylabel("Indexed value (start = 100)")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_weights_history(weights_df, show=True):
    bar_width = 60  # days; roughly one quarter, so bars sit apart rather than touching

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(weights_df))
    for symbol in ETFS:
        values = weights_df[symbol].values * 100
        ax.bar(weights_df.index, values, bottom=bottom, width=bar_width, label=f"{symbol} ({ETFS[symbol]})")
        bottom += values
    ax.set_title("Risk Parity Weight History")
    ax.set_ylabel("Weight (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def run(plot=True):
    print(
        f"Risk parity ETF allocation — {', '.join(f'{s} ({label})' for s, label in ETFS.items())}\n"
        f"Inverse-volatility weighted ({VOL_LOOKBACK_DAYS}-day lookback), rebalanced quarterly\n"
    )

    equity, weights_log, prices = backtest()
    benchmark = buy_and_hold_benchmark(prices, BENCHMARK_SYMBOL, equity.index[0])

    strat_return, strat_vol, strat_sharpe, strat_dd = performance_metrics(equity)
    bench_return, bench_vol, bench_sharpe, bench_dd = performance_metrics(benchmark)
    common = equity.pct_change().dropna().index.intersection(benchmark.pct_change().dropna().index)
    correlation = equity.pct_change()[common].corr(benchmark.pct_change()[common])

    latest_date, latest_weights, latest_vol = weights_log[-1]
    print(f"Rebalances: {len(weights_log)}")
    print(f"Current target weights (as of {latest_date.date()}):")
    for symbol, label in ETFS.items():
        print(f"  {symbol:5s} {label:24s} weight={latest_weights[symbol]*100:5.1f}%  ann.vol={latest_vol[symbol]*100:5.1f}%")
    print()

    weights_df = weights_history_frame(weights_log)
    print("Historical weights (%) at each quarterly rebalance:")
    print((weights_df * 100).round(1).to_string())
    print()

    print(f"{'':20s}{'Strategy':>12s}{'Benchmark':>14s}")
    print(f"{'Total return:':20s}{strat_return*100:>+11.2f}%{bench_return*100:>+13.2f}%")
    print(f"{'Annualized vol:':20s}{strat_vol*100:>11.2f}%{bench_vol*100:>13.2f}%")
    print(f"{'Sharpe ratio:':20s}{strat_sharpe:>12.2f}{bench_sharpe:>14.2f}")
    print(f"{'Max drawdown:':20s}{strat_dd*100:>11.2f}%{bench_dd*100:>13.2f}%")
    print(f"\nDaily return correlation (strategy vs. {BENCHMARK_SYMBOL}): {correlation:+.2f}")

    if plot:
        plot_equity(equity, benchmark)
        plot_weights_history(weights_df)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest a risk parity ETF allocation strategy.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the equity curve chart.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(plot=not args.no_plot)
