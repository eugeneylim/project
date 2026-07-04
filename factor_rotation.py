"""
Factor rotation backtest for US factor ETFs.

This script builds an ETF rotation strategy across common US factor ETFs
(Value, Growth, Dividend Yield, Minimum Volatility, Quality, Momentum). It
rebalances monthly, ranks factor ETFs by trailing momentum, and compares the
rotation strategy against SPY buy & hold.

It also saves comparison charts and optional CSV summaries.
"""

import argparse
import math
import os

import ffn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

CHARTS_DIR = "charts"
DEFAULT_PERIOD = "7y"
STARTING_CAPITAL = 100_000.0
BENCHMARK_SYMBOL = "SPY"
REQUIRED_CURRENCY = "USD"
DEFAULT_LOOKBACK_MONTHS = 12
DEFAULT_TOP_N = 3
DEFAULT_TRANSACTION_COST_BPS = 0.0
METRIC_CHOICES = ["sharpe", "total_return", "calmar", "sortino"]

FACTOR_ETFS = {
    "VLUE": "Value",
    "VTV": "Value",
    "IVE": "Value",
    "IVW": "Growth",
    "VUG": "Growth",
    "QUAL": "Quality",
    "MTUM": "Momentum",
    "USMV": "Min Volatility",
    "SPLV": "Low Volatility",
    "VYM": "Dividend Yield",
    "SDY": "Dividend Yield",
    "NOBL": "Dividend Yield",
}

ALL_LABELS = {**FACTOR_ETFS, BENCHMARK_SYMBOL: "Benchmark"}


def validate_currency(symbols):
    bad = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        currency = getattr(ticker, "fast_info", {}).get("currency") if hasattr(ticker, "fast_info") else None
        if currency != REQUIRED_CURRENCY:
            bad[symbol] = currency
    if bad:
        raise ValueError(f"Non-{REQUIRED_CURRENCY} tickers found: {bad}")


def fetch_prices(symbols, start=None, end=None, period=DEFAULT_PERIOD):
    if start or end:
        data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    else:
        data = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=symbols[0] if len(symbols) == 1 else data.name)

    data = data.dropna(how="all").ffill().dropna(axis=1, how="all")
    if data.empty:
        raise ValueError("No price data could be loaded for the requested symbols/dates.")
    return data


def parse_date(value):
    if value is None:
        return None
    return pd.to_datetime(value, utc=False)


def rebalance_dates(prices, start_date=None, end_date=None):
    window = prices.copy()
    if start_date is not None:
        window = window.loc[window.index >= start_date]
    if end_date is not None:
        window = window.loc[window.index <= end_date]
    if window.empty:
        raise ValueError("No price data is available in the requested date range.")
    month_ends = window.resample("ME").last().index
    positions = window.index.get_indexer(month_ends, method="ffill")
    return window.index[sorted(set(positions))]


def momentum_reference_index(prices, asof_date, lookback_months):
    reference_date = asof_date - pd.DateOffset(months=lookback_months)
    position = prices.index.get_indexer([reference_date], method="ffill")[0]
    if position < 0:
        raise ValueError(
            f"Cannot form a {lookback_months}-month momentum window for {asof_date.date()} because source data starts too late."
        )
    return position


def momentum_scores(prices, symbols, asof_date, lookback_months):
    ref_i = momentum_reference_index(prices, asof_date, lookback_months)
    ref_prices = prices.iloc[ref_i][symbols]
    current_prices = prices.loc[asof_date, symbols]
    return (current_prices / ref_prices - 1).dropna().sort_values(ascending=False)


def build_target_weights(prices, lookback_months, top_n):
    target_weights = {}
    rebalance_dates_list = rebalance_dates(prices)

    for asof_date in rebalance_dates_list:
        try:
            scores = momentum_scores(prices, list(FACTOR_ETFS), asof_date, lookback_months)
        except ValueError:
            continue
        picks = list(scores.index[:top_n])
        if not picks:
            continue
        weight = 1.0 / len(picks)
        target_weights[asof_date] = {symbol: weight for symbol in picks}

    if not target_weights:
        raise ValueError(
            "No valid rebalance points could be formed with the requested lookback and data range."
        )

    return target_weights


def backtest(prices, target_weights, transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS):
    equity_curve = []
    holdings_log = []
    base_equity = STARTING_CAPITAL
    previous_weights = {}

    rebalance_dates_list = sorted(target_weights)
    for idx, asof_date in enumerate(rebalance_dates_list):
        weights = target_weights[asof_date]
        turnover = sum(
            abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in set(weights) | set(previous_weights)
        )
        cost = base_equity * transaction_cost_bps * turnover
        net_equity = max(base_equity - cost, 0.0)

        entry_prices = prices.loc[asof_date]
        shares = {symbol: (net_equity * weight) / entry_prices[symbol] for symbol, weight in weights.items()}
        holdings_log.append((asof_date, weights, turnover, cost))

        next_date = rebalance_dates_list[idx + 1] if idx + 1 < len(rebalance_dates_list) else None
        segment = prices.loc[asof_date:next_date].iloc[:-1] if next_date is not None else prices.loc[asof_date:]

        for date, px in segment.iterrows():
            equity_curve.append((date, sum(shares[symbol] * px[symbol] for symbol in shares)))

        if equity_curve:
            base_equity = equity_curve[-1][1]
        previous_weights = weights

    if not equity_curve:
        raise ValueError("No valid rebalance points were found. Adjust the date range or lookback.")

    return pd.Series({date: value for date, value in equity_curve}), holdings_log


def normalize_equity(curves):
    base = curves.iloc[0]
    return curves.div(base).mul(STARTING_CAPITAL)


def calculate_metrics(equity):
    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / len(returns)) - 1.0
    volatility = returns.std() * math.sqrt(252.0)
    sharpe = annual_return / volatility if volatility > 0 else float("nan")
    downside = returns[returns < 0.0]
    downside_std = downside.std() * math.sqrt(252.0) if not downside.empty else float("nan")
    sortino = annual_return / downside_std if downside_std > 0 else float("nan")
    max_drawdown = (equity / equity.cummax() - 1.0).min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }


def plot_equity_curves(curves, show=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    curves.plot(ax=ax)
    ax.set_title("Factor Rotation vs SPY")
    ax.set_ylabel("Portfolio Value")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_weights_history(holdings_log, show=True):
    df = pd.DataFrame({date: weights for date, weights, _, _ in holdings_log}).T.fillna(0.0)
    fig, ax = plt.subplots(figsize=(14, 6))
    df.plot.area(ax=ax, cmap="tab20", alpha=0.85)
    ax.set_title("Factor Rotation Weights")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_drawdown(curves, show=True):
    dd = curves.div(curves.cummax()).sub(1.0).mul(100)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dd.index, dd.iloc[:, 0], label="Factor Rotation")
    ax.plot(dd.index, dd.iloc[:, 1], label=BENCHMARK_SYMBOL)
    ax.fill_between(dd.index, dd.iloc[:, 0], 0, alpha=0.2)
    ax.fill_between(dd.index, dd.iloc[:, 1], 0, alpha=0.2)
    ax.set_title("Drawdown — Factor Rotation vs SPY")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_correlation_heatmap(corr, show=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Return Correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Factor ETF rotation backtest vs SPY.")
    parser.add_argument("--start-date", type=parse_date, help="Backtest start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=parse_date, help="Backtest end date (YYYY-MM-DD).")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance period if start/end are not provided.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_MONTHS, help="Momentum lookback in months.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of factor ETFs to hold.")
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS, help="Transaction cost in basis points per turnover.")
    parser.add_argument("--best-metric", choices=METRIC_CHOICES, default="sharpe", help="Metric used to compare the strategy against benchmark.")
    parser.add_argument("--save-csv", action="store_true", help="Save CSV outputs for equity curves and holdings.")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation.")
    return parser.parse_args()


def run(plot=True, start_date=None, end_date=None, period=DEFAULT_PERIOD, lookback=DEFAULT_LOOKBACK_MONTHS, top_n=DEFAULT_TOP_N, transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS, save_csv=False):
    symbols = sorted(list(FACTOR_ETFS) + [BENCHMARK_SYMBOL])
    validate_currency(symbols)
    prices = fetch_prices(symbols, start=start_date, end=end_date, period=period)
    factor_prices = prices[list(FACTOR_ETFS)]
    benchmark_prices = prices[BENCHMARK_SYMBOL]

    target_weights = build_target_weights(factor_prices, lookback, top_n)
    equity, holdings_log = backtest(factor_prices, target_weights, transaction_cost_bps=transaction_cost_bps)

    benchmark = benchmark_prices.loc[equity.index]
    benchmark = benchmark / benchmark.iloc[0] * STARTING_CAPITAL

    combined = pd.DataFrame({"Factor Rotation": equity, BENCHMARK_SYMBOL: benchmark}).dropna()
    normalized = normalize_equity(combined)

    metrics = {"Factor Rotation": calculate_metrics(normalized["Factor Rotation"]), BENCHMARK_SYMBOL: calculate_metrics(normalized[BENCHMARK_SYMBOL])}
    metrics_df = pd.DataFrame(metrics).T

    print(f"Factor ETFs: {', '.join(sorted(FACTOR_ETFS))}")
    print(f"Backtest dates: {normalized.index[0].date()} to {normalized.index[-1].date()}")
    print(f"Holding top {top_n} factor ETFs by {lookback}-month momentum each month.")
    print("\nPerformance metrics:")
    print(metrics_df.round(4).to_string())

    if save_csv:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        combined.to_csv(os.path.join(CHARTS_DIR, "factor_rotation_equity_curves.csv"))
        metrics_df.to_csv(os.path.join(CHARTS_DIR, "factor_rotation_metrics.csv"))
        holdings_df = pd.DataFrame({date: weights for date, weights, _, _ in holdings_log}).T.fillna(0.0)
        holdings_df.to_csv(os.path.join(CHARTS_DIR, "factor_rotation_holdings.csv"))
        print(f"Saved CSV outputs to {os.path.abspath(CHARTS_DIR)}/")

    if plot:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        equity_fig = plot_equity_curves(normalized, show=False)
        weights_fig = plot_weights_history(holdings_log, show=False)
        drawdown_fig = plot_drawdown(normalized, show=False)
        corr_matrix = normalized.pct_change().dropna().corr()
        corr_fig = plot_correlation_heatmap(corr_matrix, show=False)

        equity_fig.savefig(os.path.join(CHARTS_DIR, "factor_rotation_equity_curves.png"), dpi=150)
        weights_fig.savefig(os.path.join(CHARTS_DIR, "factor_rotation_weights.png"), dpi=150)
        drawdown_fig.savefig(os.path.join(CHARTS_DIR, "factor_rotation_drawdown.png"), dpi=150)
        corr_fig.savefig(os.path.join(CHARTS_DIR, "factor_rotation_correlation.png"), dpi=150)
        print(f"Saved charts to {os.path.abspath(CHARTS_DIR)}/")


def main():
    args = parse_args()
    run(
        plot=not args.no_plot,
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period,
        lookback=args.lookback,
        top_n=args.top_n,
        transaction_cost_bps=args.transaction_cost_bps / 10000.0,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
