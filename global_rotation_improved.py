"""
Enhanced global_rotation.py with safer calendar-month lookbacks, CLI filters,
benchmark normalization, turnover and transaction-cost accounting, risk-adjusted
variant selection, CSV artifact output, and a cleaner weight-history chart.

This file is intentionally separate so you can compare it directly against the
original `global_rotation.py` without overwriting it.
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
TOP_N_COUNTRIES = 3
TOP_N_SECTORS = 3
STARTING_CAPITAL = 100_000.0
BENCHMARK_SYMBOL = "SPY"
REQUIRED_CURRENCY = "USD"
MOMENTUM_LOOKBACKS_MONTHS = [1, 3, 6, 12]
DEFAULT_TRANSACTION_COST_BPS = 0.0
METRIC_CHOICES = ["total_return", "sharpe", "calmar", "sortino"]

COUNTRY_ETFS = {
    "SPY": "United States",
    "EWJ": "Japan",
    "EWG": "Germany",
    "EWU": "United Kingdom",
    "EWQ": "France",
    "EWA": "Australia",
    "EWC": "Canada",
    "EWZ": "Brazil",
    "FXI": "China",
    "INDA": "India",
    "EWY": "South Korea",
    "EWT": "Taiwan",
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
ALL_LABELS = {**COUNTRY_ETFS, **SECTOR_ETFS}

STRATEGY_LEGS = {
    "Country+Sector": [(COUNTRY_ETFS, TOP_N_COUNTRIES, 0.5), (SECTOR_ETFS, TOP_N_SECTORS, 0.5)],
    "Country Only": [(COUNTRY_ETFS, TOP_N_COUNTRIES, 1.0)],
    "Sector Only": [(SECTOR_ETFS, TOP_N_SECTORS, 1.0)],
}

VARIANTS = {
    f"{name} ({lookback}mo)": (legs, lookback)
    for name, legs in STRATEGY_LEGS.items()
    for lookback in MOMENTUM_LOOKBACKS_MONTHS
}


def validate_currency(symbols):
    bad = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        currency = getattr(ticker, "fast_info", {}).get("currency") if hasattr(ticker, "fast_info") else None
        if currency != REQUIRED_CURRENCY:
            bad[symbol] = currency
    if bad:
        raise ValueError(
            f"Non-{REQUIRED_CURRENCY} tickers found, mixing currencies would corrupt P&L: {bad}"
        )


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


def momentum_reference_index(prices, asof_date, lookback_months):
    reference_date = asof_date - pd.DateOffset(months=lookback_months)
    position = prices.index.get_indexer([reference_date], method="ffill")[0]
    if position < 0:
        raise ValueError(
            f"Cannot form a {lookback_months}-month momentum window for {asof_date.date()} because source data starts too late."
        )
    return position


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


def momentum_scores(prices, symbols, asof_date, lookback_months):
    ref_i = momentum_reference_index(prices, asof_date, lookback_months)
    ref_prices = prices.iloc[ref_i][symbols]
    current_prices = prices.loc[asof_date, symbols]
    return (current_prices / ref_prices - 1).dropna().sort_values(ascending=False)


def leg_weights(prices, symbols, asof_date, lookback_months, top_n):
    scores = momentum_scores(prices, symbols, asof_date, lookback_months)
    picks = list(scores.index[:top_n])
    if not picks:
        return {}
    return {symbol: 1.0 / len(picks) for symbol in picks}


def normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {s: w / total for s, w in weights.items()}


def build_target_weights(prices, legs, asof_date, lookback_months):
    portfolio = {}
    for universe, top_n, fraction in legs:
        weights = leg_weights(prices, list(universe), asof_date, lookback_months, top_n)
        normalized = normalize_weights(weights)
        for symbol, weight in normalized.items():
            portfolio[symbol] = portfolio.get(symbol, 0.0) + fraction * weight
    return normalize_weights(portfolio)


def calculate_performance_metrics(equity):
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {
            "total_return": float("nan"),
            "annual_return": float("nan"),
            "volatility": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
            "calmar": float("nan"),
        }

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


def backtest(prices, legs, lookback_months, transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS):
    rebalance_points = rebalance_dates(prices)
    equity_curve = []
    holdings_log = []
    base_equity = STARTING_CAPITAL
    previous_weights = {}

    for i, asof_date in enumerate(rebalance_points):
        try:
            target_weights = build_target_weights(prices, legs, asof_date, lookback_months)
        except ValueError:
            continue
        if not target_weights:
            continue

        turnover = sum(
            abs(target_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            for symbol in set(target_weights) | set(previous_weights)
        )
        cost = base_equity * transaction_cost_bps * turnover
        net_equity = max(base_equity - cost, 0.0)

        entry_prices = prices.loc[asof_date]
        shares = {symbol: (net_equity * weight) / entry_prices[symbol] for symbol, weight in target_weights.items()}
        holdings_log.append((asof_date, target_weights, turnover, cost))

        next_date = rebalance_points[i + 1] if i + 1 < len(rebalance_points) else None
        if next_date is not None:
            segment = prices.loc[asof_date:next_date].iloc[:-1]
        else:
            segment = prices.loc[asof_date:]

        for date, px in segment.iterrows():
            equity_curve.append((date, sum(shares[symbol] * px[symbol] for symbol in shares)))

        if equity_curve:
            base_equity = equity_curve[-1][1]
        previous_weights = target_weights

    if not equity_curve:
        raise ValueError("No valid rebalance points were found. Adjust start/end dates or momentum lookback.")

    return pd.Series({date: value for date, value in equity_curve}), holdings_log


def weights_history_frame(holdings_log):
    rows = {}
    for date, weights, turnover, cost in holdings_log:
        rows[date] = weights
    return pd.DataFrame(rows).T.fillna(0.0)


def plot_weights_history(weights_df, title, show=True):
    fig, ax = plt.subplots(figsize=(14, 6))
    weights_df.sort_index(axis=1).plot.area(ax=ax, cmap="tab20", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [f"{symbol} ({ALL_LABELS.get(symbol, symbol)})" for symbol in weights_df.columns],
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize="small",
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def drawdown_series(equity):
    return equity / equity.cummax() - 1.0


def plot_drawdown(combined, best_name, show=True):
    dd_strategy = drawdown_series(combined[best_name]) * 100
    dd_bench = drawdown_series(combined[BENCHMARK_SYMBOL]) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd_strategy.index, dd_strategy.values, 0, color="tab:purple", alpha=0.3)
    ax.plot(dd_strategy.index, dd_strategy.values, color="tab:purple", label=best_name)
    ax.fill_between(dd_bench.index, dd_bench.values, 0, color="tab:gray", alpha=0.3)
    ax.plot(dd_bench.index, dd_bench.values, color="tab:gray", label=f"{BENCHMARK_SYMBOL} Buy & Hold")
    ax.set_title(f"Drawdown — {best_name} vs. {BENCHMARK_SYMBOL}")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_returns_heatmap(total_returns, show=True):
    strategy_names = list(STRATEGY_LEGS.keys())
    grid = np.array(
        [[total_returns.get(f"{name} ({lb}mo)", float("nan")) * 100 for lb in MOMENTUM_LOOKBACKS_MONTHS] for name in strategy_names]
    )
    vmax = np.nanmax(np.abs(grid))

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MOMENTUM_LOOKBACKS_MONTHS)))
    ax.set_xticklabels([f"{lb}mo" for lb in MOMENTUM_LOOKBACKS_MONTHS])
    ax.set_yticks(range(len(strategy_names)))
    ax.set_yticklabels(strategy_names)
    for i in range(len(strategy_names)):
        for j in range(len(MOMENTUM_LOOKBACKS_MONTHS)):
            value = grid[i, j]
            if not math.isnan(value):
                ax.text(j, i, f"{value:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Total Return by Strategy Type × Momentum Lookback")
    fig.colorbar(im, ax=ax, label="Total Return (%)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_correlation_heatmap(corr, show=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Pairwise Daily-Return Correlation")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def holdings_log_frame(holdings_log):
    rows = []
    for date, weights, turnover, cost in holdings_log:
        row = {"date": date, "turnover": turnover, "cost": cost}
        row.update(weights)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def normalize_equity(curves):
    base = curves.iloc[0]
    return curves.div(base).mul(STARTING_CAPITAL)


def filter_variants(strategy=None, lookback=None):
    variants = {}
    for name, data in VARIANTS.items():
        if strategy and not name.startswith(strategy):
            continue
        if lookback is not None and data[1] != lookback:
            continue
        variants[name] = data
    if not variants:
        raise ValueError("No variants match the requested strategy/lookback filters.")
    return variants


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest and compare global country + sector momentum rotation variants."
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation.")
    parser.add_argument("--strategy", choices=list(STRATEGY_LEGS.keys()), help="Evaluate only the named strategy type.")
    parser.add_argument("--lookback", type=int, choices=MOMENTUM_LOOKBACKS_MONTHS, help="Evaluate only the requested momentum lookback in months.")
    parser.add_argument("--start-date", type=parse_date, help="Backtest start date (YYYY-MM-DD). If omitted, uses the default history window.")
    parser.add_argument("--end-date", type=parse_date, help="Backtest end date (YYYY-MM-DD). If omitted, uses the latest available data.")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance history period to download if start/end are not provided.")
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS, help="Round-trip transaction costs in basis points per full portfolio turnover.")
    parser.add_argument("--best-metric", choices=METRIC_CHOICES, default="sharpe", help="Metric used to choose the best variant.")
    parser.add_argument("--save-csv", action="store_true", help="Save equity curves, performance metrics, and holdings history to CSV files.")
    return parser.parse_args()


def run(
    plot=True,
    strategy=None,
    lookback=None,
    start_date=None,
    end_date=None,
    period=DEFAULT_PERIOD,
    transaction_cost_bps=DEFAULT_TRANSACTION_COST_BPS,
    best_metric="sharpe",
    save_csv=False,
):
    selected_variants = filter_variants(strategy=strategy, lookback=lookback)
    all_symbols = sorted(set(COUNTRY_ETFS) | set(SECTOR_ETFS))
    validate_currency(all_symbols)
    prices = fetch_prices(all_symbols, start=start_date, end=end_date, period=period)
    print(f"Downloaded {prices.shape[1]} tickers from {prices.index.min().date()} to {prices.index.max().date()}.\n")

    results = {}
    holdings_by_variant = {}
    metric_table = {}

    for name, (legs, lookback_months) in selected_variants.items():
        equity, holdings_log = backtest(prices, legs, lookback_months, transaction_cost_bps=transaction_cost_bps)
        results[name] = equity
        holdings_by_variant[name] = holdings_log
        metric_table[name] = calculate_performance_metrics(equity)

    common_start = max(eq.index[0] for eq in results.values())
    curves = {name: eq.loc[common_start:] for name, eq in results.items()}
    benchmark = prices.loc[common_start:, BENCHMARK_SYMBOL]
    benchmark = benchmark / benchmark.iloc[0] * STARTING_CAPITAL
    curves[BENCHMARK_SYMBOL] = benchmark
    combined = pd.DataFrame(curves).dropna()
    normalized = normalize_equity(combined)

    print("Current holdings (as of latest rebalance):")
    for name, holdings_log in holdings_by_variant.items():
        _, weights, turnover, cost = holdings_log[-1]
        picks = [f"{symbol} ({ALL_LABELS.get(symbol, symbol)})" for symbol in sorted(weights)]
        print(f"  {name:24s} {', '.join(picks)} | turnover={turnover:.3f} cost=${cost:,.2f}")
    print()

    stats = ffn.calc_stats(normalized)
    stats.display()

    daily_corr = normalized.pct_change().dropna().corr().round(2)
    print("\nPairwise daily-return correlation:")
    print(daily_corr.to_string())

    metrics_df = pd.DataFrame(metric_table).T
    metrics_df = metrics_df[sorted(metrics_df.columns)]
    metrics_df.index.name = "variant"
    print("\nVariant performance metrics:")
    print(metrics_df.round(4).to_string())

    if best_metric not in metrics_df.columns:
        raise ValueError(f"Unknown best-metric {best_metric}. Choose one of {METRIC_CHOICES}.")
    best_name = metrics_df[best_metric].idxmax()
    print(f"\nBest performing variant by {best_metric}: {best_name} ({metrics_df.loc[best_name, best_metric]:.4f})")

    if (plot or save_csv) and not os.path.exists(CHARTS_DIR):
        os.makedirs(CHARTS_DIR, exist_ok=True)

    if save_csv:
        normalized.to_csv(os.path.join(CHARTS_DIR, "equity_curves.csv"))
        metrics_df.to_csv(os.path.join(CHARTS_DIR, "performance_metrics.csv"))
        holdings_log_frame(holdings_by_variant[best_name]).to_csv(os.path.join(CHARTS_DIR, "best_variant_holdings.csv"))
        if hasattr(stats, "stats"):
            stats.stats.to_csv(os.path.join(CHARTS_DIR, "ffn_stats.csv"))
        print(f"Saved CSV artifacts to {os.path.abspath(CHARTS_DIR)}/")

    if plot:
        ax = stats.plot(figsize=(10, 5))
        ax.set_title("Global Rotation Variants vs. SPY Buy & Hold")
        equity_fig = ax.get_figure()
        equity_fig.tight_layout()

        weights_df = weights_history_frame(holdings_by_variant[best_name])
        weights_fig = plot_weights_history(weights_df, f"{best_name} — Weight History (Best Performer)", show=False)

        heatmap_fig = plot_returns_heatmap({name: metrics["total_return"] for name, metrics in metric_table.items()}, show=False)
        drawdown_fig = plot_drawdown(normalized, best_name, show=False)
        corr_fig = plot_correlation_heatmap(normalized.pct_change().dropna().corr(), show=False)

        figures = {
            "equity_curves.png": equity_fig,
            "weights_history.png": weights_fig,
            "returns_heatmap.png": heatmap_fig,
            "drawdown.png": drawdown_fig,
            "correlation_heatmap.png": corr_fig,
        }
        for filename, fig in figures.items():
            fig.savefig(os.path.join(CHARTS_DIR, filename), dpi=150)
        print(f"Saved {len(figures)} charts to {os.path.abspath(CHARTS_DIR)}/")


def main():
    args = parse_args()
    run(
        plot=not args.no_plot,
        strategy=args.strategy,
        lookback=args.lookback,
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period,
        transaction_cost_bps=args.transaction_cost_bps / 10000.0,
        best_metric=args.best_metric,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
