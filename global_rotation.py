"""
Global country + sector momentum rotation strategy.

Tactical ETF allocation: rank a basket of country ETFs and a basket of US
sector ETFs separately by trailing momentum, hold only the strongest names
in each basket, rebalance monthly. Unlike etf_allocation.py's risk-parity
book (deliberately diversified across low-correlation assets to smooth
returns), this concentrates capital into whichever countries/sectors are
currently leading -- the mechanism by which relative-strength rotation
strategies can actually beat a cap-weighted benchmark like SPY, at the cost
of higher turnover and correlation to whatever is already trending.

Long-only, no shorting -- a rotation among fund-of-ETF holdings, not a
market-neutral book like ls_equity.py.

Backtests the cross product of 3 strategy types (combined country+sector,
country-only, sector-only) x 4 momentum lookbacks (1/3/6/12 months) = 12
variants side by side against a SPY buy & hold benchmark, scored with
ffn.calc_stats() for a full statistics table (CAGR, Sortino, Calmar,
skew/kurtosis, best/worst month, drawdown duration, etc.). The best
performer by total return gets its weight history plotted as a stacked
bar chart.
"""

import argparse
import os

import ffn
import matplotlib
matplotlib.use("Agg")  # headless: this environment's Qt/X11 GUI backend crashes on load, so render straight to file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

CHARTS_DIR = "charts"

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

TRADING_DAYS_PER_MONTH = 21
TOP_N_COUNTRIES = 3
TOP_N_SECTORS = 3
STARTING_CAPITAL = 100_000.0
BENCHMARK_SYMBOL = "SPY"
REQUIRED_CURRENCY = "USD"
MOMENTUM_LOOKBACKS_MONTHS = [1, 3, 6, 12]

# A leg is (universe_dict, top_n, capital_fraction); fractions across a
# strategy's legs must sum to 1.0. Each strategy type is tested at every
# lookback in MOMENTUM_LOOKBACKS_MONTHS below.
STRATEGY_LEGS = {
    "Country+Sector": [(COUNTRY_ETFS, TOP_N_COUNTRIES, 0.5), (SECTOR_ETFS, TOP_N_SECTORS, 0.5)],
    "Country Only": [(COUNTRY_ETFS, TOP_N_COUNTRIES, 1.0)],
    "Sector Only": [(SECTOR_ETFS, TOP_N_SECTORS, 1.0)],
}

# variant name -> (legs, lookback_months)
VARIANTS = {
    f"{name} ({lookback}mo)": (legs, lookback)
    for name, legs in STRATEGY_LEGS.items()
    for lookback in MOMENTUM_LOOKBACKS_MONTHS
}


def validate_currency(symbols):
    bad = {s: cur for s in symbols if (cur := yf.Ticker(s).fast_info.get("currency")) != REQUIRED_CURRENCY}
    if bad:
        raise ValueError(f"Non-{REQUIRED_CURRENCY} tickers found, mixing currencies would corrupt P&L: {bad}")


def fetch_prices(symbols, period="7y"):
    validate_currency(symbols)
    data = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]
    return data.dropna(how="all").ffill().dropna(axis=1)


def rebalance_indices(prices, lookback):
    month_ends = prices.resample("ME").last().index
    positions = sorted({prices.index.get_indexer([d], method="pad")[0] for d in month_ends})
    return [i for i in positions if i >= lookback]


def momentum_scores(prices, symbols, start_i, lookback):
    window = prices[symbols].iloc[start_i - lookback : start_i + 1]
    return (window.iloc[-1] / window.iloc[0] - 1).dropna().sort_values(ascending=False)


def leg_weights(prices, symbols, start_i, lookback, top_n):
    scores = momentum_scores(prices, symbols, start_i, lookback)
    picks = list(scores.index[:top_n])
    return {s: 1.0 / len(picks) for s in picks}


def backtest(prices, legs, lookback_months):
    lookback = lookback_months * TRADING_DAYS_PER_MONTH
    rebal_idx = rebalance_indices(prices, lookback)

    equity_curve = []
    holdings_log = []
    base_equity = STARTING_CAPITAL

    for k, start_i in enumerate(rebal_idx):
        end_i = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else len(prices)

        entry_px = prices.iloc[start_i]
        shares = {}
        leg_holdings = []
        for universe, top_n, fraction in legs:
            weights = leg_weights(prices, list(universe), start_i, lookback, top_n)
            leg_holdings.append(weights)
            for s, w in weights.items():
                shares[s] = shares.get(s, 0.0) + (base_equity * fraction * w) / entry_px[s]
        holdings_log.append((prices.index[start_i], leg_holdings))

        for i in range(start_i, end_i):
            px = prices.iloc[i]
            equity_curve.append((prices.index[i], sum(q * px[s] for s, q in shares.items())))

        base_equity = equity_curve[-1][1]

    equity = pd.Series(dict(equity_curve))
    return equity, holdings_log


def weights_history_frame(holdings_log, legs):
    rows = {}
    for date, leg_holdings in holdings_log:
        row = {}
        for (universe, top_n, fraction), weights in zip(legs, leg_holdings):
            for s, w in weights.items():
                row[s] = row.get(s, 0.0) + fraction * w
        rows[date] = row
    return pd.DataFrame(rows).T.fillna(0.0)


def plot_weights_history(weights_df, title, show=True):
    bar_width = 21  # days; roughly one month, so bars sit apart rather than touching
    colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(weights_df))
    for i, symbol in enumerate(weights_df.columns):
        values = weights_df[symbol].values * 100
        ax.bar(
            weights_df.index, values, bottom=bottom, width=bar_width,
            label=f"{symbol} ({ALL_LABELS[symbol]})", color=colors[i % len(colors)],
        )
        bottom += values
    ax.set_title(title)
    ax.set_ylabel("Weight (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
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
        [[total_returns[f"{name} ({lb}mo)"] * 100 for lb in MOMENTUM_LOOKBACKS_MONTHS] for name in strategy_names]
    )
    vmax = np.abs(grid).max()

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MOMENTUM_LOOKBACKS_MONTHS)), [f"{lb}mo" for lb in MOMENTUM_LOOKBACKS_MONTHS])
    ax.set_yticks(range(len(strategy_names)), strategy_names)
    for i in range(len(strategy_names)):
        for j in range(len(MOMENTUM_LOOKBACKS_MONTHS)):
            ax.text(j, i, f"{grid[i, j]:+.1f}%", ha="center", va="center", fontsize=9)
    ax.set_title("Total Return by Strategy Type × Momentum Lookback")
    fig.colorbar(im, ax=ax, label="Total Return (%)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_correlation_heatmap(corr, show=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Pairwise Daily-Return Correlation")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def run(plot=True):
    all_symbols = sorted(set(COUNTRY_ETFS) | set(SECTOR_ETFS))
    prices = fetch_prices(all_symbols)
    print(f"All {len(all_symbols)} ETFs confirmed {REQUIRED_CURRENCY}-denominated.\n")

    results = {}
    holdings_by_variant = {}
    for name, (legs, lookback_months) in VARIANTS.items():
        equity, holdings_log = backtest(prices, legs, lookback_months)
        results[name] = equity
        holdings_by_variant[name] = holdings_log

    common_start = max(eq.index[0] for eq in results.values())
    curves = {name: eq.loc[common_start:] for name, eq in results.items()}
    curves[BENCHMARK_SYMBOL] = prices.loc[common_start:, BENCHMARK_SYMBOL]
    combined = pd.DataFrame(curves).dropna()

    print("Current holdings (as of latest rebalance):")
    for name, holdings_log in holdings_by_variant.items():
        _, leg_holdings = holdings_log[-1]
        picks = [f"{s} ({ALL_LABELS[s]})" for weights in leg_holdings for s in weights]
        print(f"  {name:24s} {', '.join(picks)}")
    print()

    stats = ffn.calc_stats(combined)
    stats.display()

    print("\nPairwise daily-return correlation:")
    print(combined.pct_change().dropna().corr().round(2).to_string())

    total_returns = {name: combined[name].iloc[-1] / combined[name].iloc[0] - 1 for name in VARIANTS}
    best_name = max(total_returns, key=total_returns.get)
    print(f"\nBest performing variant by total return: {best_name} ({total_returns[best_name]*100:+.2f}%)")

    if plot:
        os.makedirs(CHARTS_DIR, exist_ok=True)

        ax = stats.plot(figsize=(10, 5))
        ax.set_title("Global Rotation Variants vs. SPY Buy & Hold")
        equity_fig = ax.get_figure()
        equity_fig.tight_layout()

        best_legs, _ = VARIANTS[best_name]
        weights_df = weights_history_frame(holdings_by_variant[best_name], best_legs)
        weights_fig = plot_weights_history(weights_df, f"{best_name} — Weight History (Best Performer)", show=False)

        heatmap_fig = plot_returns_heatmap(total_returns, show=False)
        drawdown_fig = plot_drawdown(combined, best_name, show=False)
        corr_fig = plot_correlation_heatmap(combined.pct_change().dropna().corr(), show=False)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest and compare global country + sector momentum rotation variants.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the equity curve chart.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(plot=not args.no_plot)
