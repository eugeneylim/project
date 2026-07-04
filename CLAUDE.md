# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

```bash
cd ~/projects/myproject
source .venv/bin/activate
```

The venv (Python 3.14) has `yfinance`, `pandas`, `matplotlib`, `ib_async`, `pandas-datareader`, and `ffn` installed. There is no requirements.txt — install new dependencies directly with `pip install <package>` inside the activated venv.

## Running the scripts

- `python test.py --symbol SPY --strategy sma [--fast 20] [--slow 50] [--qty 100] [--no-plot]` — standalone backtest runner, no IBKR connection needed. Strategy choices: `sma`, `ema`, `rsi`, `macd`, `bollinger`, `donchian`.
- `python cta.py [--no-plot]` — macro CTA trend-following backtest: runs the Donchian breakout rule (imported from `test.py`) across a basket of futures markets (equities, rates, gold, oil, FX) with ATR-based position sizing rotated by the macro regime from `macro.py` (procyclical markets cut, defensive markets boosted on the same risk-off signal), and reports per-market plus portfolio-level returns.
- `python ls_equity.py [--no-plot]` — cross-sectional momentum long/short equity backtest: ranks a ~30-stock universe by 12-1 month momentum, goes long the top 6 / short the bottom 6 equal-weighted (dollar-neutral), rebalances monthly, and reports current holdings plus return/vol/Sharpe/drawdown against an equal-weighted long-only benchmark of the same universe (and their return correlation).
- `python etf_allocation.py [--no-plot]` — risk parity / "All Weather" ETF allocation: holds SPY/TLT/IEF/GLD/DBC weighted by inverse trailing volatility (so each contributes roughly equal risk, not equal dollars), rebalanced quarterly, and reports current weights/vols plus return/vol/Sharpe/drawdown against SPY buy & hold.
- `python global_rotation.py [--no-plot]` — global country + sector momentum rotation: ranks 12 country ETFs and 11 US sector ETFs separately by trailing momentum, holds the top 3 of each equal-weighted. Backtests the cross product of 3 strategy types (combined country+sector, country-only, sector-only) x 4 momentum lookbacks (1/3/6/12 months) = 12 variants side by side against SPY buy & hold, scored with `ffn.calc_stats()` for a full statistics table (CAGR, Sortino, Calmar, skew/kurtosis, drawdown duration, etc.) and a pairwise correlation matrix. Whichever variant has the highest total return gets its weight history plotted as a stacked bar chart.
- `python paper_test.py` — IBKR paper-trading smoke test via `ib_async`. Requires TWS or IB Gateway running in paper mode with the API enabled, reachable from WSL at the `HOST` IP hardcoded in the file (derive it from `grep nameserver /etc/resolv.conf`). Port 7497 = paper TWS, 4002 = paper Gateway.

There is no test suite, linter, or build step in this repo.

## Architecture

`test.py` is a self-contained backtest engine built around two swappable layers, both explicitly designed to be later replaced with IBKR equivalents:

- `DataFeed` — wraps `yfinance` for historical OHLCV data; swap for `ib.reqHistoricalData` later.
- `PaperBroker` — an in-memory cash/position simulator; swap for a live `ib_async.IB()` broker later.

Strategies are plain functions of the form `add_<name>_signal(df, ...) -> (df, display_name, overlay_columns)` that annotate the price DataFrame with a boolean `signal` column (long/flat). `apply_signal()` dispatches to these by name. `run()` drives the feed → signal → broker loop bar-by-bar, converting long/flat transitions into BUY/SELL orders, then reports trades/equity and calls `plot_results()` for a two-panel price+equity chart.

`macro.py` builds a daily macro regime overlay for `cta.py`: yield curve spread and VIX from `yfinance`, plus a Sahm Rule unemployment indicator from FRED (via `pandas_datareader`, no API key needed). The yield-curve signal is a "recession watch" window triggered by *un-inversion* after a period of inversion, not raw inversion (raw inversion has too long/unreliable a lead time — it was flagged 42% of the time over 2022-2024 with no recession following). `compute_regime()` produces two columns: `risk_multiplier` (halved per triggered risk-off flag, for procyclical markets) and `risk_boost` (the inverse, capped at 1.5x, for defensive/haven markets) — havens have historically trended up exactly when risk assets sell off, so `cta.py`'s `MARKET_CLASS` dict routes each market to the column matching its behavior rather than cutting the whole book uniformly. Only scales position sizing, does not gate entries/exits.

`ls_equity.py` is standalone from `test.py`/`cta.py` — it implements its own position accounting since holding concurrent long AND short legs across ~30 symbols doesn't fit the single-symbol long/flat model those scripts share. Momentum score is 12-month return skipping the most recent month (the standard "12-1" academic momentum factor, since the skipped month is dominated by short-term reversal, not the momentum effect). Note the stock universe is a hardcoded list of current well-known large caps, so the backtest has survivorship bias — it doesn't reflect stocks that were delisted or fell out of favor over the period. `equal_weighted_benchmark()` builds a long-only equal-weight index of the same universe as the yardstick — since the strategy is dollar-neutral by construction, expect it to underperform the benchmark in a strong bull market (it isn't capturing market beta on purpose) but to show near-zero return correlation with it, which is the actual measure of whether the market-neutral design is working.

`etf_allocation.py` is standalone — it holds a static long basket that drifts between quarterly rebalances rather than reacting to any entry/exit signal, so it doesn't fit the other scripts' models. `inverse_vol_weights()` recomputes each ETF's weight at every rebalance from its trailing 63-day annualized volatility, so low-vol assets (bonds) naturally get larger weights than high-vol ones (gold, commodities) — this is risk parity's defining trait, weighting by risk contribution rather than fixed dollar percentages. Expect much lower vol/drawdown than SPY buy & hold but also much lower raw return over a strong equity bull run — the payoff is a smoother ride at similar or better Sharpe, not beating the benchmark outright.

`global_rotation.py` is standalone, long-only, and the one script in this repo actually designed to try to beat a cap-weighted benchmark rather than diversify away from it: it concentrates capital into whatever countries/sectors currently have the strongest trailing momentum instead of holding a fixed diversified basket, at the cost of higher turnover (monthly) and higher correlation to the benchmark than `etf_allocation.py` or `ls_equity.py`. Default history is 7y, not 10y like `etf_allocation.py` — sector ETFs `XLC` (2018) and `XLRE` (2015) have shorter histories, and `fetch_prices()`'s `dropna(axis=1)` would silently drop them from the universe on a longer lookback since leading NaNs before an ETF's inception can't be forward-filled. `validate_currency()` asserts every ticker is USD-denominated before backtesting (all current country/sector ETFs are US-listed, so this always passes today, but it's a real guard, not just documentation, against silently mixing currencies if the universe is ever edited). `VARIANTS` is generated as the cross product of `STRATEGY_LEGS` (which legs, what capital split) x `MOMENTUM_LOOKBACKS_MONTHS` (1/3/6/12) — adding a new strategy type or lookback to either list automatically expands the comparison, no per-variant code needed. Variants with different lookbacks need different warmup, so `run()` aligns all curves to the latest common start date across variants before comparing — an apples-to-oranges bug if skipped. The "best performer" is picked purely by total return over this one historical window and its weight history is what gets charted — a fast, noisy 1-month lookback variant can win this way (as it does in the current run) without that implying it's the most robust choice; a Sharpe- or Calmar-based pick would likely favor a different, more diversified variant.

`paper_test.py` is unrelated to `test.py`'s backtest loop — it's a standalone connectivity/order-flow smoke test against a live IBKR paper account (account summary → quote → market order → poll status → positions).
