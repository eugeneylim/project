"""
Interactive dashboard for the backtest strategies in this repo.

Launch with:
    streamlit run dashboard.py

Reuses each strategy script's existing backtest()/plot_*() functions rather
than reimplementing anything -- this file is purely a UI layer. Backtests
are cached (st.cache_data) so switching between widgets doesn't re-fetch
yfinance/FRED data unless the underlying parameters actually change.
"""

import datetime

import pandas as pd
import streamlit as st

import cta
import etf_allocation
import global_rotation
import ls_equity
import test as test_strategy
from ffn import calc_stats

st.set_page_config(page_title="Strategy Backtest Dashboard", layout="wide")
st.title("Strategy Backtest Dashboard")

STRATEGY_PAGES = [
    "Technical Signal (test.py)",
    "Macro CTA Trend Following",
    "Long/Short Momentum Equity",
    "Risk Parity ETF Allocation",
    "Global Country + Sector Rotation",
]
page = st.sidebar.radio("Strategy", STRATEGY_PAGES)


@st.cache_data(ttl=3600, show_spinner=False)
def run_test_backtest(symbol, strategy, fast, slow, qty):
    return test_strategy.backtest(symbol=symbol, strategy=strategy, fast=fast, slow=slow, qty=qty)


@st.cache_data(ttl=3600, show_spinner=False)
def run_cta_backtest():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=5 * 365)
    regime = cta.compute_regime(start, end)
    results = [cta.backtest_market(symbol, label, regime) for symbol, label in cta.MARKETS.items()]
    return regime, results


@st.cache_data(ttl=3600, show_spinner=False)
def run_ls_equity_backtest():
    return ls_equity.backtest()


@st.cache_data(ttl=3600, show_spinner=False)
def run_etf_allocation_backtest():
    return etf_allocation.backtest()


@st.cache_data(ttl=3600, show_spinner=False)
def run_global_rotation_backtest():
    all_symbols = sorted(set(global_rotation.COUNTRY_ETFS) | set(global_rotation.SECTOR_ETFS))
    prices = global_rotation.fetch_prices(all_symbols)
    results, holdings_by_variant = {}, {}
    for name, (legs, lookback_months) in global_rotation.VARIANTS.items():
        equity, holdings_log = global_rotation.backtest(prices, legs, lookback_months)
        results[name] = equity
        holdings_by_variant[name] = holdings_log
    return prices, results, holdings_by_variant


def metrics_table(strat_metrics, bench_metrics, bench_label="Benchmark"):
    total_return, ann_vol, sharpe, max_dd = strat_metrics
    b_total_return, b_ann_vol, b_sharpe, b_max_dd = bench_metrics
    return pd.DataFrame(
        {
            "Strategy": [f"{total_return*100:+.2f}%", f"{ann_vol*100:.2f}%", f"{sharpe:.2f}", f"{max_dd*100:.2f}%"],
            bench_label: [f"{b_total_return*100:+.2f}%", f"{b_ann_vol*100:.2f}%", f"{b_sharpe:.2f}", f"{b_max_dd*100:.2f}%"],
        },
        index=["Total Return", "Annualized Vol", "Sharpe Ratio", "Max Drawdown"],
    )


if page == "Technical Signal (test.py)":
    st.header("Technical Signal Backtest")
    st.caption("Single-symbol technical backtest: SMA/EMA/RSI/MACD/Bollinger/Donchian signals against yfinance data.")

    c1, c2, c3, c4, c5 = st.columns(5)
    symbol = c1.text_input("Symbol", "SPY")
    strategy = c2.selectbox("Signal", ["sma", "ema", "rsi", "macd", "bollinger", "donchian"])
    fast = c3.number_input("Fast window", 5, 200, 20)
    slow = c4.number_input("Slow window", 5, 400, 50)
    qty = c5.number_input("Qty per trade", 1, 10000, 100)

    if st.button("Run Backtest", key="run_test"):
        with st.spinner("Running backtest..."):
            broker, df, strategy_name, overlay_columns, markers, equity_curve = run_test_backtest(
                symbol, strategy, fast, slow, qty
            )
        last_px = float(df["close"].iloc[-1])
        final_equity = broker.equity(last_px)

        c1, c2, c3 = st.columns(3)
        c1.metric("Trades", len(broker.trades))
        c2.metric("Final Equity", f"${final_equity:,.2f}")
        c3.metric("Return", f"{(final_equity/100_000 - 1)*100:+.2f}%")

        fig = test_strategy.plot_results(symbol, strategy_name, df, markers, equity_curve, overlay_columns, show=False)
        st.pyplot(fig)

        st.subheader("Trades")
        st.dataframe(pd.DataFrame(broker.trades, columns=["Symbol", "Side", "Qty", "Price"]), hide_index=True)

elif page == "Macro CTA Trend Following":
    st.header("Macro CTA Trend Following")
    st.caption("Donchian breakout across a diversified futures basket, position sizing scaled by a macro regime overlay.")

    if st.button("Run Backtest", key="run_cta"):
        with st.spinner("Fetching macro regime and running per-market backtests..."):
            regime, results = run_cta_backtest()

        latest = regime.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Yield Curve Spread", f"{latest['yield_curve_spread']:+.2f}")
        c2.metric("VIX", f"{latest['vix']:.1f}")
        c3.metric("Sahm Indicator", f"{latest['sahm_indicator']:.2f}")
        c4.metric("Procyclical Risk Mult.", f"{latest['risk_multiplier']:.2f}")

        total_start = cta.CAPITAL_PER_MARKET * len(cta.MARKETS)
        total_final = sum(equity_curve[-1][1] for _, equity_curve, _ in results)
        st.metric("Portfolio Return", f"{(total_final/total_start - 1)*100:+.2f}%", f"${total_final:,.2f} final")

        st.subheader("Per-Market Results")
        results_df = pd.DataFrame(
            [{"Market": label, "Return %": f"{ret_pct:+.2f}%"} for label, _, ret_pct in results]
        )
        st.dataframe(results_df, hide_index=True)

        fig = cta.plot_portfolio(results, show=False)
        st.pyplot(fig)

elif page == "Long/Short Momentum Equity":
    st.header("Long/Short Momentum Equity")
    st.caption("Cross-sectional 12-1 month momentum: long top 6 / short bottom 6 of a 32-stock universe, monthly rebalance.")

    if st.button("Run Backtest", key="run_ls"):
        with st.spinner("Running backtest..."):
            equity, holdings_log, prices = run_ls_equity_backtest()
        benchmark = ls_equity.equal_weighted_benchmark(prices, equity.index[0])

        latest_date, latest_longs, latest_shorts = holdings_log[-1]
        st.write(f"**Current holdings** (as of {latest_date.date()})")
        c1, c2 = st.columns(2)
        c1.write("Long: " + ", ".join(latest_longs))
        c2.write("Short: " + ", ".join(latest_shorts))

        strat_metrics = ls_equity.performance_metrics(equity)
        bench_metrics = ls_equity.performance_metrics(benchmark)
        st.dataframe(metrics_table(strat_metrics, bench_metrics, "Benchmark (Equal-Weight)"))

        fig = ls_equity.plot_equity(equity, benchmark, show=False)
        st.pyplot(fig)

elif page == "Risk Parity ETF Allocation":
    st.header("Risk Parity ETF Allocation")
    st.caption("SPY/TLT/IEF/GLD/DBC weighted by inverse trailing volatility so each contributes roughly equal risk, rebalanced quarterly.")

    if st.button("Run Backtest", key="run_etf"):
        with st.spinner("Running backtest..."):
            equity, weights_log, prices = run_etf_allocation_backtest()
        benchmark = etf_allocation.buy_and_hold_benchmark(prices, etf_allocation.BENCHMARK_SYMBOL, equity.index[0])

        latest_date, latest_weights, latest_vol = weights_log[-1]
        st.write(f"**Current weights** (as of {latest_date.date()})")
        weights_display = pd.DataFrame(
            {
                "ETF": list(etf_allocation.ETFS.keys()),
                "Label": list(etf_allocation.ETFS.values()),
                "Weight %": [f"{latest_weights[s]*100:.1f}%" for s in etf_allocation.ETFS],
                "Ann. Vol %": [f"{latest_vol[s]*100:.1f}%" for s in etf_allocation.ETFS],
            }
        )
        st.dataframe(weights_display, hide_index=True)

        strat_metrics = etf_allocation.performance_metrics(equity)
        bench_metrics = etf_allocation.performance_metrics(benchmark)
        st.dataframe(metrics_table(strat_metrics, bench_metrics, f"{etf_allocation.BENCHMARK_SYMBOL} Buy & Hold"))

        fig1 = etf_allocation.plot_equity(equity, benchmark, show=False)
        st.pyplot(fig1)

        weights_df = etf_allocation.weights_history_frame(weights_log)
        fig2 = etf_allocation.plot_weights_history(weights_df, show=False)
        st.pyplot(fig2)

elif page == "Global Country + Sector Rotation":
    st.header("Global Country + Sector Rotation")
    st.caption(
        "Momentum rotation across 12 country ETFs and 11 US sector ETFs. "
        "Compares 3 strategy types x 4 lookbacks (1/3/6/12mo) = 12 variants against SPY buy & hold."
    )

    if st.button("Run Backtest", key="run_rotation"):
        with st.spinner("Fetching prices and running all 12 variants..."):
            prices, results, holdings_by_variant = run_global_rotation_backtest()

        common_start = max(eq.index[0] for eq in results.values())
        curves = {name: eq.loc[common_start:] for name, eq in results.items()}
        curves[global_rotation.BENCHMARK_SYMBOL] = prices.loc[common_start:, global_rotation.BENCHMARK_SYMBOL]
        combined = pd.DataFrame(curves).dropna()

        stats = calc_stats(combined)
        total_returns = {name: combined[name].iloc[-1] / combined[name].iloc[0] - 1 for name in global_rotation.VARIANTS}
        best_name = max(total_returns, key=total_returns.get)
        st.success(f"Best performing variant by total return: **{best_name}** ({total_returns[best_name]*100:+.2f}%)")

        st.subheader("Statistics Comparison")
        st.dataframe(stats.stats.astype(str))

        st.subheader("Pairwise Daily-Return Correlation")
        st.dataframe(combined.pct_change().dropna().corr().round(2))

        st.subheader("Equity Curves")
        st.line_chart(combined / combined.iloc[0] * 100)

        st.subheader("Weight History")
        variant_choice = st.selectbox("Variant", list(global_rotation.VARIANTS.keys()), index=list(global_rotation.VARIANTS.keys()).index(best_name))
        legs, _ = global_rotation.VARIANTS[variant_choice]
        weights_df = global_rotation.weights_history_frame(holdings_by_variant[variant_choice], legs)
        fig = global_rotation.plot_weights_history(weights_df, f"{variant_choice} — Weight History", show=False)
        st.pyplot(fig)
