"""
Macro regime overlay for cta.py.

Combines fast market-based proxies (yield curve, VIX) via yfinance with
slower fundamental data (unemployment) via FRED into a daily regime read.

Produces two columns for cta.py to apply by market class:
- risk_multiplier: scales DOWN procyclical markets (equities, oil) when the
  macro backdrop deteriorates.
- risk_boost: scales UP defensive/haven markets (bonds, gold) on the same
  signal, since havens have historically trended up exactly when risk
  assets sell off — a blanket cut across the whole book fights that
  diversification instead of using it.
"""

import datetime

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

VIX_RISK_OFF = 25.0  # elevated volatility threshold
SAHM_TRIGGER = 0.5  # percentage points; classic Sahm Rule recession threshold
FRED_WARMUP_DAYS = 400  # extra history so the 12-month rolling window is full by `start`

# Raw yield-curve inversion has too long/unreliable a lead time to trade on
# directly (e.g. the curve stayed inverted through most of 2022-2024 without a
# recession). The tighter historical signal is the *un-inversion* (bear
# steepening) that follows a period of inversion — recessions have tended to
# start within the following year, so treat that as a rolling danger window.
CURVE_INVERSION_THRESHOLD = -0.1
CURVE_UNINVERSION_THRESHOLD = 0.1
CURVE_LOOKBACK_DAYS = 252
CURVE_WATCH_WINDOW_DAYS = 252

RISK_BOOST_CAP = 1.5  # cap on how much a haven position can be scaled up


def fetch_market_proxies(start, end):
    tnx = yf.download("^TNX", start=start, end=end, progress=False)["Close"]
    irx = yf.download("^IRX", start=start, end=end, progress=False)["Close"]
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]

    df = pd.concat([tnx, irx, vix], axis=1)
    df.columns = ["ust_10y", "ust_3m", "vix"]
    df["yield_curve_spread"] = df["ust_10y"] - df["ust_3m"]
    return df[["yield_curve_spread", "vix"]].dropna()


def fetch_fred_fundamentals(start, end):
    fred_start = start - datetime.timedelta(days=FRED_WARMUP_DAYS)
    unrate = web.DataReader("UNRATE", "fred", fred_start, end)
    df = unrate.rename(columns={"UNRATE": "unemployment_rate"})
    # Bridge isolated missing releases (e.g. reports delayed by a government
    # shutdown) so one gap doesn't blank out the rolling windows for a year.
    df["unemployment_rate"] = df["unemployment_rate"].interpolate(limit=2)

    # Sahm Rule: 3-month avg unemployment rate minus its min over the trailing 12 months.
    three_mo_avg = df["unemployment_rate"].rolling(3).mean()
    df["sahm_indicator"] = three_mo_avg - three_mo_avg.rolling(12).min()
    return df


def compute_regime(start, end):
    """Daily risk_multiplier/risk_boost columns, aligned to market trading days."""
    market = fetch_market_proxies(start, end)
    fundamentals = fetch_fred_fundamentals(start, end)
    fundamentals_daily = fundamentals.reindex(market.index, method="ffill")

    regime = pd.DataFrame(index=market.index)
    regime["yield_curve_spread"] = market["yield_curve_spread"]
    regime["vix"] = market["vix"]
    regime["sahm_indicator"] = fundamentals_daily["sahm_indicator"]

    was_inverted = (
        (regime["yield_curve_spread"] < CURVE_INVERSION_THRESHOLD)
        .rolling(CURVE_LOOKBACK_DAYS, min_periods=1)
        .max()
        .astype(bool)
    )
    uninverted_now = regime["yield_curve_spread"] > CURVE_UNINVERSION_THRESHOLD
    uninversion_event = was_inverted & uninverted_now
    regime["curve_recession_watch"] = (
        uninversion_event.rolling(CURVE_WATCH_WINDOW_DAYS, min_periods=1).max().astype(bool)
    )

    multiplier = pd.Series(1.0, index=regime.index)
    multiplier[regime["curve_recession_watch"]] *= 0.5
    multiplier[regime["vix"] > VIX_RISK_OFF] *= 0.5
    multiplier[regime["sahm_indicator"] >= SAHM_TRIGGER] *= 0.5
    regime["risk_multiplier"] = multiplier
    regime["risk_boost"] = (2.0 - multiplier).clip(upper=RISK_BOOST_CAP)

    return regime
