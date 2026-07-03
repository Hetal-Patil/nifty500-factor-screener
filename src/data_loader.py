"""
Data loading for the Nifty 500 Factor Screener.

Two modes:

1. DEMO mode (`generate_demo_data`) - builds a synthetic but realistic panel
   (fundamentals + returns for N stocks x T months) with a small embedded
   factor premium baked in. This lets the full scoring/ranking/backtest
   pipeline run end-to-end offline, for development and testing.

2. LIVE mode (`fetch_live_prices`) - pulls real monthly prices from Yahoo
   Finance via yfinance and computes trailing 3M/6M momentum. yfinance does
   NOT reliably expose P/E, P/B, ROE, D/E for NSE-listed stocks, so
   fundamentals must be sourced separately (e.g. screener.in exports,
   Trendlyne, NSE bhavcopy + annual report data, or a paid vendor) and
   merged in on the same (Ticker, Date) keys used by the DEMO schema:
   Date, Ticker, PE, PB, ROE, DE, Ret_3M, Ret_6M, MonthlyReturn, IndexReturn.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_sample_tickers(path: Path | None = None) -> list[str]:
    path = path or DATA_DIR / "nifty500_sample_tickers.csv"
    return pd.read_csv(path)["Ticker"].tolist()


def fetch_live_prices(tickers: list[str], start="2019-01-01", end=None) -> pd.DataFrame:
    """Pulls monthly close prices and computes 3M/6M trailing momentum.
    Returns a long DataFrame: Date, Ticker, Ret_3M, Ret_6M, MonthlyReturn.
    Fundamentals (PE, PB, ROE, DE) still need to be merged in separately.
    """
    import yfinance as yf

    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    px = yf.download(tickers, start=start, end=end, interval="1mo", auto_adjust=True)["Close"]
    px = px.dropna(how="all")

    ret_3m = px.pct_change(3)
    ret_6m = px.pct_change(6)
    fwd_ret = px.pct_change().shift(-1)  # return realized AFTER the scoring date

    frames = []
    for tkr in px.columns:
        frames.append(pd.DataFrame({
            "Date": px.index,
            "Ticker": tkr,
            "Ret_3M": ret_3m[tkr].values,
            "Ret_6M": ret_6m[tkr].values,
            "MonthlyReturn": fwd_ret[tkr].values,
        }))
    return pd.concat(frames, ignore_index=True).dropna(subset=["MonthlyReturn"])


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def generate_demo_data(n_stocks: int = 500, n_months: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic panel for pipeline development/testing.

    IMPORTANT: This is illustrative data, not real Nifty 500 fundamentals or
    prices. A small true factor premium is embedded on purpose so the demo
    backtest behaves sensibly - swap in fetch_live_prices() + a real
    fundamentals feed for genuine results before quoting any performance
    numbers.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    tickers = [f"STK{i:04d}.NS" for i in range(n_stocks)]

    sectors = ["Financials", "IT", "Energy", "Consumer", "Industrials",
               "Healthcare", "Materials", "Utilities", "Auto", "Telecom"]
    # sector effects create realistic cross-sector differences in leverage/valuation
    # (banks run higher D/E, IT runs higher P/E, etc.) that sector-neutral scoring
    # is designed to strip out.
    sector_pe_mult = {s: rng.uniform(0.7, 1.5) for s in sectors}
    sector_de_add = {s: rng.uniform(-0.3, 0.6) for s in sectors}
    stock_sector = rng.choice(sectors, size=n_stocks)

    base_pe = rng.lognormal(mean=3.0, sigma=0.4, size=n_stocks)
    base_pe *= np.array([sector_pe_mult[s] for s in stock_sector])
    base_pb = rng.lognormal(mean=1.0, sigma=0.5, size=n_stocks)
    base_roe = rng.normal(15, 8, size=n_stocks)
    base_de = np.abs(rng.normal(0.6, 0.5, size=n_stocks) + np.array([sector_de_add[s] for s in stock_sector]))

    true_premium = (
        -0.15 * _zscore(base_pe)
        - 0.15 * _zscore(base_pb)
        + 0.15 * _zscore(base_roe)
        - 0.15 * _zscore(base_de)
    )

    records = []
    for date in dates:
        index_ret = rng.normal(0.012, 0.045)
        pe = np.clip(base_pe * rng.lognormal(0, 0.05, n_stocks), 3, 150)
        pb = np.clip(base_pb * rng.lognormal(0, 0.05, n_stocks), 0.3, 30)
        roe = base_roe + rng.normal(0, 2, n_stocks)
        de = np.clip(base_de + rng.normal(0, 0.05, n_stocks), 0, 4)
        ret_3m = rng.normal(0.03, 0.08, n_stocks) + 0.02 * true_premium
        ret_6m = rng.normal(0.06, 0.12, n_stocks) + 0.03 * true_premium
        stock_specific = rng.normal(0, 0.06, n_stocks)
        monthly_ret = index_ret + 0.004 * true_premium + stock_specific

        # occasionally inject a few negative-P/E (loss-making) names and a
        # missing-data row, since real fundamentals data always has these
        loss_mask = rng.random(n_stocks) < 0.02
        pe = np.where(loss_mask, -rng.uniform(1, 20, n_stocks), pe)

        for i, tkr in enumerate(tickers):
            records.append((
                date, tkr, stock_sector[i], pe[i], pb[i], roe[i], de[i],
                ret_3m[i], ret_6m[i], monthly_ret[i], index_ret,
            ))

    return pd.DataFrame.from_records(
        records,
        columns=["Date", "Ticker", "Sector", "PE", "PB", "ROE", "DE",
                 "Ret_3M", "Ret_6M", "MonthlyReturn", "IndexReturn"],
    )
