"""
Long top-quintile backtest: each period, equal-weight the top-quintile
(highest composite score) stocks and hold for one period, versus a
benchmark return (Nifty 500 TRI in production; a synthetic index proxy in
demo mode - see README).

Includes:
- Turnover-based transaction cost modeling (round-trip cost applied on the
  fraction of the top-quintile portfolio that changes each rebalance).
- Quintile breakdown (1-5) to check the ranking is monotonic, i.e. that
  quintile 5 genuinely outperforms quintile 1 and not just the index.
- A long-short (Q5 - Q1) spread series, the standard test of whether the
  score is actually separating winners from losers.
"""

import numpy as np
import pandas as pd


def run_quintile_backtest(
    df: pd.DataFrame,
    quintile: int = 5,
    return_col: str = "MonthlyReturn",
    index_col: str = "IndexReturn",
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """cost_bps: one-way transaction cost in basis points applied to the
    fraction of the portfolio turned over each rebalance (covers brokerage +
    STT + estimated slippage). Set to 0 to see gross-of-cost performance."""
    dates = sorted(df["Date"].unique())
    rows = []
    prev_holdings: set = set()

    for d in dates:
        snap = df[df["Date"] == d]
        top = snap[snap["quintile"] == quintile]
        holdings = set(top["Ticker"])

        gross_ret = top[return_col].mean() if len(top) else np.nan
        idx_ret = snap[index_col].mean()

        if prev_holdings:
            changed = len(holdings.symmetric_difference(prev_holdings))
            turnover = changed / (2 * max(len(holdings), 1))  # one-way turnover, 0-1
        else:
            turnover = 1.0  # full turnover on the first rebalance

        cost = turnover * (cost_bps / 10_000.0)
        net_ret = gross_ret - cost

        rows.append(dict(Date=d, GrossReturn=gross_ret, Turnover=turnover,
                          Cost=cost, StrategyReturn=net_ret, IndexReturn=idx_ret,
                          NumHoldings=len(top)))
        prev_holdings = holdings

    result = pd.DataFrame(rows)
    result["StrategyEquity"] = (1 + result["StrategyReturn"]).cumprod()
    result["IndexEquity"] = (1 + result["IndexReturn"]).cumprod()
    return result


def quintile_breakdown(df: pd.DataFrame, return_col: str = "MonthlyReturn") -> pd.DataFrame:
    """Mean monthly return and annualized CAGR-equivalent by quintile (1-5),
    to check the ranking is monotonic rather than accidentally flat/inverted."""
    rows = []
    for q in sorted(df["quintile"].dropna().unique()):
        by_date = df[df["quintile"] == q].groupby("Date")[return_col].mean()
        mean_ret = by_date.mean()
        equity = (1 + by_date).cumprod()
        n_years = len(by_date) / 12
        cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
        rows.append({"Quintile": int(q), "MeanMonthlyReturn": mean_ret, "CAGR": cagr})
    return pd.DataFrame(rows).sort_values("Quintile")


def long_short_spread(df: pd.DataFrame, return_col: str = "MonthlyReturn") -> pd.DataFrame:
    """Q5 - Q1 spread series: the cleanest test of whether the composite
    score separates future winners from future losers."""
    top = df[df["quintile"] == 5].groupby("Date")[return_col].mean()
    bottom = df[df["quintile"] == 1].groupby("Date")[return_col].mean()
    spread = (top - bottom).rename("Q5_minus_Q1")
    return spread.reset_index()


def performance_metrics(result: pd.DataFrame, periods_per_year: int = 12, rf: float = 0.06) -> dict:
    metrics = {}
    for label, ret_col, eq_col in [
        ("Strategy", "StrategyReturn", "StrategyEquity"),
        ("Index", "IndexReturn", "IndexEquity"),
    ]:
        ret, eq = result[ret_col], result[eq_col]
        n_years = len(ret) / periods_per_year
        cagr = eq.iloc[-1] ** (1 / n_years) - 1
        vol = ret.std() * np.sqrt(periods_per_year)
        sharpe = (ret.mean() * periods_per_year - rf) / vol if vol > 0 else float("nan")
        downside = ret[ret < 0]
        downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) else float("nan")
        sortino = (ret.mean() * periods_per_year - rf) / downside_vol if downside_vol else float("nan")
        drawdown = eq / eq.cummax() - 1
        metrics[label] = {
            "CAGR": round(float(cagr), 4),
            "Volatility": round(float(vol), 4),
            "Sharpe": round(float(sharpe), 2),
            "Sortino": round(float(sortino), 2),
            "MaxDrawdown": round(float(drawdown.min()), 4),
        }

    metrics["Alpha_vs_Index_CAGR"] = round(
        metrics["Strategy"]["CAGR"] - metrics["Index"]["CAGR"], 4
    )
    metrics["HitRateVsIndex"] = round(
        float((result["StrategyReturn"] > result["IndexReturn"]).mean()), 3
    )
    metrics["AvgMonthlyTurnover"] = round(float(result["Turnover"].mean()), 3)
    metrics["AvgMonthlyCostDrag"] = round(float(result["Cost"].mean()), 5)
    return metrics
