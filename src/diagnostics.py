"""
Research diagnostics that go beyond a single backtest equity curve:

- Information Coefficient (IC): per-date Spearman correlation between the
  composite score and the forward return, averaged across time. This
  answers "does the score actually predict next-period returns?" directly,
  independent of any portfolio construction choices.
- Alpha significance test: a t-test on whether the strategy's average
  monthly excess return over the benchmark is distinguishable from zero,
  since a small positive average excess return can easily be noise.
"""

import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(df: pd.DataFrame, score_col: str = "composite",
                             return_col: str = "MonthlyReturn") -> pd.DataFrame:
    """Per-date Spearman IC between score and forward return, plus a
    summary row with the mean IC, its t-stat, and p-value."""
    ic_by_date = (
        df.groupby("Date")
        .apply(lambda g: g[score_col].corr(g[return_col], method="spearman"))
        .rename("IC")
        .reset_index()
    )
    ic_series = ic_by_date["IC"].dropna()
    t_stat, p_val = stats.ttest_1samp(ic_series, 0.0) if len(ic_series) > 1 else (np.nan, np.nan)
    summary = pd.DataFrame([{
        "Date": "MEAN", "IC": ic_series.mean(),
    }])
    ic_by_date = pd.concat([ic_by_date, summary], ignore_index=True)
    return ic_by_date, {"mean_IC": round(float(ic_series.mean()), 4),
                         "IC_tstat": round(float(t_stat), 2),
                         "IC_pvalue": round(float(p_val), 4)}


def excess_return_significance(result: pd.DataFrame) -> dict:
    """One-sample t-test on (strategy return - index return) each month:
    tests whether the average monthly alpha is statistically distinguishable
    from zero, not just numerically positive."""
    excess = result["StrategyReturn"] - result["IndexReturn"]
    t_stat, p_val = stats.ttest_1samp(excess.dropna(), 0.0)
    return {
        "mean_monthly_excess_return": round(float(excess.mean()), 5),
        "excess_tstat": round(float(t_stat), 2),
        "excess_pvalue": round(float(p_val), 4),
        "interpretation": (
            "statistically significant at 5%" if p_val < 0.05
            else "NOT statistically significant at 5% - could be noise"
        ),
    }
