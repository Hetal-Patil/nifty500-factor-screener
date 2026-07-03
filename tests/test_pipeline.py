import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factors import compute_factor_scores
from ranking import assign_quintiles
from backtest import (
    run_quintile_backtest, performance_metrics,
    quintile_breakdown, long_short_spread,
)
from diagnostics import information_coefficient, excess_return_significance


def _toy_df(with_sector=False):
    d = {
        "Date": ["2024-01-31"] * 10,
        "Ticker": [f"S{i}" for i in range(10)],
        "PE": [10, 12, 15, 18, 20, 22, 25, 28, 30, 35],
        "PB": [1, 1.2, 1.5, 1.8, 2, 2.2, 2.5, 2.8, 3, 3.5],
        "ROE": [25, 22, 20, 18, 16, 14, 12, 10, 8, 5],
        "DE": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Ret_3M": [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "Ret_6M": [0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02],
        "MonthlyReturn": [0.05, 0.04, 0.03, 0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04],
        "IndexReturn": [0.01] * 10,
    }
    if with_sector:
        d["Sector"] = ["Financials"] * 5 + ["IT"] * 5
    return pd.DataFrame(d)


def _multi_date_df(n_months=6):
    frames = []
    for m in range(n_months):
        f = _toy_df()
        f["Date"] = f"2024-{m + 1:02d}-28"
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_composite_score_present_and_finite():
    out = compute_factor_scores(_toy_df())
    assert "composite" in out.columns
    assert out["composite"].notnull().all()


def test_best_fundamentals_get_top_quintile():
    out = compute_factor_scores(_toy_df())
    out = assign_quintiles(out)
    assert out.loc[out["Ticker"] == "S0", "quintile"].iloc[0] == 5


def test_sector_neutral_scoring_compares_within_sector():
    # S5 (Sector=IT) has the best fundamentals within IT (S5-S9), but worse
    # than several Financials stocks in absolute terms -> should still score
    # best WITHIN its own sector once scoring is sector-neutral.
    out = compute_factor_scores(_toy_df(with_sector=True))
    it_rows = out[out["Sector"] == "IT"]
    assert it_rows.loc[it_rows["Ticker"] == "S5", "composite"].iloc[0] == it_rows["composite"].max()


def test_negative_pe_treated_as_expensive_not_inverted():
    df = _toy_df()
    df["PB"] = 2.0  # hold P/B constant across all stocks so only P/E drives val_score
    df.loc[df["Ticker"] == "S0", "PE"] = -5  # loss-making, was previously the cheapest
    out = compute_factor_scores(df)
    s0_val = out.loc[out["Ticker"] == "S0", "val_score"].iloc[0]
    other_val = out.loc[out["Ticker"] != "S0", "val_score"]
    assert s0_val < other_val.min()  # penalized, not rewarded, for negative P/E


def test_backtest_runs_and_produces_metrics():
    out = compute_factor_scores(_toy_df())
    out = assign_quintiles(out)
    result = run_quintile_backtest(out)
    assert len(result) == 1
    metrics = performance_metrics(result, periods_per_year=12)
    assert "Strategy" in metrics and "Index" in metrics
    assert "AvgMonthlyTurnover" in metrics


def test_quintile_breakdown_and_long_short_spread():
    out = compute_factor_scores(_multi_date_df())
    out = assign_quintiles(out)
    qb = quintile_breakdown(out)
    assert set(qb["Quintile"]) == {1, 2, 3, 4, 5}
    spread = long_short_spread(out)
    assert "Q5_minus_Q1" in spread.columns
    assert len(spread) == 6


def test_information_coefficient_runs():
    out = compute_factor_scores(_multi_date_df())
    ic_by_date, summary = information_coefficient(out)
    assert "mean_IC" in summary
    assert not np.isnan(summary["mean_IC"])


def test_excess_return_significance_runs():
    out = compute_factor_scores(_multi_date_df())
    out = assign_quintiles(out)
    result = run_quintile_backtest(out)
    sig = excess_return_significance(result)
    assert "interpretation" in sig
