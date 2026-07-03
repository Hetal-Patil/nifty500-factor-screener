"""
Nifty 500 Factor Screener - CLI entry point.

Usage:
    python src/main.py --mode demo
    python src/main.py --mode live   # requires wiring in a fundamentals feed, see data_loader.py

Demo mode runs the full research pipeline end-to-end:
  data -> sector-neutral factor scores -> quintiles -> backtest (net of
  estimated costs) -> quintile monotonicity check -> long-short spread ->
  information coefficient -> alpha significance test -> charts.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import generate_demo_data
from factors import compute_factor_scores
from ranking import assign_quintiles
from backtest import (
    run_quintile_backtest, performance_metrics,
    quintile_breakdown, long_short_spread,
)
from diagnostics import information_coefficient, excess_return_significance

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main(mode: str = "demo") -> dict:
    OUT_DIR.mkdir(exist_ok=True)

    if mode == "demo":
        df = generate_demo_data()
    else:
        raise SystemExit(
            "Live mode needs a fundamentals feed merged in first - "
            "see fetch_live_prices() and the module docstring in data_loader.py."
        )

    df = compute_factor_scores(df)   # sector-neutral, winsorized, negative-P/E handled
    df = assign_quintiles(df)
    df.to_csv(OUT_DIR / "scored_universe.csv", index=False)

    result = run_quintile_backtest(df, cost_bps=10.0)
    result.to_csv(OUT_DIR / "backtest_results.csv", index=False)
    metrics = performance_metrics(result)

    q_breakdown = quintile_breakdown(df)
    q_breakdown.to_csv(OUT_DIR / "quintile_breakdown.csv", index=False)

    spread = long_short_spread(df)
    spread.to_csv(OUT_DIR / "long_short_spread.csv", index=False)

    ic_by_date, ic_summary = information_coefficient(df)
    ic_by_date.to_csv(OUT_DIR / "information_coefficient.csv", index=False)

    sig = excess_return_significance(result)

    diagnostics = {**metrics, "InformationCoefficient": ic_summary, "AlphaSignificance": sig}
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    # --- Chart 1: equity curve, net of estimated transaction costs ---
    plt.figure(figsize=(9, 5))
    plt.plot(result["Date"], result["StrategyEquity"], label="Long Top Quintile (net of est. costs)", linewidth=2)
    plt.plot(result["Date"], result["IndexEquity"], label="Nifty 500 TRI proxy (benchmark)", linewidth=2, linestyle="--")
    plt.title("Factor Strategy (Top Quintile) vs Benchmark - synthetic demo data")
    plt.xlabel("Date"); plt.ylabel("Growth of Rs 1"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "equity_curve.png", dpi=150); plt.close()

    # --- Chart 2: quintile monotonicity ---
    plt.figure(figsize=(7, 4.5))
    plt.bar(q_breakdown["Quintile"].astype(str), q_breakdown["CAGR"] * 100)
    plt.title("CAGR by Quintile (should rise monotonically 1 -> 5)")
    plt.xlabel("Quintile (1=worst score, 5=best score)"); plt.ylabel("CAGR (%)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quintile_breakdown.png", dpi=150); plt.close()

    # --- Chart 3: rolling 12M Sharpe of the strategy ---
    roll = result.set_index("Date")["StrategyReturn"]
    rolling_sharpe = (roll.rolling(12).mean() * 12) / (roll.rolling(12).std() * (12 ** 0.5))
    plt.figure(figsize=(9, 4))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values)
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.title("Rolling 12-Month Sharpe Ratio - Strategy")
    plt.xlabel("Date"); plt.ylabel("Rolling Sharpe"); plt.tight_layout()
    plt.savefig(OUT_DIR / "rolling_sharpe.png", dpi=150); plt.close()

    print(json.dumps(diagnostics, indent=2))
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    args = parser.parse_args()
    main(args.mode)
