"""
Cross-sectional factor scoring: valuation, momentum, quality -> composite score.

Scores are computed within each (Date x Sector) group by default, so a bank
is only compared to other banks and an IT stock only to other IT stocks at
the same point in time. This avoids the model accidentally turning into a
sector bet (e.g. penalizing all banks for running structurally higher D/E).
Falls back to Date-only grouping if `sector_col` isn't present.

Outliers are winsorized before z-scoring, and negative/loss-making P/E is
treated as "expensive" (worst valuation decile) rather than left to produce
a nonsensical sign flip.
"""

import numpy as np
import pandas as pd

WINSOR_LOWER, WINSOR_UPPER = 0.01, 0.99


def _winsorize(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(WINSOR_LOWER), s.quantile(WINSOR_UPPER)
    return s.clip(lo, hi)


def _cs_zscore(s: pd.Series) -> pd.Series:
    s = _winsorize(s)
    std = s.std()
    if not std or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def _clean_pe(s: pd.Series) -> pd.Series:
    """Negative/zero P/E (loss-making companies) can't be meaningfully ranked
    on 'cheapness' - treat them as the most expensive decile within the group
    rather than letting a negative number invert the sign."""
    s = s.astype(float).copy()
    positive_max = s[s > 0].max() if (s > 0).any() else s.max()
    s[s <= 0] = positive_max * 1.5  # worse than the most expensive real P/E
    return s


def compute_factor_scores(df: pd.DataFrame, sector_col: str = "Sector") -> pd.DataFrame:
    """
    Expects columns: Date, Ticker, PE, PB, ROE, DE, Ret_3M, Ret_6M
    (+ Sector if sector-neutral scoring is desired).
    Adds: val_score, mom_score, qual_score, composite (higher = more attractive).
    """
    df = df.copy()
    df["PE_clean"] = _clean_pe(df["PE"])

    group_cols = ["Date", sector_col] if sector_col in df.columns else ["Date"]
    by_group = df.groupby(group_cols)

    pe_z = by_group["PE_clean"].transform(_cs_zscore)
    pb_z = by_group["PB"].transform(_cs_zscore)
    ret3_z = by_group["Ret_3M"].transform(_cs_zscore)
    ret6_z = by_group["Ret_6M"].transform(_cs_zscore)
    roe_z = by_group["ROE"].transform(_cs_zscore)
    de_z = by_group["DE"].transform(_cs_zscore)

    df["val_score"] = 0.5 * -pe_z + 0.5 * -pb_z
    df["mom_score"] = 0.5 * ret3_z + 0.5 * ret6_z
    df["qual_score"] = 0.5 * roe_z + 0.5 * -de_z
    df["composite"] = (df["val_score"] + df["mom_score"] + df["qual_score"]) / 3.0
    return df.drop(columns=["PE_clean"])
