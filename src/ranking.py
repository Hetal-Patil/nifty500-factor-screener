"""Cross-sectional quintile bucketing on a composite score, per date."""

import pandas as pd


def assign_quintiles(df: pd.DataFrame, score_col: str = "composite") -> pd.DataFrame:
    """Adds a `quintile` column (1 = worst, 5 = best) computed within each Date."""
    df = df.copy()
    df["quintile"] = df.groupby("Date")[score_col].transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1
    )
    return df
