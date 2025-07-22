import pandas as pd

def compute_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate momentum = percentage change over a window.
    Returns a Series aligned with df.
    """
    return df["收盘"].pct_change(periods=window)
