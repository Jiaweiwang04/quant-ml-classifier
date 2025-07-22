import pandas as pd

def compute_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) for a given window.
    """
    delta = df["收盘"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))
