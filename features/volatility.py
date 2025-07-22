import pandas as pd

def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    """
    returns = df["收盘"].pct_change()
    return returns.rolling(window=window).std()
