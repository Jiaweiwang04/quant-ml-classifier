import pandas as pd

def compute_bb_width(df: pd.DataFrame, period: int = 20, std_factor: float = 2.0) -> pd.Series:
    """
    caculate the width of Bollinger Band
    """
    ma = df["收盘"].rolling(window=period).mean()
    std = df["收盘"].rolling(window=period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    width = (upper - lower) / ma
    return width
