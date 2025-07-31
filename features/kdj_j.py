import pandas as pd

def compute_kdj_j(df: pd.DataFrame, n: int = 9) -> pd.Series:
    """
    caculate J in KDJ
    """
    low_min = df["最低"].rolling(window=n, min_periods=1).min()
    high_max = df["最高"].rolling(window=n, min_periods=1).max()
    rsv = (df["收盘"] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return j
