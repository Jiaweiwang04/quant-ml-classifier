import pandas as pd
import numpy as np

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    caculate OBV
    """
    direction = np.sign(df["收盘"].diff()).fillna(0)
    obv = (direction * df["成交量"]).fillna(0).cumsum()
    return obv
