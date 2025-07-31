import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import features
from sklearn.preprocessing import StandardScaler

def compute_features_for_stock(filepath):
    code = os.path.basename(filepath).split(".")[0]
    df = pd.read_csv(filepath, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    
    features_dict = pd.DataFrame({
        "date": df["日期"],
        "code": code,
        "close": df["收盘"],
        "momentum": features.compute_momentum(df),
        "volatility": features.compute_volatility(df),
        "rsi": features.compute_rsi(df),
        "bb_width": features.compute_bb_width(df),
        "kdj_j": features.compute_kdj_j(df),
        "macd": features.compute_macd(df),
        "obv": features.compute_obv(df)
    })
    return features_dict.dropna()

def main():
    data_dir = "data"
    all_features = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and filename != "selected_stocks.csv":
            try:
                filepath = os.path.join(data_dir, filename)
                stock_features = compute_features_for_stock(filepath)
                all_features.append(stock_features)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    merged = pd.concat(all_features, ignore_index=True)
    
    # Standardize features
    scaler = StandardScaler()
    feature_cols = ["momentum", "volatility", "rsi", "bb_width", "kdj_j", "macd", "obv"]
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

    # Save
    os.makedirs("docs", exist_ok=True)
    merged.to_csv("docs/features_2025-07-31.csv", index=False)
    print("Saved all features to docs/features_2025-07-31.csv")

if __name__ == "__main__":
    main()
