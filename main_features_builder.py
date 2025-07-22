import pandas as pd
import os
from features.momentum import compute_momentum
from features.volatility import compute_volatility
from features.rsi import compute_rsi
from sklearn.preprocessing import StandardScaler

def compute_features_for_stock(filepath):
    code = os.path.basename(filepath).split(".")[0]
    df = pd.read_csv(filepath, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    
    features = pd.DataFrame({
        "date": df["日期"],
        "code": code,
        "momentum": compute_momentum(df),
        "volatility": compute_volatility(df),
        "rsi": compute_rsi(df)
    })
    return features.dropna()

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
    feature_cols = ["momentum", "volatility", "rsi"]
    merged[feature_cols] = scaler.fit_transform(merged[feature_cols])

    # Save
    os.makedirs("docs", exist_ok=True)
    merged.to_csv("docs/features_2025-07-22.csv", index=False)
    print("Saved all features to docs/features_2025-07-22.csv")

if __name__ == "__main__":
    main()
