import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


FEATURE_PATH = "docs/features_2025-07-22.csv"
ORIGINAL_DATA_PATH = "data"
MODEL_OUTPUT_PATH = "models/random_forest_model.pkl"
REPORT_OUTPUT_PATH = "docs/model_report.txt"


df = pd.read_csv(FEATURE_PATH)
df["code"] = df["code"].astype(str)
df["date"] = pd.to_datetime(df["date"])


def load_close_prices():
    close_dict = {}
    for filename in os.listdir(ORIGINAL_DATA_PATH):
        if filename.endswith(".csv") and filename != "selected_stocks.csv":
            code = filename.replace(".csv", "")
            try:
                raw = pd.read_csv(os.path.join(ORIGINAL_DATA_PATH, filename), parse_dates=["日期"])
                raw = raw[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})
                raw["code"] = code
                close_dict[code] = raw
            except Exception as e:
                print(f"Failed loading close for {code}: {e}")
    return pd.concat(close_dict.values(), ignore_index=True)


close_df = load_close_prices()
df = pd.merge(df, close_df, on=["code", "date"], how="left")

# generate label
df = df.sort_values(by=["code", "date"]).reset_index(drop=True)
df["future_close"] = df.groupby("code")["close"].shift(-5)
df["future_return"] = df["future_close"] / df["close"] - 1
df["label"] = (df["future_return"] > 0).astype(int)
df = df.dropna(subset=["momentum", "volatility", "rsi", "label"])


X = df[["momentum", "volatility", "rsi"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)

print("result：")
print(report)


os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_OUTPUT_PATH)


os.makedirs("docs", exist_ok=True)
with open(REPORT_OUTPUT_PATH, "w") as f:
    f.write(report)

print(f"\n model saved to {MODEL_OUTPUT_PATH}")
print(f"report saved to {REPORT_OUTPUT_PATH}")
