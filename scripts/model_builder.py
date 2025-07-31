import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


FEATURE_PATH = "docs/features_2025-07-31.csv"
MODEL_OUTPUT_PATH = "models/random_forest_model.pkl"

def main():

    df = pd.read_csv(FEATURE_PATH)
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])


    # generate label
    df = df.sort_values(by=["code", "date"]).reset_index(drop=True)
    df["future_close"] = df.groupby("code")["close"].shift(-5)
    df["future_return"] = df["future_close"] / df["close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)
    df = df.dropna()


    exclude_cols = ["code", "date", "close", "future_close", "future_return", "label"]
    X = df[[col for col in df.columns if col not in exclude_cols]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)


    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_OUTPUT_PATH)


    print(f"\n model saved to {MODEL_OUTPUT_PATH}")


if __name__=="__main__":
    main()
