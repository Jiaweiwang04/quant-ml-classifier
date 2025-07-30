import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import os


FEATURE_PATH = "docs/features_2025-07-22.csv"
MODEL_PATH = "models/random_forest_model.pkl"
REPORT_OUTPUT_PATH = "docs/model_eval_report.txt"
ROC_PLOT_PATH = "docs/roc_curve.png"
FEATURE_IMPORTANCE_PATH = "docs/feature_importance.png"


df = pd.read_csv(FEATURE_PATH)
df["code"] = df["code"].astype(str)
df["date"] = pd.to_datetime(df["date"])

def load_close_prices():
    close_dict = {}
    for filename in os.listdir("data"):
        if filename.endswith(".csv") and filename != "selected_stocks.csv":
            code = filename.replace(".csv", "")
            try:
                raw = pd.read_csv(os.path.join("data", filename), parse_dates=["日期"])
                raw = raw[["日期", "收盘"]].rename(columns={"日期": "date", "收盘": "close"})
                raw["code"] = code
                close_dict[code] = raw
            except Exception as e:
                print(f"Failed loading close for {code}: {e}")
    return pd.concat(close_dict.values(), ignore_index=True)


close_df = load_close_prices()
df = pd.merge(df, close_df, on=["code", "date"], how="left")

df = df.sort_values(by=["code", "date"]).reset_index(drop=True)
df["future_close"] = df.groupby("code")["close"].shift(-5)
df["future_return"] = df["future_close"] / df["close"] - 1
df["label"] = (df["future_return"] > 0).astype(int)
df = df.dropna(subset=["momentum", "volatility", "rsi", "label"])


X = df[["momentum", "volatility", "rsi"]]
y = df["label"]


clf = joblib.load(MODEL_PATH)


y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)[:, 1]


report = classification_report(y, y_pred)
auc = roc_auc_score(y, y_proba)

print(report)
print(f"AUC: {auc:.4f}")


with open(REPORT_OUTPUT_PATH, "w") as f:
    f.write(report)
    f.write(f"\nAUC: {auc:.4f}\n")


fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(ROC_PLOT_PATH)
plt.close()


feat_imp = pd.Series(clf.feature_importances_, index=X.columns)
plt.figure()
feat_imp.sort_values(ascending=True).plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PATH)
plt.close()

print(f"ROC Curve saved to {ROC_PLOT_PATH}")
print(f"Feature Importance saved to {FEATURE_IMPORTANCE_PATH}")
print(f"Evaluation report saved to {REPORT_OUTPUT_PATH}")
