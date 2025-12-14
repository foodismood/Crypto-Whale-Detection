import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# DATA PATHS


BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"
FEATURE_DIR = f"{BASE}/data/processed"
MODEL_DIR = f"{BASE}/model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_FILES = {
    "BTCUSDT": f"{FEATURE_DIR}/BTCUSDT_features_1s.parquet",
    "ETHUSDT": f"{FEATURE_DIR}/ETHUSDT_features_1s.parquet",
}


# FEATURES THAT REALLY EXIST


FEATURES = [
    "return_1s",
    "return_5s",
    "return_30s",
    "volatility_30s",
    "volume_z",
    "spread",
    "whale_value",
    "whale_qty",
    "whale_side",
    "whale_buy_value",
    "whale_sell_value",
    "whale_pressure",
    "seconds_after"
]

TARGET = "target_3class"


# TRAINING FUNCTION


def train_symbol(symbol: str):

    print(f"\n========== TRAINING MODEL FOR {symbol} ==========")

    df = pd.read_parquet(FEATURE_FILES[symbol])
    print("Loaded dataset:", df.shape)

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Time-series split
    split_index = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # Handle class imbalance
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_counts)

    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weight = y_train.map(class_weight)

    print("Class counts:", class_counts.to_dict())
    print("Class weights:", class_weight)

    # Model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist"
    )

    print("Training...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluation
    preds = np.argmax(model.predict_proba(X_val), axis=1)

    print("\n=== RESULTS ===")
    print("Accuracy:", accuracy_score(y_val, preds))
    print("F1 Score:", f1_score(y_val, preds, average="weighted"))
    print("\nClassification Report:")
    print(classification_report(y_val, preds))

    # Save model
    model_path = f"{MODEL_DIR}/{symbol}_xgboost.json"
    model.save_model(model_path)
    print("\nModel saved to:", model_path)


# --------------------------------------------------
# RUN BOTH SYMBOLS
# --------------------------------------------------

for symbol in ["BTCUSDT", "ETHUSDT"]:
    train_symbol(symbol)
