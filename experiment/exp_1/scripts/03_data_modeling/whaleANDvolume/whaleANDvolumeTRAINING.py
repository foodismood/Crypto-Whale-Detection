import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ======================================================
# PATHS
# ======================================================
BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"

DATASET_PATH = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume/"
    "BTCUSDT_whale_volume_dataset.parquet"
)

MODEL_DIR = f"{BASE}/model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = f"{MODEL_DIR}/BTCUSDT_xgb_whale_volume_binary_always.json"

# ======================================================
# SETTINGS
# ======================================================
TRAIN_FRACTION = 0.8

TIME_COL = "event_timestamp"
RETURN_COL = "return_30s"

# ✅ Prof-konform: alles wird BUY/SELL gelabelt (kein Drop)
# BUY  = 1 wenn return >= 0
# SELL = 0 wenn return < 0

# ======================================================
# FEATURES (exist in your new dataset)
# ======================================================
FEATURES = [
    "whale_value_usd",
    "whale_qty",
    "whale_side",
    "volume_sum_pre_30s",
    "volume_mean_pre_30s",
    "volume_std_pre_30s",
    "volume_sum_post_10s",
    "volume_sum_post_30s",
    "volume_peak_post_30s",
    "time_to_volume_peak",
    "volume_reaction_ratio",
]


def main():
    print("Loading dataset:", DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # Drop NaNs for features + return (needed for label)
    df = df.dropna(subset=FEATURES + [RETURN_COL]).copy()
    print("Samples after dropna:", len(df))

    # ✅ Binary label for ALL rows (no neutral drop)
    df["target_2class"] = (df[RETURN_COL] >= 0).astype(int)

    # Time-based split
    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[FEATURES].astype(np.float32)
    y_train = train_df["target_2class"].astype(int)

    X_val = val_df[FEATURES].astype(np.float32)
    y_val = val_df["target_2class"].astype(int)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Class weights
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    class_weights = {c: total / (2 * cnt) for c, cnt in class_counts.items()}
    sample_weight = y_train.map(class_weights)

    print("Class counts:", class_counts.to_dict())
    print("Class weights:", class_weights)

    # Model
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("Training...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluation
    probs = model.predict_proba(X_val)[:, 1]     # P(BUY)
    preds = (probs >= 0.5).astype(int)          # always act

    print("\n--- VALIDATION METRICS ---")
    print("Accuracy:", accuracy_score(y_val, preds))
    print("F1 (weighted):", f1_score(y_val, preds, average="weighted"))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))
    print("\nClassification report:")
    print(classification_report(y_val, preds, digits=4))

    model.save_model(MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)

    # Show first 10 actions
    actions = np.where(probs >= 0.5, "BUY", "SELL")
    print("\nExample actions (first 10 val probs):")
    for p, a in list(zip(probs[:10], actions[:10])):
        print(f"p_buy={p:.3f} -> {a}")


if __name__ == "__main__":
    main()
