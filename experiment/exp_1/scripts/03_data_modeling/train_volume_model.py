import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------
# FEATURES & TARGET
# -----------------------------
FEATURES = [
    "volume_z",
    "volume_usd",
    "buy_volume",
    "sell_volume",
    "buy_sell_imbalance",
    "return_pre_1s",
    "return_pre_5s",
    "volatility_pre_30s",
]

TARGET = "target_3class"
TRAIN_FRACTION = 0.8

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_parquet("BTCUSDT_volume_event_dataset.parquet")
df = df.sort_values("event_bar_timestamp").reset_index(drop=True)

print(f"Gesamt Samples: {len(df)}")

# -----------------------------
# DROP NaNs
# -----------------------------
df = df.dropna(subset=FEATURES + [TARGET])

# -----------------------------
# TIME-BASED SPLIT
# -----------------------------
split_idx = int(len(df) * TRAIN_FRACTION)

train_df = df.iloc[:split_idx]
val_df = df.iloc[split_idx:]

X_train = train_df[FEATURES].astype(np.float32)
y_train = train_df[TARGET].astype(int)

X_val = val_df[FEATURES].astype(np.float32)
y_val = val_df[TARGET].astype(int)

print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
class_counts = y_train.value_counts().sort_index()
total = len(y_train)
n_classes = len(class_counts)

class_weights = {c: total / (n_classes * cnt) for c, cnt in class_counts.items()}
sample_weight = y_train.map(class_weights)

print("Class Weights:", class_weights)

# -----------------------------
# MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train, sample_weight=sample_weight)

# -----------------------------
# EVALUATION
# -----------------------------
probs = model.predict_proba(X_val)
preds = np.argmax(probs, axis=1)

print("\n--- VALIDATION METRICS ---")
print("Accuracy:", accuracy_score(y_val, preds))
print("F1 (weighted):", f1_score(y_val, preds, average="weighted"))
print("\nClassification Report:")
print(classification_report(y_val, preds, digits=4))

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_model("xgb_volume_model.json")
print("\nModell gespeichert: xgb_volume_model.json")
