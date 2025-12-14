import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
PLOT_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURES = [
    "return_1m",
    "return_5m",
    "volatility_5m",
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

EASY_NAMES = ["DOWN", "NEUTRAL", "UP"]


def evaluate_symbol(symbol: str):

    print("\n==============================")
    print(f" AUSWERTUNG FÜR {symbol}")
    print("==============================")

    path = os.path.join(PROCESSED_DIR, f"{symbol}_features_1s.parquet")
    df = pd.read_parquet(path)

    df_eval = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df_eval[FEATURES].astype(np.float32)
    y = df_eval[TARGET].astype(int)

    split_idx = int(len(df_eval) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)

    treffsicherheit = precision_score(y_test, preds, average="weighted", zero_division=0)
    erkennungsrate = recall_score(y_test, preds, average="weighted", zero_division=0)
    gesamtqualitaet = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("\nErgebnisse (vereinfacht):")
    print(f"Treffsicherheit (Precision): {treffsicherheit:.3f}")
    print(f"Erkennungsrate (Recall):     {erkennungsrate:.3f}")
    print(f"Gesamtqualität (F1-Score):   {gesamtqualitaet:.3f}")
    print(f"Gesamt-Genauigkeit:          {acc:.3f}")

    print("\nDetailierte Auswertung pro Klasse:")
    print(classification_report(y_test, preds, target_names=EASY_NAMES, zero_division=0))

    cm = confusion_matrix(y_test, preds)

    print("Konfusionsmatrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Konfusionsmatrix – {symbol}")
    plt.colorbar()
    plt.xticks([0, 1, 2], EASY_NAMES)
    plt.yticks([0, 1, 2], EASY_NAMES)
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"confusion_matrix_{symbol}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Gespeichert unter: {out_path}")


if __name__ == "__main__":
    evaluate_symbol("BTCUSDT")
    evaluate_symbol("ETHUSDT")
    print("Fertig.")