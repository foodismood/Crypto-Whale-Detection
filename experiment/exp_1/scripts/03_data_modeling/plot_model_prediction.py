import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ============================================================
# BASIS-PFADE
# ============================================================

BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"
PROCESSED_DIR = f"{BASE}/data/processed"
OUTPUT_DIR = f"{BASE}/plots/model_predictions_distinct_whales"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Features für das Modell
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
    "seconds_after",
]

TARGET = "target_3class"
CLASS_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


# ============================================================
# MODELLTRAINING
# ============================================================

def train_model_for_symbol(symbol: str):
    print(f"\n========== TRAINING MODEL FOR {symbol} ==========")

    path = f"{PROCESSED_DIR}/{symbol}_features_1s.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_model = df.dropna(subset=FEATURES + [TARGET]).copy()
    X = df_model[FEATURES].astype(np.float32)
    y = df_model[TARGET].astype(int)

    # Zeitbasierter Split 80/20
    split_idx = int(len(df_model) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Class Weights
    class_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in class_counts.items()}
    sample_weight = y_train.map(class_weight)

    print("Class counts:", class_counts.to_dict())
    print("Class weights:", class_weight)

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

    model.fit(X_train, y_train, sample_weight=sample_weight)

    probs_val = model.predict_proba(X_val)
    preds_val = np.argmax(probs_val, axis=1)

    print("\n=== VALIDATION RESULTS ===")
    print("Accuracy:", accuracy_score(y_val, preds_val))
    print("F1 (weighted):", f1_score(y_val, preds_val, average="weighted"))
    print(classification_report(y_val, preds_val))

    return df, model


# ============================================================
# BUY / SELL HELPER
# ============================================================

def get_whale_side_label(row: pd.Series) -> str:
    if row["whale_side"] > 0:
        return "BUY"
    if row["whale_side"] < 0:
        return "SELL"

    # Fallback über Buy/Sell-Values
    if row["whale_buy_value"] > row["whale_sell_value"]:
        return "BUY"
    if row["whale_sell_value"] > row["whale_buy_value"]:
        return "SELL"
    return "UNKNOWN"


# ============================================================
# EINZELNES WHALE-EVENT PLOTTEN
# ============================================================

def plot_whale_event(df: pd.DataFrame,
                     model: XGBClassifier,
                     event_row: pd.Series,
                     symbol: str,
                     out_dir: str,
                     horizon_seconds: int = 300):
    """
    Nimmt eine Zeile (echtes Whale-Event) und:
    - macht eine Modell-Prediction (UP/NEUTRAL/DOWN) für target_3class (30s-Horizont)
    - plottet die nächsten horizon_seconds Sekunden Preisverlauf
    """

    event_time = event_row["timestamp"]

    # Fiktive Prediction aus echten Features
    X_row = event_row[FEATURES].values.reshape(1, -1).astype(np.float32)
    probs = model.predict_proba(X_row)[0]
    pred_class = int(np.argmax(probs))
    pred_label = CLASS_MAP[pred_class]

    whale_size = float(event_row["whale_value"])
    whale_side_label = get_whale_side_label(event_row)

    # Preisfenster
    end_time = event_time + pd.Timedelta(seconds=horizon_seconds)
    window = df[(df["timestamp"] >= event_time) & (df["timestamp"] <= end_time)].copy()

    if window.empty:
        print(f"⚠️ Kein Datenfenster nach {event_time} gefunden – überspringe.")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(window["timestamp"], window["close"], label=f"{symbol} Close (1s)", linewidth=1.2)

    # Whale-Markierung
    plt.axvline(event_time, color="red", linestyle="--", linewidth=2, label=f"Whale {whale_side_label}")

    info_text = (
        f"Whale Side: {whale_side_label}\n"
        f"Whale Size:  ${whale_size:,.0f}\n"
        f"Prediction (30s): {pred_label}\n"
        f"P(UP):      {probs[2]:.3f}\n"
        f"P(NEUTRAL): {probs[1]:.3f}\n"
        f"P(DOWN):    {probs[0]:.3f}"
    )

    plt.text(
        window["timestamp"].iloc[0],
        window["close"].max(),
        info_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.9),
    )

    plt.title(f"{symbol} – Whale {whale_side_label} @ {event_time} (Prediction next 30s, 5min Plot)")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"{symbol}_whale_{event_time.strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Saved plot → {save_path}")


# ============================================================
# PRO SYMBOL: TOP-5 DISTINCT WHALE SIZES
# ============================================================

def run_for_symbol(symbol: str):

    df, model = train_model_for_symbol(symbol)

    # Alle Whale-Zeilen mit mindestens 3M USD
    df_events = df[df["whale_value"] >= 3_000_000].copy()

    if df_events.empty:
        print(f"⚠️ Keine Whale-Events ≥ 3M USD für {symbol} gefunden.")
        return

    # Nach Whale-Size sortieren (absteigend)
    df_events = df_events.sort_values("whale_value", ascending=False)

    # Jetzt: nur unterschiedliche Whale-Sizes wählen (damit nicht 5x derselbe Wert)
    selected_rows = []
    seen_sizes = set()

    for _, row in df_events.iterrows():
        val = float(row["whale_value"])
        # Falls Float mit Nachkommastellen, etwas runden:
        key = round(val, 2)

        if key in seen_sizes:
            continue

        seen_sizes.add(key)
        selected_rows.append(row)

        if len(selected_rows) >= 5:
            break

    if not selected_rows:
        print(f"⚠️ Konnte keine 5 verschiedenen Whale-Sizes für {symbol} wählen.")
        return

    out_dir = os.path.join(OUTPUT_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n========== PLOTTING 5 DISTINCT WHALE SIZES FOR {symbol} ==========")
    print("Ausgewählte Whale-Sizes:",
          [f"${float(r['whale_value']):,.0f}" for r in selected_rows])

    for row in selected_rows:
        plot_whale_event(df, model, row, symbol, out_dir, horizon_seconds=300)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_for_symbol("BTCUSDT")
    run_for_symbol("ETHUSDT")
    print("\n🎯 DONE: Distinct Whale Events for BTCUSDT & ETHUSDT plotted.")