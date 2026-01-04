import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier

# ======================================================
# PARAMETER
# ======================================================
N_PER_CLASS = 10
WINDOW_BEFORE = 60
WINDOW_AFTER = 60
HORIZON = 30

# Trading / Plot selection thresholds (optional filter)
P_BUY_ENTER = 0.70       # pick confident BUY examples
P_SELL_ENTER = 0.70      # pick confident SELL examples

# FEATURES (must match training)
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

# ======================================================
# PATHS
# ======================================================
BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"

DATASET_PATH = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume/"
    "BTCUSDT_whale_volume_dataset.parquet"
)

MODEL_PATH = f"{BASE}/model/BTCUSDT_xgb_whale_volume_binary_always.json"

# ⚠️ geändert: Bars laden jetzt auch "volume", damit wir Volume-Spikes plotten können
BARS_PATH = (
    f"{BASE}/data/raw/Bars_1s/BTCUSDT/"
    "BTCUSDT_1s_202406_202506.parquet"
)

# ✅ SPEICHERT JETZT DIREKT HIER (ohne validation_examples Unterordner)
OUT_DIR = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume/"
    "whaleAndvolume_plots"
)
os.makedirs(OUT_DIR, exist_ok=True)

TIME_COL = "event_timestamp"
RETURN_COL = "return_30s"


# ======================================================
# LOAD MODEL (robust) -> Booster predict
# ======================================================
def load_model_booster(model_path: str):
    # Load via XGBClassifier to access booster consistently
    m = XGBClassifier()
    m.load_model(model_path)
    return m.get_booster()


def predict_p_buy(booster, X: np.ndarray) -> float:
    dmat = xgb.DMatrix(X, feature_names=FEATURES)
    # binary:logistic -> returns p_buy directly
    return float(booster.predict(dmat)[0])


# ======================================================
# PLOT FUNCTION
# ======================================================
def plot_event(event, label, idx, bars):
    row, p_buy, p_sell = event
    t0 = row[TIME_COL]

    # align timestamp to bars (1s)
    bar_idx = bars.index.searchsorted(t0)
    if bar_idx + HORIZON >= len(bars) or bar_idx <= WINDOW_BEFORE:
        return False

    entry_px = bars.iloc[bar_idx]["close"]
    exit_px = bars.iloc[bar_idx + HORIZON]["close"]

    window = bars.iloc[
        bar_idx - WINDOW_BEFORE: bar_idx + HORIZON + WINDOW_AFTER
    ].copy()

    # Return in direction of trade
    if label == "BUY":
        trade_ret = (exit_px - entry_px) / entry_px
        side = "LONG"
    else:
        trade_ret = (entry_px - exit_px) / entry_px
        side = "SHORT"

    dominance = "BUY whale" if row["whale_side"] > 0 else "SELL whale"

    # -----------------------------
    # Volume spike stats (z-score in the plotted window)
    # -----------------------------
    vol = window["volume"].astype(float)
    vol_mean = float(vol.mean())
    vol_std = float(vol.std()) if float(vol.std()) > 0 else 0.0
    vol_z = (vol - vol_mean) / vol_std if vol_std > 0 else (vol * 0.0)

    spike_z = float(vol_z.max()) if len(vol_z) else np.nan
    spike_t = vol_z.idxmax() if len(vol_z) else None

    info = (
        f"{label} | {side}\n"
        f"P(BUY): {p_buy:.2f}\n"
        f"P(SELL): {p_sell:.2f}\n\n"
        f"{dominance}\n"
        f"Whale value: {row['whale_value_usd']:,.0f} USD\n"
        f"Whale qty: {row['whale_qty']:.4f}\n"
        f"Volume reaction: {row['volume_reaction_ratio']:.2f}\n"
        f"Peak vol (post30): {row['volume_peak_post_30s']:,.0f}\n"
        f"Window spike z: {spike_z:.2f}\n"
        f"Return({HORIZON}s): {trade_ret*100:.2f}%"
    )

    # -----------------------------
    # Plot: Price + Volume spikes (2nd axis)
    # -----------------------------
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Price
    ax1.plot(window.index, window["close"], linewidth=1.2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.grid(True)

    # Entry/Exit lines
    ax1.axvline(bars.index[bar_idx], linestyle="--", label="Entry (event-aligned)")
    ax1.axvline(bars.index[bar_idx + HORIZON], linestyle="--", label="Exit")

    # Info box
    ax1.text(
        window.index[0],
        window["close"].max(),
        info,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9),
    )

    # Volume spikes (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(window.index, window["volume"], linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Volume")

    # Highlight spike point (max z-score)
    if spike_t is not None:
        ax2.axvline(spike_t, linestyle=":", alpha=0.7, label="Volume spike (max z)")

    # Combined legend (ax1 + ax2)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title(f"Validation Example – {label} #{idx} | t={t0}")
    plt.tight_layout()

    fname = os.path.join(OUT_DIR, f"validation_{label.lower()}_{idx:02d}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

    print("Saved:", fname)
    return True


# ======================================================
# MAIN
# ======================================================
def main():
    # -----------------------------
    # LOAD DATASET
    # -----------------------------
    df = pd.read_parquet(DATASET_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # time split (same idea as training)
    split_idx = int(len(df) * 0.8)
    val_df = df.iloc[split_idx:].copy()

    # remove NaNs for inference + plotting
    val_df = val_df.dropna(subset=FEATURES + [TIME_COL]).copy()

    print(f"Validation Events: {len(val_df)}")

    # -----------------------------
    # LOAD BARS (now includes volume)
    # -----------------------------
    bars = pd.read_parquet(BARS_PATH, columns=["timestamp", "close", "volume"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    # -----------------------------
    # LOAD MODEL BOOSTER
    # -----------------------------
    booster = load_model_booster(MODEL_PATH)

    # -----------------------------
    # COLLECT CONFIDENT BUY/SELL EVENTS
    # -----------------------------
    buy_events = []
    sell_events = []

    for _, row in val_df.iterrows():
        X = row[FEATURES].values.astype(np.float32).reshape(1, -1)
        p_buy = predict_p_buy(booster, X)
        p_sell = 1.0 - p_buy

        if p_buy >= P_BUY_ENTER:
            buy_events.append((row, p_buy, p_sell))
        elif p_sell >= P_SELL_ENTER:
            sell_events.append((row, p_buy, p_sell))

        if len(buy_events) >= N_PER_CLASS and len(sell_events) >= N_PER_CLASS:
            break

    # fallback: if thresholds too strict, just take top by probability
    if len(buy_events) < N_PER_CLASS:
        tmp = []
        for _, row in val_df.iterrows():
            X = row[FEATURES].values.astype(np.float32).reshape(1, -1)
            p_buy = predict_p_buy(booster, X)
            tmp.append((row, p_buy, 1.0 - p_buy))
        tmp_sorted = sorted(tmp, key=lambda x: x[1], reverse=True)
        buy_events = tmp_sorted[:N_PER_CLASS]

    if len(sell_events) < N_PER_CLASS:
        tmp = []
        for _, row in val_df.iterrows():
            X = row[FEATURES].values.astype(np.float32).reshape(1, -1)
            p_buy = predict_p_buy(booster, X)
            tmp.append((row, p_buy, 1.0 - p_buy))
        tmp_sorted = sorted(tmp, key=lambda x: x[2], reverse=True)
        sell_events = tmp_sorted[:N_PER_CLASS]

    print(f"BUY plots: {len(buy_events)} | SELL plots: {len(sell_events)}")

    # -----------------------------
    # CREATE PLOTS
    # -----------------------------
    created = 0
    for i, e in enumerate(buy_events, 1):
        created += 1 if plot_event(e, "BUY", i, bars) else 0

    for i, e in enumerate(sell_events, 1):
        created += 1 if plot_event(e, "SELL", i, bars) else 0

    print(f"\nDONE – created {created} validation plots.")
    print("Saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
