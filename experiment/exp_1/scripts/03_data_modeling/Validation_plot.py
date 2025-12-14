import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

# ======================================================
# PARAMETER
# ======================================================
N_PER_CLASS = 10
WINDOW_BEFORE = 60
WINDOW_AFTER = 60
HORIZON = 30

# Trading-Filter
P_UP_ENTER = 0.70
P_DOWN_ENTER = 0.75
MARGIN_UP = 0.30
MARGIN_DOWN = 0.35
P_NEUTRAL_MAX = 0.20

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

# ======================================================
# LOAD DATASET
# ======================================================
df = pd.read_parquet("BTCUSDT_volume_event_dataset.parquet")
df = df.sort_values("event_bar_timestamp").reset_index(drop=True)

split_idx = int(len(df) * 0.8)
val_df = df.iloc[split_idx:].copy()

print(f"Validation Events: {len(val_df)}")

# ======================================================
# LOAD RAW BARS
# ======================================================
bars = pd.read_parquet(
    "/Users/alperademgencer/PycharmProjects/"
    "Crypto-Whale-Detection/1/experiment/exp_1/"
    "data/raw/Bars_1s/BTCUSDT/"
    "BTCUSDT_1s_202406_202506.parquet",
    columns=["timestamp", "close"]
)
bars["timestamp"] = pd.to_datetime(bars["timestamp"])
bars = bars.set_index("timestamp").sort_index()

# ======================================================
# LOAD MODEL AS BOOSTER
# ======================================================
booster = xgb.Booster()
booster.load_model("xgb_volume_model.json")

# ======================================================
# COLLECT EVENTS BY CLASS
# ======================================================
up_events = []
down_events = []
neutral_events = []

for _, row in val_df.iterrows():

    X = row[FEATURES].values.reshape(1, -1)
    dmat = xgb.DMatrix(X, feature_names=FEATURES)
    p_down, p_neutral, p_up = booster.predict(dmat)[0]

    # Buy / Sell info
    buy_v = row["buy_volume"]
    sell_v = row["sell_volume"]
    delta_v = buy_v - sell_v
    imbalance = row["buy_sell_imbalance"]

    if (
        p_up >= P_UP_ENTER
        and (p_up - p_down) >= MARGIN_UP
        and p_neutral <= P_NEUTRAL_MAX
    ):
        up_events.append((row, p_up, p_down, p_neutral, delta_v, imbalance))

    elif (
        p_down >= P_DOWN_ENTER
        and (p_down - p_up) >= MARGIN_DOWN
        and p_neutral <= P_NEUTRAL_MAX
    ):
        down_events.append((row, p_up, p_down, p_neutral, delta_v, imbalance))

    else:
        neutral_events.append((row, p_up, p_down, p_neutral, delta_v, imbalance))

# Begrenzen
up_events = up_events[:N_PER_CLASS]
down_events = down_events[:N_PER_CLASS]
neutral_events = neutral_events[:N_PER_CLASS]

print(f"UP plots: {len(up_events)} | DOWN plots: {len(down_events)} | NEUTRAL plots: {len(neutral_events)}")

# ======================================================
# PLOT FUNCTION
# ======================================================
def plot_event(event, label, idx):

    row, p_up, p_down, p_neutral, delta_v, imbalance = event
    t0 = row["event_bar_timestamp"]

    bar_idx = bars.index.searchsorted(t0)
    if bar_idx + HORIZON >= len(bars):
        return

    entry_px = bars.iloc[bar_idx]["close"]
    exit_px = bars.iloc[bar_idx + HORIZON]["close"]

    window = bars.loc[
        t0 - pd.Timedelta(seconds=WINDOW_BEFORE):
        t0 + pd.Timedelta(seconds=HORIZON + WINDOW_AFTER)
    ]

    side = "LONG" if label == "UP" else "SHORT" if label == "DOWN" else "NO TRADE"
    ret = (exit_px - entry_px) / entry_px * (1 if label == "UP" else -1)

    dominance = "BUY dominant" if delta_v > 0 else "SELL dominant"

    info = (
        f"{label} | {side}\n"
        f"P(UP): {p_up:.2f}\n"
        f"P(DOWN): {p_down:.2f}\n"
        f"P(NEUTRAL): {p_neutral:.2f}\n\n"
        f"{dominance}\n"
        f"ΔVolume: {delta_v:,.0f} USD\n"
        f"Imbalance: {imbalance:.2f}\n"
        f"Return: {ret*100:.2f}%"
    )

    plt.figure(figsize=(14, 6))
    plt.plot(window.index, window["close"], linewidth=1.2)

    plt.axvline(t0, color="green", linestyle="--", label="Event / Entry")
    plt.axvline(t0 + pd.Timedelta(seconds=HORIZON), color="red", linestyle="--", label="Exit")

    plt.text(
        window.index[0],
        window["close"].max(),
        info,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9),
    )

    plt.title(f"Validation Example – {label} #{idx}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fname = f"validation_{label.lower()}_{idx}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f"Saved: {fname}")

# ======================================================
# CREATE PLOTS
# ======================================================
for i, e in enumerate(up_events, 1):
    plot_event(e, "UP", i)

for i, e in enumerate(down_events, 1):
    plot_event(e, "DOWN", i)

for i, e in enumerate(neutral_events, 1):
    plot_event(e, "NEUTRAL", i)

print("\nDONE – Extended validation plots created.")
