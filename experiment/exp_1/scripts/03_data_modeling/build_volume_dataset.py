import pandas as pd
import numpy as np

# -----------------------------
# PARAMETER
# -----------------------------
PRE_RETURN_WINDOWS = [1, 5]      # Sekunden VOR dem Event
PRE_VOL_WINDOW = 30              # Sekunden VOR dem Event
TARGET_HORIZON = 30              # Sekunden NACH dem Event
TARGET_THRESHOLD = 0.001         # 0.1 %


def build_dataset(bars_path, events_path):

    # -----------------------------
    # RAW BARS LADEN
    # -----------------------------
    bars = pd.read_parquet(
        bars_path,
        columns=["timestamp", "close"]
    )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp").sort_index()

    # -----------------------------
    # EVENTS LADEN
    # -----------------------------
    events = pd.read_parquet(events_path)
    events["volume_event_timestamp"] = pd.to_datetime(
        events["volume_event_timestamp"]
    )

    rows = []

    # -----------------------------
    # PRO EVENT
    # -----------------------------
    for _, evt in events.iterrows():

        t_evt = evt["volume_event_timestamp"]

        idx = bars.index.searchsorted(t_evt)
        if idx < PRE_VOL_WINDOW or idx + TARGET_HORIZON >= len(bars):
            continue

        t0 = bars.index[idx]
        px0 = bars.iloc[idx]["close"]

        row = {
            "event_bar_timestamp": t0,
            "volume_z": evt["volume_z"],
            "volume_usd": evt["volume_usd"],
            "buy_volume": evt["buy_volume"],
            "sell_volume": evt["sell_volume"],
        }

        # --- RETURNS VOR DEM EVENT ---
        for w in PRE_RETURN_WINDOWS:
            px_prev = bars.iloc[idx - w]["close"]
            row[f"return_pre_{w}s"] = (px0 - px_prev) / px_prev

        # --- VOLATILITÄT VOR DEM EVENT ---
        win = bars.iloc[idx - PRE_VOL_WINDOW:idx]["close"]
        row["volatility_pre_30s"] = win.pct_change().std()

        # --- BUY / SELL IMBALANCE ---
        denom = row["buy_volume"] + row["sell_volume"]
        row["buy_sell_imbalance"] = (
            (row["buy_volume"] - row["sell_volume"]) / denom
            if denom > 0 else 0.0
        )

        # --- TARGET ---
        px_h = bars.iloc[idx + TARGET_HORIZON]["close"]
        ret_h = (px_h - px0) / px0

        if ret_h > TARGET_THRESHOLD:
            row["target_3class"] = 2     # UP
        elif ret_h < -TARGET_THRESHOLD:
            row["target_3class"] = 0     # DOWN
        else:
            row["target_3class"] = 1     # NEUTRAL

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    BARS_PATH = (
        "/Users/alperademgencer/PycharmProjects/"
        "Crypto-Whale-Detection/1/experiment/exp_1/"
        "data/raw/Bars_1s/BTCUSDT/"
        "BTCUSDT_1s_202406_202506.parquet"
    )

    EVENTS_PATH = "BTCUSDT_volume_events.parquet"

    dataset = build_dataset(BARS_PATH, EVENTS_PATH)

    print("Dataset size:", len(dataset))
    print(dataset.head())

    dataset.to_parquet("BTCUSDT_volume_event_dataset.parquet")
