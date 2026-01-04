import os
import pandas as pd
import numpy as np

# ======================================================
# BASE PATH
# ======================================================
BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"

# ======================================================
# INPUT PATHS
# ======================================================
BARS_PATH = (
    f"{BASE}/data/raw/Bars_1s/BTCUSDT/"
    "BTCUSDT_1s_202406_202506.parquet"
)

WHALES_PATH = (
    f"{BASE}/data/raw/Orderbook/BTCUSDT/"
    "BTCUSDT_20240620-20250620_whales.parquet"
)

# ======================================================
# OUTPUT PATH
# ======================================================
OUTPUT_DIR = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = f"{OUTPUT_DIR}/BTCUSDT_whale_volume_dataset.parquet"

# ======================================================
# PARAMETERS
# ======================================================
MIN_WHALE_USD = 1_000_000

PRE_WINDOW_SECONDS = 30
POST_WINDOW_SHORT = 10
POST_WINDOW_LONG = 30

# ======================================================
# LOAD FUNCTIONS
# ======================================================
def load_bars(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    bars = pd.read_parquet(path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    if "volume" not in bars.columns:
        raise ValueError("Bars müssen 'volume' enthalten")

    return bars


def load_whales(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    whales = pd.read_parquet(path)
    whales = whales[whales["value_usd"] >= MIN_WHALE_USD].copy()
    whales = whales[whales["time"] < 10**13]

    whales["timestamp"] = pd.to_datetime(whales["time"], unit="ms")
    whales["whale_side"] = whales["isBuyerMaker"].apply(
        lambda x: -1 if x else 1
    )

    return whales.sort_values("timestamp")


# ======================================================
# DATASET BUILDER
# ======================================================
def build_whale_volume_dataset(bars, whales):
    rows = []
    bar_index = bars.index

    for _, w in whales.iterrows():
        t_whale = w["timestamp"]

        # 🔑 Map whale → nächste 1s-Bar
        idx = bar_index.searchsorted(t_whale)
        if idx <= PRE_WINDOW_SECONDS or idx + POST_WINDOW_LONG >= len(bars):
            continue

        t0 = bar_index[idx]

        # -------------------------
        # WINDOWS
        # -------------------------
        pre = bars.iloc[idx - PRE_WINDOW_SECONDS:idx]
        post_10 = bars.iloc[idx: idx + POST_WINDOW_SHORT]
        post_30 = bars.iloc[idx: idx + POST_WINDOW_LONG]

        # -------------------------
        # PRE VOLUME
        # -------------------------
        pre_sum = pre["volume"].sum()
        pre_mean = pre["volume"].mean()
        pre_std = pre["volume"].std()

        # -------------------------
        # POST VOLUME
        # -------------------------
        post_sum_10 = post_10["volume"].sum()
        post_sum_30 = post_30["volume"].sum()

        peak_volume = post_30["volume"].max()
        peak_idx = post_30["volume"].idxmax()
        time_to_peak = (peak_idx - t0).total_seconds()

        # -------------------------
        # PRICE REACTION
        # -------------------------
        px0 = bars.iloc[idx]["close"] if "close" in bars.columns else np.nan
        px30 = bars.iloc[idx + 30]["close"] if "close" in bars.columns else np.nan

        ret_30s = (px30 - px0) / px0 if px0 and px30 else np.nan

        rows.append({
            "event_timestamp": t0,

            # WHALE
            "whale_price": w["price"],
            "whale_qty": w["qty"],
            "whale_value_usd": w["value_usd"],
            "whale_side": w["whale_side"],

            # VOLUME PRE
            "volume_sum_pre_30s": pre_sum,
            "volume_mean_pre_30s": pre_mean,
            "volume_std_pre_30s": pre_std,

            # VOLUME POST
            "volume_sum_post_10s": post_sum_10,
            "volume_sum_post_30s": post_sum_30,
            "volume_peak_post_30s": peak_volume,
            "time_to_volume_peak": time_to_peak,

            # RELATIVE
            "volume_reaction_ratio": (
                post_sum_30 / pre_sum if pre_sum > 0 else np.nan
            ),

            # PRICE
            "return_30s": ret_30s,
        })

    return pd.DataFrame(rows)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    print("📥 Loading 1s bars...")
    bars = load_bars(BARS_PATH)

    print("📥 Loading whale trades...")
    whales = load_whales(WHALES_PATH)

    print(f"🐋 Whale events ≥ {MIN_WHALE_USD:,}: {len(whales)}")

    print("🛠 Building whale + volume dataset...")
    dataset = build_whale_volume_dataset(bars, whales)

    print("✅ Final dataset shape:", dataset.shape)
    print(dataset.head())

    dataset.to_parquet(OUTPUT_PATH)
    print("💾 Saved dataset to:")
    print(OUTPUT_PATH)
