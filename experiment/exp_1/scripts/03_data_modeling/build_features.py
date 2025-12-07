import os
import pandas as pd
import numpy as np

BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data"

EVENTS = {
    "BTCUSDT": f"{BASE}/merged/BTCUSDT_events_1s.parquet",
    "ETHUSDT": f"{BASE}/merged/ETHUSDT_events_1s.parquet",
}

OUTPUT_DIR = f"{BASE}/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target Horizon
H = 30  # 30 Sekunden Zukunft

# Thresholds
UP = 0.001   # +0.1 %
DOWN = -0.001


def build_features(df):
    df = df.sort_values(["whale_timestamp", "timestamp"]).reset_index(drop=True)

    # ----------------------------
    # MARKET FEATURES
    # ----------------------------

    # Returns
    df["return_1s"] = np.log(df["close"] / df["close"].shift(1))
    df["return_5s"] = np.log(df["close"] / df["close"].shift(5))
    df["return_30s"] = np.log(df["close"] / df["close"].shift(30))

    df["return_1s"] = df["return_1s"].fillna(0)
    df["return_5s"] = df["return_5s"].fillna(0)
    df["return_30s"] = df["return_30s"].fillna(0)

    # Volatility 30s
    df["volatility_30s"] = df["return_1s"].rolling(30, min_periods=1).std()

    # Volume Z-Score (60s)
    vol_mean = df["volume"].rolling(60, min_periods=1).mean()
    vol_std = df["volume"].rolling(60, min_periods=1).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["volume_z"] = df["volume_z"].fillna(0)

    # Spread
    df["spread"] = df["high"] - df["low"]

    # ----------------------------
    # WHALE FEATURES
    # ----------------------------

    df["whale_buy_value"] = df["whale_value"] * (df["whale_side"] > 0).astype(int)
    df["whale_sell_value"] = df["whale_value"] * (df["whale_side"] < 0).astype(int)
    df["whale_pressure"] = df["whale_qty"] * df["whale_side"]

    # ----------------------------
    # TARGETS
    # ----------------------------

    df["future_close"] = df.groupby("whale_timestamp")["close"].shift(-H)
    df["future_return_30s"] = (df["future_close"] - df["close"]) / df["close"]

    # Klassifikation
    df["target_3class"] = 1  # neutral
    df.loc[df["future_return_30s"] > UP, "target_3class"] = 2
    df.loc[df["future_return_30s"] < DOWN, "target_3class"] = 0

    # Unvollständige Fenster entfernen
    df = df.dropna(subset=["future_return_30s"])

    return df


def main():
    for symbol, path in EVENTS.items():
        print(f"\n========== BUILDING FEATURES FOR {symbol} ==========")

        df = pd.read_parquet(path)
        print(f"Loaded {df.shape}")

        df_feat = build_features(df)

        out_path = f"{OUTPUT_DIR}/{symbol}_features_1s.parquet"
        df_feat.to_parquet(out_path)

        print(f"Saved features → {out_path}")
        print("Final shape:", df_feat.shape)
        print(df_feat.head())


if __name__ == "__main__":
    main()
