import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


# ==============================================================================
# SETTINGS – Pfade anpassen
# ==============================================================================

BASE_DIR = Path("/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1")
IMG_DIR = BASE_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BARS_PATH = {
    "BTCUSDT": BASE_DIR / "data/raw/Bars_1m/BTCUSDT/2023-06-20_to_2025-06-20.parquet",
    "ETHUSDT": BASE_DIR / "data/raw/Bars_1m/ETHUSDT/2023-06-20_to_2025-06-20.parquet"
}

WHALES_PATH = {
    "BTCUSDT": BASE_DIR / "data/raw/Orderbook/BTCUSDT/BTCUSDT_20240620-20250620_whales.parquet",
    "ETHUSDT": BASE_DIR / "data/raw/Orderbook/ETHUSDT/ETHUSDT_20240620-20250620_whales.parquet"
}



# ==============================================================================
# 🔧 UNIVERSAL TIMESTAMP NORMALIZER (ms, s, µs → auto fix)
# ==============================================================================

def normalize_timestamp_column(df):
    """
    Detect and fix timestamp columns (ms, s, µs).
    Result: df['timestamp'] as datetime.
    """
    possible_cols = ["timestamp", "time", "T", "trade_time", "event_time", "ts"]

    found = None
    for col in possible_cols:
        if col in df.columns:
            found = col
            break

    if found is None:
        raise ValueError(f"❌ No timestamp column found. Columns: {df.columns}")

    col = found

    # If numbers → UNIX format detection
    if pd.api.types.is_numeric_dtype(df[col]):
        max_digits = df[col].astype(str).str.len().max()

        if max_digits == 10:
            df["timestamp"] = pd.to_datetime(df[col], unit="s")
        elif max_digits == 13:
            df["timestamp"] = pd.to_datetime(df[col], unit="ms")
        elif max_digits == 16:
            df["timestamp"] = pd.to_datetime(df[col], unit="us")
        else:
            raise ValueError(f"❌ Unsupported timestamp format ({max_digits} digits).")
    else:
        df["timestamp"] = pd.to_datetime(df[col], errors="coerce")

    if df["timestamp"].isna().all():
        raise ValueError("❌ Timestamp conversion failed.")

    return df



# ==============================================================================
# 📈 PLOT: EXACT 30 MINUTES AFTER A WHALE EVENT
# ==============================================================================

def plot_after_whale_event(df_bars, whale_timestamp, symbol, save_path, minutes_after=30):
    df_bars = normalize_timestamp_column(df_bars)
    whale_ts = pd.to_datetime(whale_timestamp)

    # --------------------------------------------------------------
    # FIND FIRST BAR >= WHALE TIMESTAMP
    # --------------------------------------------------------------
    start_idx = df_bars.index[df_bars["timestamp"] >= whale_ts].min()

    if pd.isna(start_idx):
        print(f"⚠️ No bars found after whale event {whale_ts}")
        return

    # EXACT 30 BARS
    end_idx = start_idx + minutes_after
    subset = df_bars.loc[start_idx:end_idx].copy()

    if subset.empty:
        print(f"⚠️ No bars available for window after {whale_ts}")
        return

    subset = subset.reset_index(drop=True)

    # --------------------------------------------------------------
    # PLOT
    # --------------------------------------------------------------
    plt.figure(figsize=(18, 9))
    plt.plot(subset.index, subset["close"], label=f"{symbol} Close", color="blue")

    # Whale event is always x=0
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Whale Event")

    # Mark new days
    day_change = subset["timestamp"].dt.date.ne(subset["timestamp"].dt.date.shift())
    for i, is_new in enumerate(day_change):
        if i > 0 and is_new:
            plt.axvline(i, color="green", linestyle="--", alpha=0.6)

    # Timestamp labels
    tick_positions = subset.index[::max(1, len(subset) // 10)]
    tick_labels = subset.loc[tick_positions, "timestamp"].dt.strftime('%Y-%m-%d %H:%M')

    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    plt.title(f"{symbol} — 30 Minutes AFTER Whale Event ({whale_ts})")
    plt.xlabel("Minutes After Whale")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"✅ Saved plot: {save_path}")



# ==============================================================================
# MAIN PIPELINE — Loop through Whale Events for BTC & ETH
# ==============================================================================

if __name__ == "__main__":

    for symbol in ["BTCUSDT", "ETHUSDT"]:

        print("\n=======================================")
        print(f"📡 PROCESSING WHALES FOR {symbol}")
        print("=======================================\n")

        # Load data
        df_whales = pd.read_parquet(WHALES_PATH[symbol])
        df_bars = pd.read_parquet(BARS_PATH[symbol])

        # Normalize whale timestamps
        df_whales = normalize_timestamp_column(df_whales)

        if df_whales.empty:
            print(f"⚠️ No whale events found for {symbol}")
            continue

        # Loop through all whale events
        for idx, whale in df_whales.iterrows():

            whale_ts = whale["timestamp"]
            save_name = f"{symbol}_whale_{idx}_{whale_ts.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            save_path = IMG_DIR / save_name

            print(f"📈 Plotting whale event #{idx} at {whale_ts}")

            plot_after_whale_event(
                df_bars=df_bars,
                whale_timestamp=whale_ts,
                symbol=symbol,
                save_path=save_path,
                minutes_after=30
            )