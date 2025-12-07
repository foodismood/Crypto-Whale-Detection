import os
import pandas as pd

BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data"

BARS = {
    "BTCUSDT": f"{BASE}/raw/Bars_1s/BTCUSDT/BTCUSDT_1s_20240620_20250620.parquet",
    "ETHUSDT": f"{BASE}/raw/Bars_1s/ETHUSDT/ETHUSDT_1s_20240620_20250620.parquet",
}

WHALES = {
    "BTCUSDT": f"{BASE}/raw/Orderbook/BTCUSDT/BTCUSDT_20240620-20250620_whales.parquet",
    "ETHUSDT": f"{BASE}/raw/Orderbook/ETHUSDT/ETHUSDT_20240620-20250620_whales.parquet",
}

#  Speichern jetzt unter "merged"
OUTPUT_DIR = f"{BASE}/merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SECONDS = 300
MIN_WHALE_USD = 1_000_000


def load_bars(path):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def load_whales(path):
    df = pd.read_parquet(path)
    df = df[df["value_usd"] >= MIN_WHALE_USD].copy()
    df = df[df["time"] < 10**13]
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms")
    df["side"] = df["isBuyerMaker"].apply(lambda x: -1 if x else 1)
    return df.sort_values("timestamp")


def extract_windows(bars, whales, symbol):
    events = []

    for idx, w in whales.iterrows():
        t0 = w["timestamp"]
        t1 = t0 + pd.Timedelta(seconds=WINDOW_SECONDS)

        window = bars[(bars["timestamp"] >= t0) & (bars["timestamp"] <= t1)].copy()
        if window.empty:
            continue

        window["whale_timestamp"] = t0
        window["whale_price"] = w["price"]
        window["whale_qty"] = w["qty"]
        window["whale_value"] = w["value_usd"]
        window["whale_side"] = w["side"]
        window["seconds_after"] = (window["timestamp"] - t0).dt.total_seconds()

        events.append(window)

    if not events:
        return None

    df_events = pd.concat(events, ignore_index=True)
    print(f"{symbol}: {df_events.shape[0]} rows extracted.")
    return df_events


def main():
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n========== PROCESSING {symbol} ==========")

        bars = load_bars(BARS[symbol])
        whales = load_whales(WHALES[symbol])

        if whales.empty:
            print(f"{symbol}: No whales ≥ {MIN_WHALE_USD}")
            continue

        events = extract_windows(bars, whales, symbol)
        if events is None:
            print(f"{symbol}: No event windows found.")
            continue

        out_path = f"{OUTPUT_DIR}/{symbol}_events_1s.parquet"
        events.to_parquet(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
