import pandas as pd
import requests
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "bars_1m")

os.makedirs(os.path.join(DATA_DIR, "BTCUSDT"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ETHUSDT"), exist_ok=True)


def fetch_klines_full(symbol, interval, start_str, end_str):
    url = "https://api.binance.com/api/v3/klines"

    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts   = int(pd.to_datetime(end_str).timestamp() * 1000)

    all_df = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open","high","low","close","volume","quote_asset_volume"]] = \
            df[["open","high","low","close","volume","quote_asset_volume"]].astype(float)

        df["vwap"] = df["quote_asset_volume"] / df["volume"]

        df_final = df[[
            "open_time","open","high","low","close","volume","vwap"
        ]].rename(columns={"open_time": "timestamp"})

        all_df.append(df_final)

        # nächsten Block starten
        last_open_time = int(df["open_time"].iloc[-1].timestamp() * 1000)
        start_ts = last_open_time + 60_000  # +1 Minute

        time.sleep(0.2)  # API-Limit nicht überschreiten

        # Wenn weniger als 1000 Kerzen kam → fertig
        if len(data) < 1000:
            break

    return pd.concat(all_df, ignore_index=True)


# -----------------------------------------
# 2 Jahre Daten laden
# -----------------------------------------
START = "2023-06-20"
END   = "2025-06-20"

btc_df = fetch_klines_full("BTCUSDT", "1m", START, END)
eth_df = fetch_klines_full("ETHUSDT", "1m", START, END)

btc_path = os.path.join(DATA_DIR, "BTCUSDT", f"{START}_to_{END}.parquet")
eth_path = os.path.join(DATA_DIR, "ETHUSDT", f"{START}_to_{END}.parquet")

btc_df.to_parquet(btc_path)
eth_df.to_parquet(eth_path)

print("Saved:", btc_path)
print("Saved:", eth_path)

