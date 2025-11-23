import pandas as pd
import requests
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "bars_1m")

os.makedirs(os.path.join(DATA_DIR, "BTCUSDT"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ETHUSDT"), exist_ok=True)

def fetch_klines(symbol, interval, start_str, end_str=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.to_datetime(start_str).timestamp() * 1000)
    }
    if end_str:
        params["endTime"] = int(pd.to_datetime(end_str).timestamp() * 1000)

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])

    # timestamps
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # numeric conversion
    df[["open","high","low","close","volume","quote_asset_volume"]] = \
        df[["open","high","low","close","volume","quote_asset_volume"]].astype(float)

    # -----------------------------------------
    # VWAP berechnen
    # -----------------------------------------
    df["vwap"] = df["quote_asset_volume"] / df["volume"]

    # final cleaned df
    df_final = df[[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap"
    ]].rename(columns={"open_time": "timestamp"})

    return df_final

# BTC speichern
btc_df = fetch_klines("BTCUSDT", "1m", "2025-06-19", "2025-06-20")
btc_path = os.path.join(DATA_DIR, "BTCUSDT", "2025-06-19.parquet")
btc_df.to_parquet(btc_path)
print("Saved:", btc_path)

# ETH speichern
eth_df = fetch_klines("ETHUSDT", "1m", "2025-06-19", "2025-06-20")
eth_path = os.path.join(DATA_DIR, "ETHUSDT", "2025-06-19.parquet")
eth_df.to_parquet(eth_path)
print("Saved:", eth_path)
