import pandas as pd
import requests

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
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df

# Beispiel: BTC 1-Minute Bars
btc_df = fetch_klines("BTCUSDT", "1m", "2025-06-19", "2025-06-20")
btc_df.to_parquet("data/Raw/Bars_1m/BTCUSDT/2025-06-19.parquet")

# Beispiel: ETH 1-Minute Bars
eth_df = fetch_klines("ETHUSDT", "1m", "2025-06-19", "2025-06-20")
eth_df.to_parquet("data/Raw/Bars_1m/ETHUSDT/2025-06-19.parquet")
