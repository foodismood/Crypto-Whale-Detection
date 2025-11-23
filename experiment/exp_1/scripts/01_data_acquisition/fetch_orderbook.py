import requests
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Projekt- und Datenordner
# -----------------------------
BASE_DIR = "/Users/sgenk/PycharmProjects/Crypto-Whale-Detection"
DATA_DIR = os.path.join(BASE_DIR, "experiment", "exp_1", "data", "Raw", "Orderbook")

# Orderbook-Ordner erstellen
os.makedirs(os.path.join(DATA_DIR, "BTCUSDT"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "ETHUSDT"), exist_ok=True)

# -----------------------------
# Funktion: Orderbook abrufen und nur Whales filtern
# -----------------------------
def fetch_whale_orderbook(symbol, limit=100, threshold_usd=100000):
    """
    Holt das aktuelle Orderbook von Binance und filtert nur Orders > threshold_usd.
    """
    url = "https://api.binance.com/api/v3/depth"
    params = {"symbol": symbol, "limit": limit}
    resp = requests.get(url, params=params)
    data = resp.json()

    # DataFrames vorbereiten
    bids = pd.DataFrame(data["bids"], columns=["price", "quantity"]).astype(float)
    asks = pd.DataFrame(data["asks"], columns=["price", "quantity"]).astype(float)

    # Dollar-Wert berechnen
    bids["value_usd"] = bids["price"] * bids["quantity"]
    asks["value_usd"] = asks["price"] * asks["quantity"]

    # Nur Orders >= threshold
    bids = bids[bids["value_usd"] >= threshold_usd]
    asks = asks[asks["value_usd"] >= threshold_usd]

    # Side-Spalte
    bids["side"] = "bid"
    asks["side"] = "ask"

    # Symbol & Timestamp
    df = pd.concat([bids, asks], ignore_index=True)
    df["symbol"] = symbol
    df["timestamp"] = pd.Timestamp.now()

    return df

# -----------------------------
# BTC Whales speichern
# -----------------------------
btc_df = fetch_whale_orderbook("BTCUSDT", limit=100)
btc_file = os.path.join(DATA_DIR, "BTCUSDT", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_whales.parquet")
btc_df.to_parquet(btc_file)
print("BTC Whales gespeichert:", btc_file)

# -----------------------------
# ETH Whales speichern
# -----------------------------
eth_df = fetch_whale_orderbook("ETHUSDT", limit=100)
eth_file = os.path.join(DATA_DIR, "ETHUSDT", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_whales.parquet")
eth_df.to_parquet(eth_file)
print("ETH Whales gespeichert:", eth_file)