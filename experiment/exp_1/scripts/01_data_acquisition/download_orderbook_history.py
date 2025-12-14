import os
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta


# SETTINGS

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START = datetime(2024, 6, 20)
END = datetime(2025, 6, 20)
THRESHOLD_USD = 1_000_000  # Filter: nur Trades über 100 Mio USD

# Basisverzeichnis für CSV & Parquet-Dateien
BASE_DIR = os.path.expanduser(
    "~/PycharmProjects/Crypto-Whale-Detection/experiment/exp_1/data/raw/Orderbook"
)
os.makedirs(BASE_DIR, exist_ok=True)

# Binance CSV-Spalten
COLUMNS = ["tradeId", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]


# HILFSFUNKTIONEN

def download_and_extract(symbol, year, month):
    url = f"https://data.binance.vision/data/spot/monthly/trades/{symbol}/{symbol}-trades-{year}-{month:02d}.zip"
    month_dir = os.path.join(BASE_DIR, symbol, f"{year}-{month:02d}")
    os.makedirs(month_dir, exist_ok=True)
    zip_path = os.path.join(month_dir, f"{symbol}-trades-{year}-{month:02d}.zip")
    csv_path = os.path.join(month_dir, f"{symbol}-trades-{year}-{month:02d}.csv")

    if os.path.exists(csv_path):
        print(f" CSV existiert bereits: {csv_path}")
        return csv_path

    print(f" Lade herunter: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f" Download abgeschlossen: {zip_path}")

    # Entpacken
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(month_dir)
    print(f" Entpackt: {zip_path}")

    # Zip löschen
    os.remove(zip_path)
    return csv_path

def filter_and_save_parquet(csv_path):
    print(f" Verarbeite {csv_path} ...")
    chunks = []

    try:
        for chunk in pd.read_csv(csv_path, names=COLUMNS, header=None, chunksize=1_000_000):
            chunk["value_usd"] = chunk["price"] * chunk["qty"]
            filtered = chunk[chunk["value_usd"] >= THRESHOLD_USD]
            if not filtered.empty:
                chunks.append(filtered)
    except Exception as e:
        print(f" Fehler beim Verarbeiten: {e}")
        return None

    if chunks:
        df_month = pd.concat(chunks, ignore_index=True)
        parquet_path = csv_path.replace(".csv", "_whales.parquet")
        df_month.to_parquet(parquet_path, index=False)
        print(f"Monatliches Parquet gespeichert: {parquet_path}")
    else:
        parquet_path = None
        print(f" Keine Trades über {THRESHOLD_USD} USD gefunden.")

    # CSV löschen
    os.remove(csv_path)
    print(f" CSV gelöscht: {csv_path}")
    return parquet_path


# MONATSWEISE VERARBEITUNG

current = START
parquet_files = {symbol: [] for symbol in SYMBOLS}

while current <= END:
    year = current.year
    month = current.month

    for symbol in SYMBOLS:
        csv_path = download_and_extract(symbol, year, month)
        parquet_file = filter_and_save_parquet(csv_path)
        if parquet_file:
            parquet_files[symbol].append(parquet_file)

    # Nächster Monat
    current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)


# FINALE PARQUET-ZUSAMMENFÜHRUNG

for symbol in SYMBOLS:
    final_parquet = os.path.join(BASE_DIR, f"{symbol}_{START.strftime('%Y%m%d')}-{END.strftime('%Y%m%d')}_whales.parquet")
    if parquet_files[symbol]:
        df_list = [pd.read_parquet(f) for f in parquet_files[symbol]]
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_parquet(final_parquet, index=False)
        print(f" Finale Parquet-Datei für {symbol}: {final_parquet}")

        # Monatliche Parquets löschen
        for f in parquet_files[symbol]:
            os.remove(f)
            print(f" Monatliche Parquet gelöscht: {f}")
