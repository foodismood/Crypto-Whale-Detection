import os
import requests
import zipfile
import pandas as pd
from pathlib import Path


# ================================
# SETTINGS
# ================================

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START_YEAR = 2024
START_MONTH = 6
END_YEAR = 2025
END_MONTH = 6

BASE_DIR = Path("/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Bars_1s")


# ================================
# HILFSMETHODEN
# ================================

def ensure_dirs(symbol: str):
    """Erstellt Zielordner."""
    out_dir = BASE_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def detect_and_fix_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Erkennt automatisch, ob open_time in ms, µs oder ns ist."""

    ot = df["open_time"].astype("int64")
    max_len = ot.max().astype(str).__len__()

    if max_len == 13:
        # richtige Millisekunden
        pass
    elif max_len == 16:
        # Mikrosekunden → auf Millisekunden runter
        ot = ot // 1_000
    elif max_len == 19:
        # Nanosekunden → auf Millisekunden runter
        ot = ot // 1_000_000
    else:
        raise ValueError(f"Unbekanntes Timestamp-Format mit Länge {max_len}")

    df["timestamp"] = pd.to_datetime(ot, unit="ms", errors="coerce")
    return df


def download_month(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Lädt einen Monats-1s-Kline-Datensatz von Binance Vision."""

    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1s/{symbol}-1s-{year}-{month:02d}.zip"
    print(f" Downloading: {url}")

    tmp_zip = f"/tmp/{symbol}-1s-{year}-{month:02d}.zip"

    r = requests.get(url)
    if r.status_code != 200:
        print(f"️ Datei nicht gefunden: {url}")
        return None

    # ZIP speichern
    with open(tmp_zip, "wb") as f:
        f.write(r.content)

    # entpacken
    with zipfile.ZipFile(tmp_zip, "r") as zip_ref:
        zip_ref.extractall("/tmp")

    csv_file = f"/tmp/{symbol}-1s-{year}-{month:02d}.csv"

    if not os.path.exists(csv_file):
        print(f" CSV fehlt nach dem Entpacken: {csv_file}")
        return None

    # CSV laden
    df = pd.read_csv(
        csv_file,
        header=None,
        names=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
    )

    # Zeit reparieren
    df = detect_and_fix_timestamp(df)

    # Datentypen konvertieren
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    # sortieren
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def download_all():
    """Lädt alle Monate herunter, fügt sie zusammen und speichert pro Symbol ein Parquet."""

    for symbol in SYMBOLS:
        print("\n==============================")
        print(f"  LADER STARTE FÜR: {symbol}")
        print("==============================")

        out_dir = ensure_dirs(symbol)
        dfs = []

        year = START_YEAR
        month = START_MONTH

        while True:
            df = download_month(symbol, year, month)
            if df is not None:
                dfs.append(df)

            # nächster Monat
            if year == END_YEAR and month == END_MONTH:
                break

            month += 1
            if month > 12:
                month = 1
                year += 1

        # alles zusammenfügen
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            save_path = out_dir / f"{symbol}_1s_20240620_20250620.parquet"
            final_df.to_parquet(save_path, index=False)
            print(f" Gespeichert: {save_path}")
        else:
            print(f" Keine Daten für {symbol}")


# ================================
# START
# ================================

if __name__ == "__main__":
    download_all()
