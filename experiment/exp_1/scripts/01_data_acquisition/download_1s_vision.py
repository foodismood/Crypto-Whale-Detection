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

BASE_DIR = Path(
    "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Bars_1s"
)


# ================================
# HILFSFUNKTIONEN
# ================================

def ensure_dirs(symbol: str):
    out_dir = BASE_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def detect_and_fix_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    ot = df["open_time"].astype("int64")
    max_len = ot.max().astype(str).__len__()

    if max_len == 13:       # ms
        pass
    elif max_len == 16:     # µs → ms
        ot = ot // 1_000
    elif max_len == 19:     # ns → ms
        ot = ot // 1_000_000
    else:
        raise ValueError(f"Unbekanntes Timestamp-Format: {max_len}")

    df["timestamp"] = pd.to_datetime(ot, unit="ms", errors="coerce")
    return df


def download_month(symbol: str, year: int, month: int):
    url = (
        f"https://data.binance.vision/data/spot/monthly/klines/"
        f"{symbol}/1s/{symbol}-1s-{year}-{month:02d}.zip"
    )

    zip_path = f"/tmp/{symbol}-1s-{year}-{month:02d}.zip"

    # --------------------------------------------------
    # ZIP nur laden, wenn es noch nicht existiert
    # --------------------------------------------------
    if not os.path.exists(zip_path):
        print(f"📥 Downloading: {url}")
        r = requests.get(url)
        if r.status_code != 200:
            print(f"⚠️ Datei nicht gefunden: {url}")
            return None
        with open(zip_path, "wb") as f:
            f.write(r.content)
    else:
        print(f"⏩ ZIP existiert bereits – überspringe Download {year}-{month:02d}")

    # --------------------------------------------------
    # CSV DIREKT AUS ZIP LESEN (KEIN extractall!)
    # --------------------------------------------------
    print(f"📖 Lese CSV direkt aus ZIP: {symbol} {year}-{month:02d}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_name = zip_ref.namelist()[0]
        with zip_ref.open(csv_name) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "num_trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ]
            )

    # --------------------------------------------------
    # Verarbeitung
    # --------------------------------------------------
    df = detect_and_fix_timestamp(df)

    float_cols = [
        "open", "high", "low", "close",
        "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote"
    ]
    df[float_cols] = df[float_cols].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def download_all():
    for symbol in SYMBOLS:
        print("\n==============================")
        print(f"⬇️  Starte Download für: {symbol}")
        print("==============================")

        out_dir = ensure_dirs(symbol)
        dfs = []

        year = START_YEAR
        month = START_MONTH

        while True:
            df_month = download_month(symbol, year, month)
            if df_month is not None:
                dfs.append(df_month)

            if year == END_YEAR and month == END_MONTH:
                break

            month += 1
            if month > 12:
                month = 1
                year += 1

        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            save_path = out_dir / f"{symbol}_1s_{START_YEAR}{START_MONTH:02d}_{END_YEAR}{END_MONTH:02d}.parquet"
            final_df.to_parquet(save_path, index=False)

            print(f"✅ Gespeichert: {save_path}")
            print(f"✔ Datensätze: {final_df.shape}")
        else:
            print(f"⚠️ Keine Daten für {symbol}")


# ================================
# START
# ================================
if __name__ == "__main__":
    download_all()
