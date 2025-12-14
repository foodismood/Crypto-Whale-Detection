import pandas as pd
import numpy as np


def detect_volume_events_per_day(
    path: str,
    resample_sec: int = 5,
    rolling_seconds: int = 120,
    z_thresh: float = 3.5,
    events_per_day: int = 3,
):
    """
    Detectet Volume-Spikes:
    - ausreichend viele Kandidaten
    - aber MAXIMAL `events_per_day` pro Kalendertag
    -> ideal für ML (XGBoost)
    """

    # --------------------------------------------------
    # Daten laden (RAM-schonend)
    # --------------------------------------------------
    df = pd.read_parquet(
        path,
        columns=["timestamp", "quote_volume", "taker_buy_quote"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    print(f"Bars geladen: {len(df):,}")

    # --------------------------------------------------
    # Resample (5s)
    # --------------------------------------------------
    df = df.resample(f"{resample_sec}s").sum()

    # --------------------------------------------------
    # Volumen
    # --------------------------------------------------
    df["volume_usd"] = df["quote_volume"]
    df["buy_volume"] = df["taker_buy_quote"]
    df["sell_volume"] = df["quote_volume"] - df["taker_buy_quote"]

    # --------------------------------------------------
    # Rolling Z-Score
    # --------------------------------------------------
    win = max(20, int(rolling_seconds / resample_sec))
    roll = df["volume_usd"].rolling(win, min_periods=win)

    df["volume_z"] = (df["volume_usd"] - roll.mean()) / roll.std()

    print("Max volume_z:", round(df["volume_z"].max(), 2))

    # --------------------------------------------------
    # Kandidaten (moderat, nicht zu streng!)
    # --------------------------------------------------
    candidates = df[df["volume_z"] >= z_thresh].copy()
    print(f"Kandidaten gesamt: {len(candidates):,}")

    if candidates.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # Peak-Filter (lokales Maximum)
    # --------------------------------------------------
    vol = df["volume_usd"]
    local_peak = vol == vol.rolling(3, center=True).max()
    candidates = candidates[local_peak.reindex(candidates.index).fillna(False)]

    print(f"Kandidaten nach Peak-Filter: {len(candidates):,}")

    if candidates.empty:
        return pd.DataFrame()

    # --------------------------------------------------
    # PRO TAG: Top-N Events
    # --------------------------------------------------
    candidates = candidates.reset_index()
    candidates["day"] = candidates["timestamp"].dt.date

    events = (
        candidates
        .sort_values("volume_z", ascending=False)
        .groupby("day")
        .head(events_per_day)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print(f"Finale Events gesamt: {len(events):,}")

    return events[[
        "timestamp",
        "volume_z",
        "volume_usd",
        "buy_volume",
        "sell_volume",
    ]].rename(columns={
        "timestamp": "volume_event_timestamp"
    })


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    PATH = (
        "/Users/alperademgencer/PycharmProjects/"
        "Crypto-Whale-Detection/1/experiment/exp_1/"
        "data/raw/Bars_1s/BTCUSDT/"
        "BTCUSDT_1s_202406_202506.parquet"
    )

    print("\n--- Volume Event Detection (per Day) gestartet ---\n")

    events = detect_volume_events_per_day(
        path=PATH,
        resample_sec=5,
        rolling_seconds=120,
        z_thresh=3.5,
        events_per_day=3,   # <<< 2–5 HIER EINSTELLEN
    )

    print("\n--- Ergebnis ---")
    print(f"Anzahl Volume-Events gesamt: {len(events)}")

    if not events.empty:
        print(events.head())

        events["day"] = events["volume_event_timestamp"].dt.date
        print("\nEvents pro Tag (Summary):")
        print(events.groupby("day").size().describe())

        out_path = "BTCUSDT_volume_events.parquet"
        events.drop(columns=["day"]).to_parquet(out_path)
        print(f"\nEvents gespeichert unter: {out_path}")
    else:
        print("⚠️ Keine Events gefunden – Parameter anpassen.")

    print("\n--- Fertig ---")
