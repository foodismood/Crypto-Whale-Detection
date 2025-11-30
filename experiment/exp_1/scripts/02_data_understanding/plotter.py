import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# SETTINGS – Pfade anpassen
# ==============================================================================

BASE_DIR = Path("/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1")
IMG_DIR = BASE_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

BARS_PATH = {
    "BTCUSDT": BASE_DIR / "data/raw/Bars_1m/BTCUSDT/2023-06-20_to_2025-06-20.parquet",
    "ETHUSDT": BASE_DIR / "data/raw/Bars_1m/ETHUSDT/2023-06-20_to_2025-06-20.parquet",
}

WHALES_PATH = {
    "BTCUSDT": BASE_DIR / "data/raw/Orderbook/BTCUSDT/BTCUSDT_20240620-20250620_whales.parquet",
    "ETHUSDT": BASE_DIR / "data/raw/Orderbook/ETHUSDT/ETHUSDT_20240620-20250620_whales.parquet",
}


# ==============================================================================
# TIMESTAMP FUNKTIONEN
# ==============================================================================

def normalize_whale_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whales: timestamp in Millisekunden (13-stellig) -> Datetime in Spalte 'whale_ts'.
    Originalspalte 'timestamp' bleibt unangetastet.
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"No 'timestamp' column in whales dataframe. Columns: {df.columns}")

    ts_raw = df["timestamp"].astype("int64")
    df["whale_ts"] = pd.to_datetime(ts_raw, unit="ms", utc=True)
    return df


def normalize_bars_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bars: 'timestamp' kann Sekunden (10-stellig) oder Millisekunden (13-stellig) sein.
    Ergebnis steht in 'bar_ts'.
    """
    if "timestamp" not in df.columns:
        raise ValueError(f"No 'timestamp' column in bars dataframe. Columns: {df.columns}")

    ts_raw = df["timestamp"].astype("int64")
    digits = ts_raw.astype(str).str.len().max()

    if digits == 10:
        unit = "s"
    elif digits == 13:
        unit = "ms"
    elif digits == 16:
        unit = "us"
    elif digits == 19:
        unit = "ns"
    else:
        raise ValueError(f"Unsupported timestamp length for bars: {digits} digits")

    df["bar_ts"] = pd.to_datetime(ts_raw, unit=unit, utc=True)

    # zur Sicherheit nach Zeit sortieren
    df = df.sort_values("bar_ts").reset_index(drop=True)
    return df


# ==============================================================================
# PLOT FUNKTION
# ==============================================================================

def plot_after_whale_event(
    df_bars: pd.DataFrame,
    whale_timestamp: pd.Timestamp,
    symbol: str,
    save_path: Path,
    minutes_after: int = 30,
) -> None:
    """
    Plottet die 1m-Bars für 'minutes_after' Minuten NACH dem Whale-Event.
    Start = Zeitpunkt des Whale-Events.
    """

    # Zeitfenster definieren
    end_time = whale_timestamp + pd.Timedelta(minutes=minutes_after)

    mask = (df_bars["bar_ts"] >= whale_timestamp) & (df_bars["bar_ts"] <= end_time)
    subset = df_bars.loc[mask].copy()

    if subset.empty:
        print(f"No candle data available for window after {whale_timestamp} ({symbol})")
        return

    # X-Achse = Minuten nach Whale
    subset = subset.sort_values("bar_ts").reset_index(drop=True)
    subset["minutes_after"] = (subset["bar_ts"] - whale_timestamp).dt.total_seconds() / 60.0

    plt.figure(figsize=(18, 9))
    plt.plot(subset["minutes_after"], subset["close"], label=f"{symbol} Close Price")

    # Vertikale Linie bei Minute 0 (Whale-Event)
    plt.axvline(0, linestyle="--", linewidth=2, label="Whale Event")

    # Day-Change Marker (optional)
    day_change = subset["bar_ts"].dt.date.ne(subset["bar_ts"].dt.date.shift())
    for i, new_day in enumerate(day_change):
        if i > 0 and new_day:
            x = subset.loc[i, "minutes_after"]
            plt.axvline(x, linestyle="--", alpha=0.6)

    plt.title(f"{symbol} — {minutes_after} Minutes After Whale Event ({whale_timestamp})")
    plt.xlabel("Minutes After Whale")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

if __name__ == "__main__":

    for symbol in ["BTCUSDT", "ETHUSDT"]:

        print("\n=======================================")
        print(f"PROCESSING WHALE EVENTS FOR {symbol}")
        print("=======================================\n")

        # Daten laden
        df_whales = pd.read_parquet(WHALES_PATH[symbol])
        df_bars = pd.read_parquet(BARS_PATH[symbol])

        # Timestamps normalisieren (ohne irgendwas zu überschreiben)
        df_whales = normalize_whale_timestamp(df_whales)
        df_bars = normalize_bars_timestamp(df_bars)

        # OPTIONAL: nur Whales über bestimmtem Wert filtern
        # Beispiel, falls du eine USD-Notional-Spalte hast:
        # if "usd_value" in df_whales.columns:
        #     df_whales = df_whales[df_whales["usd_value"] >= 1_000_000]

        if df_whales.empty:
            print(f"No whale events found for {symbol}")
            continue

        # Jede Whale-Order einzeln plottet die nächsten 30 Minuten
        for idx, whale in df_whales.iterrows():
            whale_ts = whale["whale_ts"]  # korrekt konvertierte Datetime

            save_name = (
                f"{symbol}_whale_{idx}_"
                f"{whale_ts.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            )
            save_path = IMG_DIR / save_name

            print(f"Plotting whale event #{idx} at {whale_ts}")

            plot_after_whale_event(
                df_bars=df_bars,
                whale_timestamp=whale_ts,
                symbol=symbol,
                save_path=save_path,
                minutes_after=30,
            )
