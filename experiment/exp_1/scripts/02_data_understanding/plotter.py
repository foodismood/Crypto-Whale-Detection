import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



# Hilfsfunktion: Bars vorbereiten
def prepare_bars(bars_parquet: str) -> pd.DataFrame:
    df_bars = pd.read_parquet(bars_parquet).copy()

    # Sicherstellen, dass timestamp eine echte Datetime-Spalte ist
    df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"], errors="coerce")
    df_bars = df_bars.dropna(subset=["timestamp"])
    df_bars = df_bars.sort_values("timestamp").reset_index(drop=True)

    return df_bars


# Hilfsfunktion: Whales vorbereiten

def prepare_whales(whale_parquet: str) -> pd.DataFrame:
    df_whales = pd.read_parquet(whale_parquet).copy()

    # time numerisch erzwingen
    df_whales["time"] = pd.to_numeric(df_whales["time"], errors="coerce")

    before = len(df_whales)

    # Binance-ms-Timestamps:
    mask = (df_whales["time"] > 1_000_000_000_000) & (df_whales["time"] < 10_000_000_000_000)
    df_whales = df_whales.loc[mask].copy()

    after = len(df_whales)
    print(f" Whale cleanup: {before} → {after} Zeilen (entfernt: {before - after})")

    # Jetzt sicher in Datetime (Unix ms)
    df_whales["timestamp"] = pd.to_datetime(df_whales["time"], unit="ms", errors="coerce")
    df_whales = df_whales.dropna(subset=["timestamp"])
    df_whales = df_whales.sort_values("timestamp").reset_index(drop=True)

    return df_whales


#
# Plot price 30 minutes after whale trade
#
def plot_after_whale(df_bars: pd.DataFrame,
                     whale: pd.Series,
                     symbol: str,
                     save_path: Path,
                     minutes_after: int = 30) -> None:
    whale_ts = whale["timestamp"]

    # Alle Bars, die ab dem Whale-Zeitpunkt liegen
    matches = df_bars.index[df_bars["timestamp"] >= whale_ts]

    if len(matches) == 0:
        print(f"Keine Bars nach Whale gefunden bei {whale_ts}")
        return

    start_idx = matches.min()
    end_idx = start_idx + minutes_after

    subset = df_bars.loc[start_idx:end_idx].copy().reset_index(drop=True)

    if subset.empty:
        print(f"Leeres Zeitfenster nach Whale {whale_ts}")
        return

    # Plot
    plt.figure(figsize=(18, 9))
    plt.plot(subset.index, subset["close"],
             label=f"{symbol} Close Price",
             linewidth=1.5)

    # Event bei t=0 markieren
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Whale Event")

    # Neue Tage markieren (vertikale Linien in grün)
    day_change = subset["timestamp"].dt.date.ne(subset["timestamp"].dt.date.shift())
    for i, is_new in enumerate(day_change):
        if i > 0 and is_new:
            plt.axvline(i, color="green", linestyle="--", alpha=0.6)

    # X-Achse: ein paar Ticks
    ticks = subset.index[::max(1, len(subset) // 10)]
    tick_labels = subset.loc[ticks, "timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    plt.xticks(ticks, tick_labels, rotation=45, ha="right")


    qty = whale["qty"]
    price = whale["price"]
    usd = whale["value_usd"]
    side = "SELL" if whale["isBuyerMaker"] else "BUY"  # Binance-Logik: isBuyerMaker=True → Maker ist Käufer → Aggressor SELL

    info_text = (
        f"Whale: {qty:.2f} BTC @ ${price:,.2f}\n"
        f"Value: ${usd:,.0f}\n"
        f"Side: {side}"
    )

    plt.text(
        0.02, 0.98, info_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black")
    )


    plt.title(f"{symbol} – {minutes_after} Minuten nach Whale ({whale_ts})")
    plt.xlabel("Minuten nach Event")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved plot: {save_path}")



#Whale- und Bar-Dateien vergleichen + Plots erzeugen

def compare_parquets(whale_parquet: str, bars_parquet: str, symbol: str) -> None:
    print(f"\n==================== {symbol} ====================")
    print(f"Lade Whales: {whale_parquet}")
    print(f"Lade Bars:   {bars_parquet}")

    df_whales = prepare_whales(whale_parquet)
    df_bars = prepare_bars(bars_parquet)

    print(f"Whale-Zeitraum: {df_whales['timestamp'].min()}  →  {df_whales['timestamp'].max()}")
    print(f"Bars-Zeitraum:  {df_bars['timestamp'].min()}  →  {df_bars['timestamp'].max()}")

    output_dir = Path("plots") / symbol

    plotted = 0

    for idx, whale in df_whales.iterrows():
        ts = whale["timestamp"]

        # Sicherheit: außerhalb des Bar-Zeitraums überspringen
        if ts < df_bars["timestamp"].min():
            print(f"Whale {idx} zu früh ({ts}), liegt vor Bars-Start.")
            continue
        if ts > df_bars["timestamp"].max():
            print(f"Whale {idx} zu spät ({ts}), liegt nach Bars-Ende.")
            continue

        save_file = output_dir / f"{symbol}_whale_{idx}_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.png"

        plot_after_whale(
            df_bars=df_bars,
            whale=whale,
            symbol=symbol,
            save_path=save_file,
            minutes_after=30
        )
        plotted += 1

    print(f"Fertig: {plotted} Plots für {symbol} erstellt.\n")


if __name__ == "__main__":
    compare_parquets(
        whale_parquet=r"C:\Users\sgenk\PycharmProjects\Crypto-Whale-Detection\experiment\exp_1\data\raw\Orderbook\BTCUSDT\BTCUSDT_20240620-20250620_whales.parquet",
        bars_parquet=r"C:\Users\sgenk\PycharmProjects\Crypto-Whale-Detection\experiment\exp_1\data\raw\Bars_1m\BTCUSDT\2023-06-20_to_2025-06-20.parquet",
        symbol="BTCUSDT",
    )
