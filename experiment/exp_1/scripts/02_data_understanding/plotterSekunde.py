import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ================================
# Bars vorbereiten (1s-Klines)
# ================================
def prepare_bars(bars_parquet: str) -> pd.DataFrame:
    df_bars = pd.read_parquet(bars_parquet).copy()

    df_bars["timestamp"] = pd.to_datetime(df_bars["timestamp"], errors="coerce")
    df_bars = df_bars.dropna(subset=["timestamp"])
    df_bars = df_bars.sort_values("timestamp").reset_index(drop=True)

    return df_bars


# ================================
# Whales vorbereiten (nur > 50 Mio USD)
# ================================
def prepare_whales(whale_parquet: str) -> pd.DataFrame:
    df_whales = pd.read_parquet(whale_parquet).copy()

    df_whales["time"] = pd.to_numeric(df_whales["time"], errors="coerce")
    df_whales["timestamp"] = pd.to_datetime(df_whales["time"], unit="ms", errors="coerce")
    df_whales = df_whales.dropna(subset=["timestamp"])

    # NEU: filter whales > 50M USD
    df_whales["value_usd"] = df_whales["qty"] * df_whales["price"]
    df_whales = df_whales[df_whales["value_usd"] >= 3_000_000]

    df_whales = df_whales.sort_values("timestamp").reset_index(drop=True)

    print(f"🐋 Whale-Filter: {len(df_whales)} Trades über 5 Mio USD gefunden")

    return df_whales


# ================================
# Plot nach Whale (5 Minuten = 300 Sekunden)
# ================================
def plot_after_whale(df_bars: pd.DataFrame,
                     whale: pd.Series,
                     symbol: str,
                     save_path: Path,
                     seconds_after: int = 300) -> None:

    whale_ts = whale["timestamp"]

    # Index finden
    matches = df_bars.index[df_bars["timestamp"] >= whale_ts]
    if len(matches) == 0:
        print(f"⚠️ Keine Bars nach Whale gefunden bei {whale_ts}")
        return

    start_idx = matches.min()
    end_idx = start_idx + seconds_after  # 1 index = 1 Sekunde

    subset = df_bars.loc[start_idx:end_idx].copy().reset_index(drop=True)

    if subset.empty:
        print(f"⚠️ Leeres Zeitfenster nach Whale {whale_ts}")
        return

    # ================================
    # Plot erstellen
    # ================================
    plt.figure(figsize=(18, 9))
    plt.plot(subset.index, subset["close"], linewidth=1.3, label=f"{symbol} Price (1s)")

    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Whale Event")

    # Whale Info Box
    qty = whale["qty"]
    price = whale["price"]
    usd = whale["value_usd"]
    side = "SELL" if whale["isBuyerMaker"] else "BUY"

    info_text = (
        f"Whale: {qty:.2f} @ ${price:,.2f}\n"
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

    plt.title(f"{symbol} – 5 Minuten (300 Sekunden) nach Whale Event\n{whale_ts}")
    plt.xlabel("Sekunden nach Event")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"📈 Saved plot: {save_path}")


# ================================
# Vergleich Whale + Bars
# ================================
def compare_parquets(whale_parquet: str, bars_parquet: str, symbol: str) -> None:
    print(f"\n==================== {symbol} ====================")
    print(f"Lade Whales: {whale_parquet}")
    print(f"Lade Bars:   {bars_parquet}")

    df_whales = prepare_whales(whale_parquet)
    df_bars = prepare_bars(bars_parquet)

    output_dir = Path("plots_seconds") / symbol

    plotted = 0

    for idx, whale in df_whales.iterrows():
        ts = whale["timestamp"]

        if ts < df_bars["timestamp"].min() or ts > df_bars["timestamp"].max():
            continue

        save_file = output_dir / f"{symbol}_whale_{idx}_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.png"

        plot_after_whale(
            df_bars=df_bars,
            whale=whale,
            symbol=symbol,
            save_path=save_file,
            seconds_after=300  # 5 Minuten
        )
        plotted += 1

    print(f"🎉 Fertig: {plotted} Plots erstellt.\n")


# ================================
# MAIN
# ================================

if __name__ == "__main__":

    compare_parquets(
        whale_parquet="/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Orderbook/BTCUSDT/BTCUSDT_20240620-20250620_whales.parquet",
        bars_parquet="/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Bars_1s/BTCUSDT/BTCUSDT_1s_20240620_20250620.parquet",
        symbol="BTCUSDT"
    )

    compare_parquets(
        whale_parquet="/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Orderbook/ETHUSDT/ETHUSDT_20240620-20250620_whales.parquet",
        bars_parquet="/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Bars_1s/ETHUSDT/ETHUSDT_1s_20240620_20250620.parquet",
        symbol="ETHUSDT"
    )
