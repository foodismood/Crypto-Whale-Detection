import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# -------------------------------------------------------
# Pfad zur BTC 1-Sekunden-Kline Datei
# -------------------------------------------------------
PATH = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Bars_1s/BTCUSDT"

# -------------------------------------------------------
# Zielordner für Plots
# -------------------------------------------------------
OUT_DIR = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/scripts/02_data_understanding"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Daten laden
# -------------------------------------------------------
df = pd.read_parquet(PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -------------------------------------------------------
# Buy/Sell Volume berechnen
# -------------------------------------------------------

if "taker_buy_quote" not in df.columns:
    raise ValueError(" Die Datei enthält kein 'taker_buy_quote'. Bitte erneut mit Download-Skript laden!")

df["buy_volume"] = df["taker_buy_quote"]                     # USD der aggressiven Käufer
df["sell_volume"] = df["quote_volume"] - df["taker_buy_quote"]  # USD der aggressiven Verkäufer

# -------------------------------------------------------
# Volume-Z-Score (60s Fenster) basierend auf BUY+SELL
# -------------------------------------------------------
df["volume_usd"] = df["buy_volume"] + df["sell_volume"]

df["vol_mean"] = df["volume_usd"].rolling(60, min_periods=1).mean()
df["vol_std"] = df["volume_usd"].rolling(60, min_periods=1).std()
df["volume_z"] = (df["volume_usd"] - df["vol_mean"]) / df["vol_std"]

# Top-10 Volume Spikes
events = df.nlargest(10, "volume_z")

print("\nTop 10 USD Volume Events:")
print(events[["timestamp", "buy_volume", "sell_volume", "volume_usd", "volume_z"]])

# -------------------------------------------------------
# 5 Minuten Fenster (300s vorher + 300s nachher)
# -------------------------------------------------------
WINDOW = pd.Timedelta(seconds=300)

# -------------------------------------------------------
# Plots erzeugen
# -------------------------------------------------------
for idx, row in events.iterrows():
    t = row["timestamp"]

    df_win = df[(df["timestamp"] >= t - WINDOW) & (df["timestamp"] <= t + WINDOW)]

    # Preis
    price_at_t = df.loc[df["timestamp"] == t, "close"].iloc[0]

    # Preis 30 Sekunden später .
    try:
        price_at_t_plus_30 = df.loc[df["timestamp"] == t + pd.Timedelta(seconds=30), "close"].iloc[0]
        pct_change = (price_at_t_plus_30 - price_at_t) / price_at_t * 100
        pct_text = f"{pct_change:+.2f}%"
    except:
        pct_text = "N/A"

    # Buy & Sell Volumen des Events
    buy_v = row["buy_volume"]
    sell_v = row["sell_volume"]
    vol_text = f"Buy: {buy_v:,.0f} USD   |   Sell: {sell_v:,.0f} USD"

    plt.figure(figsize=(16, 8))

    # ----------------------- PRICE PLOT -----------------------
    plt.subplot(2, 1, 1)
    plt.plot(df_win["timestamp"], df_win["close"], linewidth=1.2)
    plt.axvline(t, color="red", linestyle="--", label="Volume Spike")
    plt.title(f"BTC – Price Around Volume Spike ({t})")
    plt.ylabel("Price (USDT)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Text: 30s-Preisänderung
    plt.text(0.02, 0.92, f"30s Change: {pct_text}", transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    plt.xticks(rotation=45)

    # ----------------------- BUY/SELL VOLUME PLOT -----------------------
    plt.subplot(2, 1, 2)

    plt.plot(df_win["timestamp"], df_win["buy_volume"], linewidth=1.2, color="green", label="Buy Volume")
    plt.plot(df_win["timestamp"], df_win["sell_volume"], linewidth=1.2, color="red", label="Sell Volume")

    plt.axvline(t, color="black", linestyle="--")
    plt.title("Buy vs Sell Volume (USD)")
    plt.ylabel("USD Volume")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Textbox mit Buy/Sell Volumen des Events
    plt.text(0.02, 0.92, vol_text, transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(mdates.SecondLocator(interval=10))
    plt.xticks(rotation=45)

    # ----------------------- SPEICHERN -----------------------
    safe_name = str(t).replace(":", "-").replace(" ", "_")
    out_path = f"{OUT_DIR}/buy_sell_volume_spike_{safe_name}.png"

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Plot gespeichert: {out_path}")
