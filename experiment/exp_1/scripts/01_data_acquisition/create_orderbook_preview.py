import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Pfade – angepasst für dein Projekt

BASE_DIR = Path(os.path.expanduser(
    "~/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/raw/Orderbook"
))
IMG_DIR = BASE_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Symbol / Monat auswählen
month_folder = "BTCUSDT"
data_folder = BASE_DIR / month_folder

# Neueste Parquet-Datei auswählen

parquet_files = sorted(data_folder.glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"Keine Parquet-Dateien gefunden in: {data_folder}")

latest_file = parquet_files[-1]
print(f"Lade Datei:\n{latest_file}\n")

# Parquet laden
df = pd.read_parquet(latest_file)

# Spalten umbenennen / anpassen, falls nötig
# Mapping Binance → erwartet
column_map = {}
if "time" in df.columns:
    column_map["time"] = "timestamp"
if "qty" in df.columns:
    column_map["qty"] = "quantity"
if "quote_qty" in df.columns:
    column_map["quote_qty"] = "value_usd"
if "is_buyer_maker" in df.columns:
    column_map["is_buyer_maker"] = "side"

df = df.rename(columns=column_map)

# Symbol hinzufügen, falls fehlt
if "symbol" not in df.columns:
    df["symbol"] = "BTCUSDT"

# Side umwandeln: True/False → buy/sell
if "side" in df.columns:
    df["side"] = df["side"].apply(lambda x: "buy" if x else "sell")
else:
    df["side"] = "unknown"

# Nur die relevanten Spalten auswählen, falls sie existieren
columns_we_need = ["timestamp", "symbol", "side", "price", "quantity", "value_usd"]
existing_columns = [c for c in columns_we_need if c in df.columns]
df = df[existing_columns]

# PNG erstellen

preview = df.head(40)  # nur die ersten 40 Zeilen

plt.figure(figsize=(14, 8))
plt.table(
    cellText=preview.values,
    colLabels=preview.columns,
    loc="center",
    cellLoc="center"
)
plt.axis("off")

out_path = IMG_DIR / f"orderbook_preview_{month_folder}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"PNG gespeichert unter:\n{out_path}")
