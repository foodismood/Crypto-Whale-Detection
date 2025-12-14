import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


# Pfade – identisch zu deinem Projekt

BASE_DIR = Path(r"C:/Users/sgenk/PycharmProjects/Crypto-Whale-Detection")
DATA_DIR = BASE_DIR / "experiment" / "exp_1" / "data" / "raw" / "orderbook"
IMG_DIR = BASE_DIR / "experiment" / "exp_1" / "images"

IMG_DIR.mkdir(parents=True, exist_ok=True)


# Datei auswählen (jüngste Whale-Datei)
symbol = "BTCUSDT"

symbol_folder = DATA_DIR / symbol
parquet_files = sorted(symbol_folder.glob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"Keine Parquet-Dateien gefunden in: {symbol_folder}")

# Neueste Datei nehmen
latest_file = parquet_files[-1]
print(f"Lade Datei:\n{latest_file}\n")

df = pd.read_parquet(latest_file)


# Nur relevante Orderbook-Whale-Spalten

columns_we_need = [
    "timestamp",
    "symbol",
    "side",
    "price",
    "quantity",
    "value_usd"
]

missing = [c for c in columns_we_need if c not in df.columns]
if missing:
    raise ValueError(f"Folgende Spalten fehlen: {missing}")

df = df[columns_we_need]

# ---------------------------------------------
# PNG erstellen
# ---------------------------------------------
preview = df.head(40)  # 40 Zeilen reichen für eine PNG

plt.figure(figsize=(14, 8))
plt.table(
    cellText=preview.values,
    colLabels=preview.columns,
    loc="center"
)
plt.axis("off")

out_path = IMG_DIR / f"orderbook_preview_{symbol}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"PNG gespeichert unter:\n{out_path}")
