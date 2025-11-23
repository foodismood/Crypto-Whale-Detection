import pandas as pd
import matplotlib.pyplot as plt
import os

# Absoluter Pfad zu deinem Projekt (bleibt so!)
BASE_DIR = r"C:/Users/sgenk/PycharmProjects/Crypto-Whale-Detection/"

# Pfad zur Parquet-Datei
file = os.path.join(
    BASE_DIR,
    "experiment",
    "exp_1",
    "data",
    "raw",
    "bars_1m",
    "BTCUSDT",
    "2025-06-19.parquet"
)

df = pd.read_parquet(file)

# Falls Datei noch 'open_time' statt 'timestamp' hat → umbenennen
if "open_time" in df.columns:
    df = df.rename(columns={"open_time": "timestamp"})

# Falls timestamp nicht existiert → Fehler werfen
if "timestamp" not in df.columns:
    raise ValueError(" 'timestamp' wurde in der Parquet-Datei nicht gefunden!")

# Nur die gewünschten Spalten extrahieren
columns_we_need = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap"
]

missing_cols = [c for c in columns_we_need if c not in df.columns]
if missing_cols:
    raise ValueError(f"folgende Spalten fehlen in deiner Parquet-Datei: {missing_cols}")

df = df[columns_we_need]

# ---------------------------------------------
# PNG erzeugen
# ---------------------------------------------
preview = df.head(50)

plt.figure(figsize=(14, 6))
plt.table(
    cellText=preview.values,
    colLabels=preview.columns,
    loc="center"
)
plt.axis("off")

out_file = os.path.join(BASE_DIR, "parquet_preview.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")

print("✅ PNG gespeichert unter:", out_file)
