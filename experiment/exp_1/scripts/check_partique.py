import pandas as pd
import matplotlib.pyplot as plt
import os

# Absoluter Pfad zu deinem Projekt
BASE_DIR = r"C:/Users/suadn/Crypto-Whale-Detection"

# Pfad zur gewünschten Parquet-Datei
file = os.path.join(
    BASE_DIR,
    "experiment",
    "exp_1",
    "data",
    "Raw",
    "Bars_1m",
    "BTCUSDT",
    "2025-06-19.parquet"
)

df = pd.read_parquet(file)

# Nur die ersten ~50 Zeilen als Vorschau
head_df = df.head(50)

plt.figure(figsize=(12, 6))
plt.table(cellText=head_df.values,
          colLabels=head_df.columns,
          loc="center")

plt.axis("off")

out_file = "parquet_preview.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")

print("Bild gespeichert als:", out_file)
