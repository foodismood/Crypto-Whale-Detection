import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------
# Paths zu deinen Feature-Dateien
# --------------------------------------------
BTC_PATH = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/processed/BTCUSDT_features_1s.parquet"
ETH_PATH = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1/data/processed/ETHUSDT_features_1s.parquet"


# --------------------------------------------
# Funktion zum Laden + Korrelation plotten
# --------------------------------------------
def plot_correlations(path, title):
    print(f"\n📊 Loading: {path}")

    df = pd.read_parquet(path)

    # nur numerische Features
    df_num = df.select_dtypes(include=["float64", "int64"])
    df_num = df_num.drop(columns=["target_3class"], errors="ignore")

    # Korrelation berechnen
    corr = df_num.corr()

    # Plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=False)
    plt.title(f"Feature-Korrelationen – {title}", fontsize=16)
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# BTC + ETH Korrelationen anzeigen
# --------------------------------------------
plot_correlations(BTC_PATH, "BTCUSDT")
plot_correlations(ETH_PATH, "ETHUSDT")
