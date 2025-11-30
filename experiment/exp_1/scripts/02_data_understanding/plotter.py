import pandas as pd
import matplotlib.pyplot as plt


def plot_after_whale(
        df: pd.DataFrame,
        whale_timestamp: str,
        minutes_after: int = 300,
        symbol: str = "BTCUSDT",
        save_path: str = None
):

    # Normalize
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")

    whale_ts = pd.to_datetime(whale_timestamp)

    # Select bars AFTER the whale event
    window_df = df[df["timestamp"] >= whale_ts].iloc[:minutes_after].copy().reset_index(drop=True)

    if len(window_df) == 0:
        raise ValueError("No bars found after whale timestamp!")

    # --- Plot ---
    plt.figure(figsize=(18, 9))
    plt.plot(window_df.index, window_df["open"], label=f"{symbol} Open Price", color="blue", linewidth=1.2)

    # Mark whale event (always x=0)
    plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Whale Event")

    # Mark new days
    day_change = window_df["timestamp"].dt.date.ne(window_df["timestamp"].dt.date.shift())

    for i, is_new_day in enumerate(day_change):
        if is_new_day and i != 0:
            plt.axvline(i, color="green", linestyle="--", alpha=0.6)

    # X-axis timestamps
    tick_positions = window_df.index[::max(1, len(window_df)//10)]
    tick_labels = window_df.loc[tick_positions, "timestamp"].dt.strftime('%Y-%m-%d %H:%M')

    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    plt.title(f"{symbol} - Price AFTER Whale Event at {whale_ts}")
    plt.xlabel("Minutes After Whale Event")
    plt.ylabel("Open Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()



# USAGE EXAMPLE

if __name__ == "__main__":

    # Load your 1-minute Parquet bars
    df = pd.read_parquet("data/raw/btc_1m.parquet")

    # Example whale event timestamp (replace with real one)
    whale_time = "2025-01-12 14:33:00"

    plot_after_whale(
        df=df,
        whale_timestamp=whale_time,
        minutes_after=300,
        symbol="BTCUSDT",
        save_path="plots/whale_event_2025-01-12.png"
    )
