import pandas as pd
import matplotlib.pyplot as plt

def plot_intraday_context(
        df: pd.DataFrame,
        index: int,
        window_before: int = 200,
        window_after: int = 200,
        symbol: str = "BTCUSDT",
):


    #Normalize expected colums

    if"timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column.")

    if"open" not in df.columns:
        raise ValueError("DataFrame must have 'open' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["open"] = pd.to_numeric(df["open"], errors="coerce")


    #Target timestamp and window

    target_time = df.loc[index]["timestamp"]

    start = max(0, index - window_before)
    end = min(len(df), index + window_after)

    window_df = df.iloc[start:end +1].copy().reset_index(drop=True)

    #Plot

    plt.figure(figsize=(18, 9))
    plt.plot(window_df.index, window_df["open"], label=f"{symbol} Open", color="blue", linewidth=1)

    #highlight marked targets

    target_local_idx = (window_df["timestamp"] - target_time).abs().idxmin()
    plt.axvline(target_local_idx, color="red", linestyle ="--", linewidth=2, label = "Target")

    #highlitght new Days

    day_change = window_df["timestamp"].dt.date.ne(window_df["timestamp"].dt.date.shift())
    for i, is_new_day in enumerate(day_change):
        if is_new_day and i != 0:
            plt.axvline(i, color="green", linestyle="--", alpha=0.6)

    # X-axis
    tick_positions = window_df.index[::max(1, len(window_df)//10)]
    tick_labels = window_df.loc[tick_positions,"timestamp"].dt.strftime('%Y-%m-%d %H:%M')
    plt.xticks(tick_positions, tick_labels, rotation = 45, ha="right")

    plt.title(f"{symbol}  - Open Price ±{window_before} Rows Around {target_time}")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    #Usage

    if __name__ == "__main__":

        df = df.pd.read_parquet()


        plot_intraday_context(
            df=df,
            index=2000,
        window_before=300,
        window_after=300,
        symbol="BTCUSDT"
        )