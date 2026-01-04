import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier

# ======================================================
# PARAMETER
# ======================================================
HORIZON = 30                  # exit after N seconds
WINDOW_BEFORE = 60            # only for sanity checks / context
TRAIN_FRACTION = 0.8          # same split as training

# Trading mode:
# - ALWAYS_ACT=True => always BUY or SELL (prof requirement)
# - ALWAYS_ACT=False => can skip trades using thresholds below
ALWAYS_ACT = True

# Used only if ALWAYS_ACT=False
P_BUY_ENTER = 0.70            # enter BUY only if p_buy >= this
P_SELL_ENTER = 0.70           # enter SELL only if p_sell >= this

# ======================================================
# FEATURES (must match training)
# ======================================================
FEATURES = [
    "whale_value_usd",
    "whale_qty",
    "whale_side",
    "volume_sum_pre_30s",
    "volume_mean_pre_30s",
    "volume_std_pre_30s",
    "volume_sum_post_10s",
    "volume_sum_post_30s",
    "volume_peak_post_30s",
    "time_to_volume_peak",
    "volume_reaction_ratio",
]

TIME_COL = "event_timestamp"
RETURN_COL = "return_30s"

# ======================================================
# PATHS
# ======================================================
BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"

DATASET_PATH = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume/"
    "BTCUSDT_whale_volume_dataset.parquet"
)

MODEL_PATH = f"{BASE}/model/BTCUSDT_xgb_whale_volume_binary_always.json"

BARS_PATH = (
    f"{BASE}/data/raw/Bars_1s/BTCUSDT/"
    "BTCUSDT_1s_202406_202506.parquet"
)

OUT_DIR = (
    f"{BASE}/scripts/03_data_modeling/whaleANDvolume/"
    "whaleAndvolume_plots"
)
os.makedirs(OUT_DIR, exist_ok=True)

TRADES_CSV = os.path.join(OUT_DIR, "backtest_trades.csv")
EQUITY_PNG = os.path.join(OUT_DIR, "equity_curve.png")
SUMMARY_TXT = os.path.join(OUT_DIR, "backtest_summary.txt")


# ======================================================
# MODEL LOADING (robust) -> Booster predict
# ======================================================
def load_booster(model_path: str):
    m = XGBClassifier()
    m.load_model(model_path)
    return m.get_booster()


def predict_p_buy(booster, X_row: np.ndarray) -> float:
    dmat = xgb.DMatrix(X_row, feature_names=FEATURES)
    # binary:logistic -> returns p_buy directly
    return float(booster.predict(dmat)[0])


# ======================================================
# METRICS
# ======================================================
def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def sharpe_per_trade(returns: pd.Series) -> float:
    # simple per-trade sharpe (not annualized)
    mu = returns.mean()
    sd = returns.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mu / sd)


# ======================================================
# MAIN
# ======================================================
def main():
    # -----------------------------
    # LOAD DATASET + SPLIT
    # -----------------------------
    df = pd.read_parquet(DATASET_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # drop NaNs needed for inference
    df = df.dropna(subset=FEATURES + [TIME_COL]).copy()

    split_idx = int(len(df) * TRAIN_FRACTION)
    val_df = df.iloc[split_idx:].copy()  # out-of-sample backtest
    print("Total events:", len(df))
    print("Backtest (val) events:", len(val_df))

    # -----------------------------
    # LOAD BARS
    # -----------------------------
    bars = pd.read_parquet(BARS_PATH, columns=["timestamp", "close", "volume"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.sort_values("timestamp").set_index("timestamp")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    booster = load_booster(MODEL_PATH)

    # -----------------------------
    # RUN EVENT-BASED BACKTEST
    # -----------------------------
    trades = []

    for _, row in val_df.iterrows():
        t0 = row[TIME_COL]
        bar_idx = bars.index.searchsorted(t0)

        # need enough bars for entry/exit
        if bar_idx <= WINDOW_BEFORE:
            continue
        if bar_idx + HORIZON >= len(bars):
            continue

        X = row[FEATURES].values.astype(np.float32).reshape(1, -1)
        p_buy = predict_p_buy(booster, X)
        p_sell = 1.0 - p_buy

        # Decide action
        if ALWAYS_ACT:
            action = "BUY" if p_buy >= 0.5 else "SELL"
        else:
            if p_buy >= P_BUY_ENTER:
                action = "BUY"
            elif p_sell >= P_SELL_ENTER:
                action = "SELL"
            else:
                action = "NO_TRADE"

        if action == "NO_TRADE":
            continue

        entry_time = bars.index[bar_idx]
        exit_time = bars.index[bar_idx + HORIZON]
        entry_px = float(bars.iloc[bar_idx]["close"])
        exit_px = float(bars.iloc[bar_idx + HORIZON]["close"])

        # Return in direction
        if action == "BUY":
            ret = (exit_px - entry_px) / entry_px
        else:  # SELL = short
            ret = (entry_px - exit_px) / entry_px

        trades.append({
            "event_time": t0,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "action": action,
            "p_buy": p_buy,
            "p_sell": p_sell,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "return": ret,
            # keep some context (optional)
            "whale_value_usd": float(row["whale_value_usd"]),
            "whale_side": float(row["whale_side"]),
            "volume_reaction_ratio": float(row["volume_reaction_ratio"]),
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise RuntimeError("No trades generated. Check data coverage / thresholds.")

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)
    trades_df["equity"] = (1.0 + trades_df["return"]).cumprod()

    # -----------------------------
    # METRICS
    # -----------------------------
    n_trades = len(trades_df)
    winrate = float((trades_df["return"] > 0).mean())
    avg_ret = float(trades_df["return"].mean())
    med_ret = float(trades_df["return"].median())
    std_ret = float(trades_df["return"].std(ddof=0))
    mdd = max_drawdown(trades_df["equity"])
    shp = sharpe_per_trade(trades_df["return"])

    # -----------------------------
    # SAVE TRADES
    # -----------------------------
    trades_df.to_csv(TRADES_CSV, index=False)
    print("Saved trades:", TRADES_CSV)

    # -----------------------------
    # EQUITY CURVE PLOT
    # -----------------------------
    plt.figure()
    plt.plot(trades_df["entry_time"], trades_df["equity"])
    plt.title("Backtest Equity Curve (event-based)")
    plt.xlabel("Time")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EQUITY_PNG, dpi=150)
    plt.close()
    print("Saved equity curve:", EQUITY_PNG)

    # -----------------------------
    # SUMMARY TXT
    # -----------------------------
    summary = (
        f"BACKTEST SUMMARY (Out-of-sample)\n"
        f"ALWAYS_ACT={ALWAYS_ACT}\n"
        f"HORIZON={HORIZON}s\n"
        f"Events in val window: {len(val_df)}\n"
        f"Trades executed: {n_trades}\n"
        f"Winrate: {winrate:.3f}\n"
        f"Avg return per trade: {avg_ret:.6f}\n"
        f"Median return per trade: {med_ret:.6f}\n"
        f"Std return per trade: {std_ret:.6f}\n"
        f"Sharpe (per-trade): {shp:.3f}\n"
        f"Max drawdown: {mdd:.3f}\n"
        f"Final equity: {float(trades_df['equity'].iloc[-1]):.4f}\n"
    )

    with open(SUMMARY_TXT, "w") as f:
        f.write(summary)

    print("\n" + summary)
    print("Saved summary:", SUMMARY_TXT)


if __name__ == "__main__":
    main()
