import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"
PROCESSED_DIR = f"{BASE}/data/processed"
MODEL_DIR = f"{BASE}/deployment_backtest"
OUT_DIR = f"{BASE}/deployment_paper_trading"
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOL = "BTCUSDT"

FEATURES = [
    "return_1s",
    "return_5s",
    "return_30s",
    "volatility_30s",
    "volume_z",
    "spread",
    "whale_value",
    "whale_qty",
    "whale_side",
    "whale_buy_value",
    "whale_sell_value",
    "whale_pressure",
    "seconds_after",
]

HORIZON_SECONDS = 30
P_ENTER = 0.55
MARGIN = 0.10
FEE_BPS_PER_SIDE = 4.0
SLIPPAGE_BPS_PER_SIDE = 2.0
NOTIONAL_PER_TRADE = 1000.0


def bps_to_frac(bps: float) -> float:
    return bps / 10000.0


class PaperAccount:
    def __init__(self, starting_equity=10000.0):
        self.equity = float(starting_equity)
        self.trades = []

    def execute_trade(self, symbol, entry_time, exit_time, entry_px, exit_px, side, probs, whale_value):
        gross_ret = side * (exit_px - entry_px) / entry_px
        total_cost = 2.0 * (bps_to_frac(FEE_BPS_PER_SIDE) + bps_to_frac(SLIPPAGE_BPS_PER_SIDE))
        net_ret = gross_ret - total_cost
        pnl = NOTIONAL_PER_TRADE * net_ret
        self.equity *= (1.0 + net_ret)

        self.trades.append({
            "symbol": symbol,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_px": float(entry_px),
            "exit_px": float(exit_px),
            "side": int(side),
            "p_down": float(probs[0]),
            "p_neutral": float(probs[1]),
            "p_up": float(probs[2]),
            "whale_value": float(whale_value),
            "net_return": float(net_ret),
            "pnl": float(pnl),
            "equity": float(self.equity),
        })


def decide_trade(probs: np.ndarray) -> int:
    p_down, p_neutral, p_up = probs[0], probs[1], probs[2]
    if p_up >= P_ENTER and (p_up - p_down) >= MARGIN:
        return 1
    if p_down >= P_ENTER and (p_down - p_up) >= MARGIN:
        return -1
    return 0


def load_feature_df(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(f"{PROCESSED_DIR}/{symbol}_features_1s.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["whale_timestamp"] = pd.to_datetime(df["whale_timestamp"])
    df = df.sort_values(["whale_timestamp", "timestamp"]).reset_index(drop=True)
    return df


def get_next_event(df: pd.DataFrame, event_times: list, idx: int):
    if idx >= len(event_times):
        return None, idx
    evt = event_times[idx]
    g = df[df["whale_timestamp"] == evt].sort_values("timestamp")
    return g, idx + 1


def find_exit_row(event_df: pd.DataFrame):
    entry = event_df.iloc[0]
    t0 = entry["timestamp"]
    target = t0 + pd.Timedelta(seconds=HORIZON_SECONDS)
    tmp = event_df.copy()
    tmp["abs_dt"] = (tmp["timestamp"] - target).abs()
    exit_row = tmp.sort_values("abs_dt").iloc[0]
    return entry, exit_row


def main():
    model_path = os.path.join(MODEL_DIR, f"{SYMBOL}_xgb.json")
    model = XGBClassifier()
    model.load_model(model_path)

    df = load_feature_df(SYMBOL)
    event_times = df["whale_timestamp"].drop_duplicates().sort_values().tolist()
    account = PaperAccount(starting_equity=10000.0)

    idx = 0
    while True:
        event_df, idx = get_next_event(df, event_times, idx)
        if event_df is None:
            break

        entry_row, exit_row = find_exit_row(event_df)
        X_row = entry_row[FEATURES].values.reshape(1, -1).astype(np.float32)
        probs = model.predict_proba(X_row)[0]
        side = decide_trade(probs)

        if side == 0:
            continue

        account.execute_trade(
            symbol=SYMBOL,
            entry_time=entry_row["timestamp"],
            exit_time=exit_row["timestamp"],
            entry_px=float(entry_row["close"]),
            exit_px=float(exit_row["close"]),
            side=side,
            probs=probs,
            whale_value=float(entry_row.get("whale_value", np.nan)),
        )

    trades = pd.DataFrame(account.trades)
    out_path = os.path.join(OUT_DIR, f"{SYMBOL}_paper_trades.csv")
    trades.to_csv(out_path, index=False)

    print("Paper trading finished.")
    print("Trades:", len(trades))
    print("Final equity:", account.equity)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()