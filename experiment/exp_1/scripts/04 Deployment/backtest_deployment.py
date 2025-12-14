import os
from lib2to3.pygram import Symbols
from string import digits
from tarfile import SYMTYPE

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.ma.extras import average
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = "C:\\Users\sgenk\PycharmProjects\Crypto-Whale-Detection\experiment\exp_1"
PROCESSED_DIR = f"{BASE}\data\Processed\\"
OUTPUT_DIR = f"{BASE}/deployment_backtest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

Symbols = ["BTCUSDT" , "ETHUSDT"]

FEATURES = ["return_1s", "return_5s", "return_30s", "volatility_30s", "volume_z", "spread", "whale_value", "whale_qty", "whale_side", "whale_buy_value", "whale_sell_value", "whale_pressure", "seconds_after"]

TARGET = "target_3class"
CLASS_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "up"}

#Backtest Horizon
HORIZON_SECONDS = 30

#Trading rules
P_ENTER = 0.55
MARGIN = 0.10

#COST model

FEE_BPS_PER_SIDE = 4.0
SLIPPAGE_BPS_PER_SIDE = 2.0

#Position sizing
NOTIONAL_PER_TRADE = 1000.0

def bps_to_frac(bps: float) -> float:
    return bps / 10_000.0

def load_features(Symbol: str) -> pd.DataFrame:
    path = f"{PROCESSED_DIR}/{Symbol}_features_1s.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["whale_timestamp"] = pd.to_datetime(df["whale_timestamp"])

    df = df.sort_values(["whale_timestap", "timestamp"]).reset_index(drop=True)
    return df

def train_model_time_split(df: pd.DataFrame):
    df_model = df.dropna(subset=FEATURES + [TARGET].copy())
    df_model = df_model.sort_values(["whale_timestap", "timestamp"]).reset_index(drop=True)

    unique_events = df.model["whale_timestap"].drop_duplicates().sort_values()
    split_evt = int(len(unique_events) * 0.8)
    train_events = set(unique_events.iloc[:split_evt])
    val_events = set(unique_events.iloc[split_evt:])

    train_df = df_model[df_model["whale_timestap"].isin(train_events)]
    val_df = df_model[df_model["whale_timestap"].isin(val_events)]

    X_train = train_df[FEATURES].astype(np.float32)
    y_train = train_df[TARGET].astype(int)

    X_val = val_df[FEATURES].astype(np.float32)
    y_val = val_df[TARGET].astype(int)

    class_conts = y_train.value_counts().sort_index()
    total = len(y_train)
    n_classes = len(class_conts)
    class_weight = {cls: total / (n_classes * cnt) for cls, cnt in class_conts.items()}
    sample_weight = y_train.map(class_weight)

    model = XGBClassifier(n_estimators = 300, learning_rate = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, objective = "multi:softprob", num_class =3, eval_metric = "mlogloss", tree_method = "hist",n_jobs = -1 )

    model.fit(X_train, y_train, sample_weight = sample_weight)
    probs_val = model.predict_proba(X_val)
    preds_val = np.argmax(probs_val, axis = 1)

    metrics = {"accuracy": float(accuracy_score(y_val, preds_val)),
               "f1_weighted":float(f1_score(y_val, preds_val, average("weighted")),
               "report":classification_report(y_val, preds_val, digits=4),
               "val_size":int(len(y_val)),
               "train_size":int(len(train_df)),
    }
    return model, train_df, val_df, metrics

def decide_trade(probs: np.ndarray) -> int:
    p_down, p_neutral, p_up = probs[0], probs[1], probs[2]
    if(p_up >= P_ENTER) and ((p_up - p_down) >= MARGIN):
        return +1
    if(p_down >= P_ENTER) and ((p_down - p_up) >= MARGIN):
        return -1
    return 0

def compute_trade_pnl(entry_px: float, exit_px: float, side: int, notional: float) -> float:
    if entry_px <= 0 or exit_px <= 0:
        return 0.0

    gross_ret = side * (exit_px - entry_px) / entry_px

    total_cost_frac = 2.0 * (bps_to_frac(FEE_BPS_PER_SIDE) + bps_to_frac(SLIPPAGE_BPS_PER_SIDE))
    net_ret = gross_ret - total_cost_frac

    return float(notional * net_ret)

def backtest_on_validation(def_val: pd.DataFrame, model: XGBClassifier, symbol: str) -> pd.DataFrame:
    trades = []

    for evt, g in df_val.groupby("whale_timestap"):
        g = g.sort_values("timestamp")
        entry_row = g.iloc[0]
        entry_time = entry_row["timestamp"]
        entry_px = float(entry_row["close"])

        X_row = entry_row[FEATURES].values.reshape(1, -1).astype(np.float32)
        probs = model.predict_proba(X_row)[0]
        signal = decide_trade(probs)

        if signal == 0:
            continue

        pnl = compute_trade_pnl(entry_px, exit_px, signal, NOTIONAL_PER_TRADE)

        trades.append({
            "symbol": symbol,
            "whale_timestamp": evt,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "side": signal,
            "p_down": float(probs[0]),
            "p_neutral": float(probs[1]),
            "p_up": float(probs[2]),
            "score": float(probs[2] - probs[0]),
            "whale_value": float(entry_row.get("whale_value", np.nan)),
            "pnl":pnl,
            "net_return": pnl/ NOTIONAL_PER_TRADE,
            "holding_seconds": float((exit_time - entry_time).total_seconds()),
        })

    if not trades:
        return pd.DataFrame()


    tdf = pd.DataFrame(trades)
    tdf = tdf.sort_values("entry_time").reset_index(drop=True)

    tdf["equity"] = (1.0 + tdf["net_return"]).cumprod()

    return tdf

def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    gross_win = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()

    eq = trades["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq/peak) - 1.0
    max_dd = float(dd.min())

    rets = trades["net_return"].values
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(len(rets))) if len(rets) > 1 else 0.0

    return {
        "n_trades": int(len(trades)),
        "win_rate": float(len(wins)) / len(trades),
        "avg_net_return": float(trades["net_return"].mean()),
        "median_net_return": float(trades["net_return"].median()),
        "profit_factor": float(gross_win/ (gross_loss + 1e-12)) if gross_loss > 0 else np.inf,
        "total_pnl": float(trades["npl"].sum()),
        "max_drawdown": max_dd,
        "sharpe_trade_based": sharpe,
        "long_share": float((trades["side"] == 1).mean()),
        "short_share": float((trades["side"] == -1).mean()),
    }

def plot_equity(trades: pd.DataFrame, symbol: str)
    if trades.empty:
        return
    plt.figure(figsize=(12,5))
    plt.plot(trades["entry_time"], trades["equity"]),
    plt.title(f"{symbol} Equity Curve (validation trades)")
    plt.xlabel("Time")
    plt.ylabel("Equity (start=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefigh(os.path.join(OUTPUT_DIR, f"{symbol}_equity_curve.png"), dpi = 150)
    plt.close()

def plot_trade_distribution(trades: pd.DataFrame, symbol: str):
    if trades.empty:
        return

    tmp = trades.copy()
    tmp["day"] = tmp["entry_time"].dt.floor("D")
    per_day = tmp.groupby("day").size()

    plt.figure(figsize=(12,4))
    plt.plot(per_day.index, per_day.values)
    plt.title(f"{symbol} Trade Count per Day(validation)")
    plt.xlabel("# trades")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{symbol}_trade_distribution.png"), dpi = 150)
    plt.close()


def plot_market_vs_equity(def_val: pd.DataFrame,trades: pd.DataFrame, symbol: str):
    if df_val.empty:
        return

    market = df_val.sort_values("timestamp")[["timestamp", "close"]].drop_dublicates("timestamp")
    market = market.set_index("timestamp")["close"]

    plt.figure(figsize=(12,5))
    plt.plot(market.index, market.values / market.iloc[0], label="Market (normalized)")
    if not trades.empty:
        plt.plot(trades["entry_time"], trades["equity"], label="Strategy Equity (trade times)")
    plt.title(f"{symbol} Market vs Strategy (validation)")
    plt.xlabel("Time")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{symbol}_market_vs_strategy.png"), dpi=150)
    plt.close()

def plot_example_trades(df_val: pd.DataFrame, trades: pd.DataFrame, symbol: str, n=5):
    if trades.empty:
        return

    examples = trades.head(n).copy()

    out_sub = os.path.join(OUT_DIR, f"{symbol}_examples")
    os.makedirs(out_sub, exist_ok=True)

    for i, tr in examples.iterrows():
        t0 = tr["entry_time"]
        t1 = tr["exit_time"]

        w0 = t0 - pd.Timedelta(seconds=10)
        w1 = t1 - pd.Timedelta(seconds=300)
        window = df_val[(df_val["timestamp"] >= w0) & (df_val["timestamp"] <= w1)].copy()
        if window.empty:
            continue

        plt.figure(figsize=(14,6))
        plt.plot(window["timestamp"], window["close"], linewidth=1.2, label = "Close(1s)")
        plt.axvline(t0, linestyle = "--", linewidth = 2, label = "Entry")
        plt.axvline(t1, linestyle = "--", linewidth = 2, label = "Exit")

        side_lbl = "LONG" if tr["side"] == 1 else "SHORT"
        info = (
            f"Side: {side_lbl}\n"
            f"P(UP)={tr['p_up']:.3f}\n"
            f"P(NEU)={tr['p_neutral']:.3f}\n"
            f"P(DOWN)={tr['p_down']:.3f}\n"
            f"NetRet={tr['net_return'] * 100:.3f}%\n"
            f"Whale=${tr['whale_value']:,.0f}"
        )

        plt.text(window["timestamp"].iloc[0], window["close"].max(),info, fontsize = 10, bbox=dict(facecolor="White",alpha=0.9))
        plt.title(f"{symbol} Example Trade @ {t0} ({side_lbl})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fname = f"example_{i}_{t0.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(os.path.join(out_sub, fname), dpi=150)
        plt.close()

def run_symbol(symbol: str):
    df = load_features(symbol)
    model, train_df, val_df, metrics = train_model_time_split(df)

    model_path = os.path.join(OUT_DIR, f"{symbol}_xgb.json")
    model.save_model(model_path)

    print(f"\n[{symbol}] Train size: {metrics['train_size']} | Val size: {metrics['val_size']}")
    print(f"[{symbol}] Accuracy: {metrics['accuracy']:.4f} | F1(w): {metrics['f1_weighted']:.4f}")
    print(metrics["report"])

    with open(os.path.join(OUT_DIR, f"{symbol}_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"F1 weighted: {metrics['f1_weighted']}\n\n")
        f.write(metrics["report"])

    trades = backtest_on_validation(val_df, model, symbol)
    trades_path = os.path.join(OUTPUT_DIR, f"{symbol}_trades.csv")
    trades.to_csv(trades_path, index=False)

    stats = summarize_trades(trades)
    print(f"\n[{symbol}] Backtest stats:", stats)

    pd.DataFrame([stats]).to_csv(os.path.join(OUT_DIR, f"{symbol}_backtest_stats.csv"), index=False)

    plot_equity(trades, symbol)
    plot_trade_distribution(trades, symbol)
    plot_market_vs_equity(val_df, trades, symbol)
    plot_example_trades(val_df, trades, symbol, n=5)

    return trades, stats

if __name__ == "__main__":
    all_stats = []
    for sym in SYMBOLS:
        trades, stats = run_symbol(sym)
        stats["symbol"] = sym
        all_stats.append(stats)

    pd.DataFrame(all_stats).to_csv(os.path.join(OUT_DIR, "summary_all_symbols.csv"), index=False)
    print("\nDONE. Outputs in:", OUT_DIR)






