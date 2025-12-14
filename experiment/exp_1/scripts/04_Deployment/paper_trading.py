import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from binance.client import Client
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
SYMBOL = "BTCUSDT"

START = "7 days ago UTC"
END = None

WINDOW_Z = 60
Z_THRESH = 4.0
HORIZON = 30  # Sekunden

WINDOW_BEFORE_SEC = 120
WINDOW_AFTER_SEC = 120

P_UP_ENTER = 0.70
P_DOWN_ENTER = 0.75
MARGIN = 0.30
P_NEUTRAL_MAX = 0.20

FEATURES = [
    "volume_z",
    "volume_usd",
    "buy_volume",
    "sell_volume",
    "buy_sell_imbalance",
    "return_pre_1s",
    "return_pre_5s",
    "volatility_pre_30s",
]

MAX_PER_CLASS = 5

# ======================================================
# LOAD MODEL
# ======================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "03_data_modeling" / "xgb_volume_model.json"

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

print(f"Loaded model from {MODEL_PATH}")

# ======================================================
# LOAD 1-MIN DATA (EVENT DETECTION – API)
# ======================================================
client = Client()

klines_1m = client.get_historical_klines(
    SYMBOL,
    Client.KLINE_INTERVAL_1MINUTE,
    start_str=START,
    end_str=END
)

bars_1m = []
for k in klines_1m:
    bars_1m.append({
        "timestamp": pd.to_datetime(k[0], unit="ms"),
        "close": float(k[4]),
        "volume_usd": float(k[7]),
        "buy_volume": float(k[9]),
        "sell_volume": float(k[7]) - float(k[9]),
    })

df_1m = pd.DataFrame(bars_1m).set_index("timestamp").sort_index()
print(f"Loaded {len(df_1m)} 1-min bars")

# ======================================================
# LOAD 1-SECOND DATA (LOCAL – REALISTIC PLOT)
# ======================================================
bars_1s = pd.read_parquet(
    "/Users/alperademgencer/PycharmProjects/"
    "Crypto-Whale-Detection/1/experiment/exp_1/"
    "data/raw/Bars_1s/BTCUSDT/"
    "BTCUSDT_1s_202406_202506.parquet",
    columns=["timestamp", "close"]
)

bars_1s["timestamp"] = pd.to_datetime(bars_1s["timestamp"])
df_1s = bars_1s.set_index("timestamp").sort_index()
print(f"Loaded {len(df_1s)} 1-sec bars (local)")

# ======================================================
# EVENT DETECTION + MODEL
# ======================================================
up, down, neutral = [], [], []

for i in range(WINDOW_Z, len(df_1m) - 1):

    window = df_1m.iloc[i - WINDOW_Z:i]
    current = df_1m.iloc[i]

    vol_z = (current.volume_usd - window.volume_usd.mean()) / (window.volume_usd.std() + 1e-9)
    if vol_z < Z_THRESH:
        continue

    feats = {
        "volume_z": vol_z,
        "volume_usd": current.volume_usd,
        "buy_volume": current.buy_volume,
        "sell_volume": current.sell_volume,
        "buy_sell_imbalance": (current.buy_volume - current.sell_volume)
        / (current.buy_volume + current.sell_volume + 1e-9),
        "return_pre_1s": window.close.pct_change().iloc[-1],
        "return_pre_5s": window.close.pct_change(5).iloc[-1],
        "volatility_pre_30s": window.close.pct_change().rolling(30).std().iloc[-1],
    }

    X = np.array([feats[f] for f in FEATURES]).reshape(1, -1)
    p_down, p_neutral, p_up = booster.predict(
        xgb.DMatrix(X, feature_names=FEATURES)
    )[0]

    label = "NEUTRAL"
    if p_up >= P_UP_ENTER and (p_up - p_down) >= MARGIN and p_neutral <= P_NEUTRAL_MAX:
        label = "UP"
    elif p_down >= P_DOWN_ENTER and (p_down - p_up) >= MARGIN and p_neutral <= P_NEUTRAL_MAX:
        label = "DOWN"

    delta_v = current.buy_volume - current.sell_volume
    dominance = "BUY dominant" if delta_v > 0 else "SELL dominant"

    target = {"UP": up, "DOWN": down, "NEUTRAL": neutral}[label]
    if len(target) >= MAX_PER_CLASS:
        continue

    target.append((
        current.name,
        label,
        feats,
        (p_down, p_neutral, p_up),
        delta_v,
        dominance
    ))

    if len(up) >= MAX_PER_CLASS and len(down) >= MAX_PER_CLASS and len(neutral) >= MAX_PER_CLASS:
        break

print(f"Collected: UP={len(up)}, DOWN={len(down)}, NEUTRAL={len(neutral)}")

# ======================================================
# PLOT FUNCTION (ROBUST – NO CRASH)
# ======================================================
def plot_event(ts, label, idx, feats, probs, delta_v, dominance):

    p_down, p_neutral, p_up = probs

    window = df_1s.loc[
        ts - pd.Timedelta(seconds=WINDOW_BEFORE_SEC):
        ts + pd.Timedelta(seconds=HORIZON + WINDOW_AFTER_SEC)
    ]

    # 🔒 GUARD: keine 1s-Daten → überspringen
    if window.empty:
        print(f"Skip plot {label} #{idx} – no 1s data around {ts}")
        return

    entry_px = df_1s["close"].asof(ts)
    exit_ts = ts + pd.Timedelta(seconds=HORIZON)
    exit_px = df_1s["close"].asof(exit_ts)

    side = "LONG" if label == "UP" else "SHORT" if label == "DOWN" else "NO TRADE"
    ret = (
        (exit_px - entry_px) / entry_px
        if label == "UP"
        else (entry_px - exit_px) / entry_px
        if label == "DOWN"
        else 0.0
    )

    info = (
        f"{label} | {side}\n"
        f"P(UP): {p_up:.2f}\n"
        f"P(DOWN): {p_down:.2f}\n"
        f"P(NEUTRAL): {p_neutral:.2f}\n\n"
        f"{dominance}\n"
        f"ΔVolume: {delta_v:,.0f} USD\n"
        f"Volume Z: {feats['volume_z']:.2f}\n"
        f"Imbalance: {feats['buy_sell_imbalance']:.2f}\n"
        f"Return: {ret*100:.2f}%"
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(window.index, window["close"], linewidth=1.2)

    ax.axvline(ts, color="green", linestyle="--", label="Event / Entry")
    ax.axvline(exit_ts, color="red", linestyle="--", label="Exit")

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
    )

    ax.text(
        window.index[0],
        window["close"].max(),
        info,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9),
        verticalalignment="top"
    )

    ax.set_title(f"Paper Trading Replay – {label} #{idx}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    fig.autofmt_xdate()
    plt.tight_layout()

    fname = f"paper_replay_realistic_{label.lower()}_{idx}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

# ======================================================
# CREATE PLOTS
# ======================================================
for i, (ts, _, feats, probs, delta_v, dominance) in enumerate(up, 1):
    plot_event(ts, "UP", i, feats, probs, delta_v, dominance)

for i, (ts, _, feats, probs, delta_v, dominance) in enumerate(down, 1):
    plot_event(ts, "DOWN", i, feats, probs, delta_v, dominance)

for i, (ts, _, feats, probs, delta_v, dominance) in enumerate(neutral, 1):
    plot_event(ts, "NEUTRAL", i, feats, probs, delta_v, dominance)

print("DONE – realistic, fast & robust paper trading plots created")
