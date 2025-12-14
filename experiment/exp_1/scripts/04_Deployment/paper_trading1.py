import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from binance.client import Client
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
START = "7 days ago UTC"
END = None

WINDOW_Z = 60
Z_THRESH = 4.0
HORIZON = 30

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
# LOAD MODEL (FIXED PATH)
# ======================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR.parent / "03_data_modeling" / "xgb_volume_model.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

booster = xgb.Booster()
booster.load_model(MODEL_PATH)

print(f"Loaded model from {MODEL_PATH}")

# ======================================================
# LOAD DATA VIA API (HISTORICAL BUT REAL)
# ======================================================
client = Client()

klines = client.get_historical_klines(
    SYMBOL,
    INTERVAL,
    start_str=START,
    end_str=END
)

bars = []
for k in klines:
    bars.append({
        "timestamp": pd.to_datetime(k[0], unit="ms"),
        "close": float(k[4]),
        "volume_usd": float(k[7]),
        "buy_volume": float(k[9]),
        "sell_volume": float(k[7]) - float(k[9]),
    })

df = pd.DataFrame(bars).set_index("timestamp")
print(f"Loaded {len(df)} replay bars")

# ======================================================
# REPLAY LOOP (LIVE-LIKE)
# ======================================================
up, down, neutral = [], [], []

for i in range(WINDOW_Z, len(df) - 1):

    window = df.iloc[i - WINDOW_Z:i].copy()
    current = df.iloc[i]

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
    probs = booster.predict(xgb.DMatrix(X, feature_names=FEATURES))[0]
    p_down, p_neutral, p_up = probs

    label = "NEUTRAL"
    if p_up >= P_UP_ENTER and (p_up - p_down) >= MARGIN and p_neutral <= P_NEUTRAL_MAX:
        label = "UP"
    elif p_down >= P_DOWN_ENTER and (p_down - p_up) >= MARGIN and p_neutral <= P_NEUTRAL_MAX:
        label = "DOWN"

    target_list = {"UP": up, "DOWN": down, "NEUTRAL": neutral}[label]
    if len(target_list) >= MAX_PER_CLASS:
        continue

    target_list.append((current.name, label, feats, probs))

    if len(up) >= 5 and len(down) >= 5 and len(neutral) >= 5:
        break

print(f"Collected: UP={len(up)}, DOWN={len(down)}, NEUTRAL={len(neutral)}")

# ======================================================
# QUICK PLOTS
# ======================================================
def plot_event(ts, label, idx):
    win = df.loc[ts - pd.Timedelta(minutes=2): ts + pd.Timedelta(minutes=2)]
    plt.figure(figsize=(12, 5))
    plt.plot(win.index, win.close)
    plt.axvline(ts, linestyle="--", label="Event")
    plt.title(f"{label} Replay Trade #{idx}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"paper_replay_{label.lower()}_{idx}.png")
    plt.close()

for i, (ts, _, _, _) in enumerate(up, 1):
    plot_event(ts, "UP", i)

for i, (ts, _, _, _) in enumerate(down, 1):
    plot_event(ts, "DOWN", i)

for i, (ts, _, _, _) in enumerate(neutral, 1):
    plot_event(ts, "NEUTRAL", i)

print("DONE – replay paper trading plots created")
