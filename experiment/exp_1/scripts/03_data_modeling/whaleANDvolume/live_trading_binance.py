import os
import time
import hmac
import hashlib
import urllib.parse
import asyncio
import json
import csv
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from xgboost import XGBClassifier

# =========================
# SETTINGS
# =========================
SYMBOL = "BTCUSDT"

MIN_WHALE_USD = 250_000            # ✅
PRE_WINDOW = 30                    # seconds
POST_WINDOW = 30                   # seconds (needed for your features)
POST_WINDOW_10 = 10
HORIZON = 30                       # exit after N seconds (after entry)
COOLDOWN_SEC = 60                  # block new trades after a cycle
TRADE_USDT = 20.0                  # position size on Spot Testnet

# =========================
# PATHS
# =========================
BASE = "/Users/alperademgencer/PycharmProjects/Crypto-Whale-Detection/1/experiment/exp_1"
MODEL_PATH = f"{BASE}/model/BTCUSDT_xgb_whale_volume_binary_always.json"

# ✅ CSV Log Path
LOG_DIR = Path(f"{BASE}/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_LOG_PATH = LOG_DIR / "live_trading_log.csv"

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

# =========================
# BINANCE TESTNET REST
# =========================
BASE_URL = "https://testnet.binance.vision"
API_KEY = os.environ["BINANCE_TESTNET_API_KEY"]
API_SECRET = os.environ["BINANCE_TESTNET_API_SECRET"]

SESSION = requests.Session()
SESSION.headers.update({"X-MBX-APIKEY": API_KEY})


def _sign(params: dict) -> dict:
    qs = urllib.parse.urlencode(params, doseq=True)
    sig = hmac.new(API_SECRET.encode("utf-8"), qs.encode("utf-8"), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def _get(path: str, params=None):
    r = SESSION.get(BASE_URL + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _post(path: str, params=None):
    r = SESSION.post(BASE_URL + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_account():
    params = _sign({"timestamp": int(time.time() * 1000)})
    return _get("/api/v3/account", params)


def get_asset_free(account, asset: str) -> float:
    for b in account.get("balances", []):
        if b.get("asset") == asset:
            return float(b.get("free", 0.0))
    return 0.0


def place_market_buy(symbol: str, quote_usdt: float):
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": f"{quote_usdt:.2f}",
        "timestamp": int(time.time() * 1000),
    }
    params = _sign(params)
    return _post("/api/v3/order", params)


def place_market_sell(symbol: str, base_qty: float):
    params = {
        "symbol": symbol,
        "side": "SELL",
        "type": "MARKET",
        "quantity": f"{base_qty:.6f}",
        "timestamp": int(time.time() * 1000),
    }
    params = _sign(params)
    return _post("/api/v3/order", params)


# =========================
# MODEL LOADING
# =========================
def load_booster(model_path: str):
    m = XGBClassifier()
    m.load_model(model_path)
    return m.get_booster()


def predict_p_buy(booster, feats: dict) -> float:
    X = np.array([[feats[f] for f in FEATURES]], dtype=np.float32)
    dmat = xgb.DMatrix(X, feature_names=FEATURES)
    # objective="binary:logistic" => booster.predict returns probability for class 1 (BUY)
    return float(booster.predict(dmat)[0])


# =========================
# LIVE VOLUME CACHE (1s)
# =========================
@dataclass
class WhaleEvent:
    t0_sec: int
    price: float
    qty: float
    value_usd: float
    whale_side: int  # +1 buy, -1 sell


class VolumeCache:
    """Stores per-second volume in quote terms (USD ~ USDT) for recent history."""
    def __init__(self, maxlen_seconds: int = 300):
        self.maxlen = maxlen_seconds
        self.vol_by_sec = defaultdict(float)
        self.secs = deque()

    def add_trade(self, t_sec: int, quote_vol: float):
        if not self.secs or t_sec > self.secs[-1]:
            self.secs.append(t_sec)
        self.vol_by_sec[t_sec] += float(quote_vol)

        while len(self.secs) > self.maxlen:
            old = self.secs.popleft()
            self.vol_by_sec.pop(old, None)

    def window(self, start_sec: int, end_sec: int):
        secs = np.arange(start_sec, end_sec + 1, dtype=np.int64)
        vols = np.array([self.vol_by_sec.get(int(s), 0.0) for s in secs], dtype=np.float64)
        return secs, vols


def build_features(cache: VolumeCache, evt: WhaleEvent) -> dict:
    t0 = evt.t0_sec

    # pre 30s: [t0-30, t0-1]
    _, pre_vols = cache.window(t0 - PRE_WINDOW, t0 - 1)
    pre_sum = float(pre_vols.sum())
    pre_mean = float(pre_vols.mean()) if len(pre_vols) else 0.0
    pre_std = float(pre_vols.std(ddof=0)) if len(pre_vols) else 0.0

    # post 10s: [t0, t0+9]
    _, post10 = cache.window(t0, t0 + POST_WINDOW_10 - 1)
    post10_sum = float(post10.sum())

    # post 30s: [t0, t0+29]
    post_secs, post30 = cache.window(t0, t0 + POST_WINDOW - 1)
    post30_sum = float(post30.sum())
    peak = float(post30.max()) if len(post30) else 0.0

    # time to peak
    if len(post30) and peak > 0:
        peak_idx = int(np.argmax(post30))
        time_to_peak = float(post_secs[peak_idx] - t0)
    else:
        time_to_peak = float(POST_WINDOW)

    reaction_ratio = (post30_sum / pre_sum) if pre_sum > 0 else 0.0

    return {
        "whale_value_usd": float(evt.value_usd),
        "whale_qty": float(evt.qty),
        "whale_side": float(evt.whale_side),

        "volume_sum_pre_30s": pre_sum,
        "volume_mean_pre_30s": pre_mean,
        "volume_std_pre_30s": pre_std,

        "volume_sum_post_10s": post10_sum,
        "volume_sum_post_30s": post30_sum,
        "volume_peak_post_30s": peak,
        "time_to_volume_peak": time_to_peak,
        "volume_reaction_ratio": reaction_ratio,
    }


# =========================
# CSV LOGGING
# =========================
CSV_COLUMNS = [
    "ts_iso",
    "symbol",
    "ws_trade_time_ms",

    "whale_value_usd",
    "whale_qty",
    "whale_side",

    "p_buy",
    "action",

    # feature snapshot
    "volume_sum_pre_30s",
    "volume_mean_pre_30s",
    "volume_std_pre_30s",
    "volume_sum_post_10s",
    "volume_sum_post_30s",
    "volume_peak_post_30s",
    "time_to_volume_peak",
    "volume_reaction_ratio",

    # balances before decision
    "usdt_free_before",
    "btc_free_before",

    # orders / execution info
    "buy_order_id",
    "sell_order_id",
    "error",
]


def _ensure_csv_header():
    if not CSV_LOG_PATH.exists() or CSV_LOG_PATH.stat().st_size == 0:
        with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def log_row(row: dict):
    # fill missing columns with empty
    out = {k: row.get(k, "") for k in CSV_COLUMNS}
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(out)


# =========================
# WEBSOCKET: public trade stream (TESTNET)
# =========================
# ✅ Correct Testnet WS host:
WS_URL = "wss://stream.testnet.binance.vision/ws/btcusdt@aggTrade"


async def live_trader():
    import websockets  # pip install websockets

    _ensure_csv_header()

    booster = load_booster(MODEL_PATH)
    cache = VolumeCache(maxlen_seconds=600)

    last_cycle_end = 0.0

    print("Connected settings:")
    print("  MIN_WHALE_USD:", MIN_WHALE_USD)
    print("  TRADE_USDT:", TRADE_USDT)
    print("  Delayed entry: wait 30s post-event to build features")
    print("  Spot Testnet: SELL is closing position (no real short)")
    print("  WS_URL:", WS_URL)
    print("  CSV_LOG:", str(CSV_LOG_PATH))

    # Reconnect loop
    while True:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_queue=1000,
            ) as ws:
                print("✅ WS connected:", WS_URL)

                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=120)
                    data = json.loads(msg)

                    # aggTrade fields: T=ms, p=price, q=qty, m=isBuyerMaker
                    t_ms = int(data["T"])
                    price = float(data["p"])
                    qty = float(data["q"])
                    is_buyer_maker = bool(data["m"])

                    t_sec = t_ms // 1000
                    value_usd = price * qty

                    cache.add_trade(t_sec, value_usd)

                    now = time.time()
                    if now < last_cycle_end:
                        continue

                    if value_usd >= MIN_WHALE_USD:
                        whale_side = -1 if is_buyer_maker else 1
                        evt = WhaleEvent(
                            t0_sec=t_sec,
                            price=price,
                            qty=qty,
                            value_usd=value_usd,
                            whale_side=whale_side,
                        )

                        print(
                            f"\n🐋 Whale detected @ {pd.to_datetime(t_ms, unit='ms')}: "
                            f"value={value_usd:,.0f} USD side={'BUY' if whale_side > 0 else 'SELL'}"
                        )

                        # Collect post-window data for features
                        await asyncio.sleep(POST_WINDOW)

                        feats = build_features(cache, evt)
                        p_buy = predict_p_buy(booster, feats)
                        action = "BUY" if p_buy >= 0.5 else "SELL"

                        print(f"Model p_buy={p_buy:.3f} -> action={action}")

                        # Prepare log row base
                        row = {
                            "ts_iso": pd.Timestamp.utcnow().isoformat(),
                            "symbol": SYMBOL,
                            "ws_trade_time_ms": t_ms,

                            "whale_value_usd": feats["whale_value_usd"],
                            "whale_qty": feats["whale_qty"],
                            "whale_side": feats["whale_side"],

                            "p_buy": p_buy,
                            "action": action,

                            # features
                            "volume_sum_pre_30s": feats["volume_sum_pre_30s"],
                            "volume_mean_pre_30s": feats["volume_mean_pre_30s"],
                            "volume_std_pre_30s": feats["volume_std_pre_30s"],
                            "volume_sum_post_10s": feats["volume_sum_post_10s"],
                            "volume_sum_post_30s": feats["volume_sum_post_30s"],
                            "volume_peak_post_30s": feats["volume_peak_post_30s"],
                            "time_to_volume_peak": feats["time_to_volume_peak"],
                            "volume_reaction_ratio": feats["volume_reaction_ratio"],
                        }

                        buy_order_id = ""
                        sell_order_id = ""
                        err = ""

                        try:
                            acc = get_account()
                            usdt_free = get_asset_free(acc, "USDT")
                            btc_free = get_asset_free(acc, "BTC")

                            row["usdt_free_before"] = usdt_free
                            row["btc_free_before"] = btc_free

                            if action == "BUY":
                                if usdt_free < TRADE_USDT:
                                    print(f"⚠️ Not enough USDT on testnet: {usdt_free:.2f}. Skipping.")
                                else:
                                    print("🟢 Placing BUY market order...")
                                    resp = place_market_buy(SYMBOL, TRADE_USDT)
                                    buy_order_id = str(resp.get("orderId", ""))
                                    print("BUY OK:", resp)

                                    # exit after horizon
                                    await asyncio.sleep(HORIZON)
                                    acc2 = get_account()
                                    btc_free2 = get_asset_free(acc2, "BTC")
                                    if btc_free2 > 0:
                                        print("🔴 Exiting (SELL all BTC)...")
                                        resp2 = place_market_sell(SYMBOL, btc_free2)
                                        sell_order_id = str(resp2.get("orderId", ""))
                                        print("SELL OK:", resp2)
                                    else:
                                        print("⚠️ No BTC to sell at exit time.")
                            else:
                                # SELL signal on Spot = close if you have BTC
                                if btc_free <= 0:
                                    print("⚠️ SELL signal but no BTC held (Spot can't short). Skipping.")
                                else:
                                    print("🔴 SELL (closing BTC position)...")
                                    resp = place_market_sell(SYMBOL, btc_free)
                                    sell_order_id = str(resp.get("orderId", ""))
                                    print("SELL OK:", resp)

                        except requests.HTTPError as e:
                            err = e.response.text if e.response is not None else str(e)
                            print("❌ REST error:", err)
                        except Exception as e:
                            err = str(e)
                            print("❌ Error:", err)

                        # Write CSV log
                        row["buy_order_id"] = buy_order_id
                        row["sell_order_id"] = sell_order_id
                        row["error"] = err
                        log_row(row)

                        # cooldown
                        last_cycle_end = time.time() + COOLDOWN_SEC
                        print(f"Cooldown {COOLDOWN_SEC}s...")

        except asyncio.TimeoutError:
            print("⏳ WS recv timeout (no messages). Reconnecting in 3s...")
            await asyncio.sleep(3)

        except ConnectionResetError as e:
            print(f"🔌 TCP reset: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)

        except Exception as e:
            print(f"🔌 WS error/disconnect: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3)


def main():
    asyncio.run(live_trader())


if __name__ == "__main__":
    main()
