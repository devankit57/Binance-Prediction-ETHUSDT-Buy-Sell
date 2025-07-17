# live_predict.py

import time
import ccxt
import numpy as np
import pandas as pd
import joblib
import json

from datetime import datetime
import pytz
from ta.trend import ema_indicator, macd_diff
from ta.momentum import rsi
from ta.volatility import average_true_range
from ta.volume import on_balance_volume

# ========== CONFIG ==========
MODEL_PATH = "eth_model_xgb.joblib"
SYMBOL = "ETH/USDT"
TIMEFRAME = "5m"
CONFIDENCE_THRESHOLD = 0.7
FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ret_lag1", "ret_lag2", "ret_lag3",
    "volatility_5", "volatility_15",
    "price_vs_ema9", "price_vs_ema21",
    "rsi", "macd", "atr", "obv",
    "candle_body", "candle_range",
    "hour", "dayofweek",
    "boll_width", "ema_cross", "vol_chg"
]
india_tz = pytz.timezone("Asia/Kolkata")

# ========== UTILS ==========
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret_1"] = df["Close"].pct_change()
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_6"] = df["Close"].pct_change(6)
    df["ret_12"] = df["Close"].pct_change(12)
    df["ret_lag1"] = df["ret_1"].shift(1)
    df["ret_lag2"] = df["ret_1"].shift(2)
    df["ret_lag3"] = df["ret_1"].shift(3)
    df["volatility_5"] = df["Close"].rolling(5).std()
    df["volatility_15"] = df["Close"].rolling(15).std()

    df["ema_9"] = ema_indicator(df["Close"], window=9)
    df["ema_21"] = ema_indicator(df["Close"], window=21)
    df["price_vs_ema9"] = df["Close"] / df["ema_9"] - 1
    df["price_vs_ema21"] = df["Close"] / df["ema_21"] - 1

    df["rsi"] = rsi(df["Close"], window=14)
    df["macd"] = macd_diff(df["Close"])
    df["atr"] = average_true_range(df["High"], df["Low"], df["Close"])
    df["obv"] = on_balance_volume(df["Close"], df["Volume"])

    df["candle_body"] = df["Close"] - df["Open"]
    df["candle_range"] = df["High"] - df["Low"]

    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek

    df["boll_width"] = df["Close"].rolling(20).std() * 2 / df["Close"]
    df["ema_cross"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["vol_chg"] = df["Volume"].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def fetch_recent_data(symbol=SYMBOL, limit=100):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    candles = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
    return df

def make_live_prediction():
    model = joblib.load(MODEL_PATH)
    df = fetch_recent_data()
    df = compute_features(df)
    df.dropna(inplace=True)

    if df.empty or any(col not in df.columns for col in FEATURE_COLUMNS):
        print("⚠️ Missing features for prediction.")
        return

    X_latest = df[FEATURE_COLUMNS].iloc[[-1]]
    X_latest = X_latest.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    if X_latest.shape[1] < len(FEATURE_COLUMNS):
        print("⚠️ Incomplete feature set. Skipping prediction.")
        return

    proba = model.predict_proba(X_latest)[0]
    predicted_class = np.argmax(proba)
    confidence = float(proba[predicted_class])  # Ensure plain float

    timestamp = datetime.now(india_tz).strftime("%Y-%m-%d %H:%M:%S")

   

    signal = "BUY 🟢" if predicted_class == 1 else "SELL 🔴"

    output = {
        "confidence": round(confidence, 2),
        "signal": signal,
        "timestamp": timestamp
    }

    print(json.dumps(output, indent=2))

# ========== MAIN ==========
if __name__ == "__main__":
    print("🚀 Starting Live Prediction Loop...")
    while True:
        make_live_prediction()
        time.sleep(60)  # wait 1 minute
