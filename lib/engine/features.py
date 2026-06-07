# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def calc_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()
    eps = 1e-10
    df["volume"] = df["volume"].fillna(0).replace(0, 0.001)

    # Time features
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # MACD
    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    df["MACD"] = 2 * (macd_line - macd_line.ewm(span=9, adjust=False).mean())
    df["macd_hist_change"] = df["MACD"] - df["MACD"].shift(1)

    # KDJ
    low_9 = df["low"].rolling(9).min()
    high_9 = df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + eps) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    df["J"] = 3 * k - 2 * d

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["rsi_change"] = df["RSI"] - df["RSI"].shift(5)

    # Bollinger Bands
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["BB_Pos"] = (df["close"] - (mid - 2 * std)) / (4 * std + eps)
    df["bb_width"] = ((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)

    # NATR / ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["NATR"] = tr.rolling(14).mean() / (df["close"] + eps)
    df["volatility_ratio"] = df["NATR"] / (df["NATR"].rolling(20).mean() + eps)

    # ADX
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    tr_smooth = tr.rolling(14).sum().replace(0, eps)
    plus_di = 100 * plus_dm.rolling(14).sum() / tr_smooth
    minus_di = 100 * minus_dm.rolling(14).sum() / tr_smooth
    df["ADX"] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + eps)).rolling(14).mean().fillna(0)
    df["adx_change"] = df["ADX"] - df["ADX"].shift(5)

    # VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (df["volume"] * tp).cumsum() / (df["volume"].cumsum() + eps)
    df["VWAP_Dist"] = (df["close"] - vwap) / (vwap + eps)

    # Volume
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(5).mean() + eps)

    # Buy/Sell Pressure
    hl = (df["high"] - df["low"]) + eps
    buy_raw = (df["close"] - df["low"]) / hl * df["volume"]
    sell_raw = (df["high"] - df["close"]) / hl * df["volume"]
    for w in [5, 15, 30]:
        df[f"BSP_{w}"] = np.log((buy_raw.rolling(w).sum() + eps) / (sell_raw.rolling(w).sum() + eps))

    # Volume Energy
    df["VEV"] = df["volume_ratio"] / (df["NATR"] + eps)

    # Moving averages
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["close_to_ma50"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).mean() + eps)
    df["Macro_Trend"] = (df["close"] - df["close"].ewm(span=100, adjust=False).mean()) / (df["close"].ewm(span=100, adjust=False).mean() + eps)

    # Momentum
    df["momentum_3"] = df["close"] - df["close"].shift(3)

    # Candle patterns
    hl_range = df["high"] - df["low"] + eps
    df["wick_upper_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range
    df["wick_lower_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl_range
    df["body_ratio"] = (df["close"] - df["open"]).abs() / hl_range

    # CCI
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - tp_sma) / (0.015 * tp_mad + eps)

    # Choppiness
    atr_1 = tr
    sum_atr = atr_1.rolling(14).sum()
    highest_high = df["high"].rolling(14).max()
    lowest_low = df["low"].rolling(14).min()
    df["CHOP"] = 100 * np.log10(sum_atr / (highest_high - lowest_low + eps)) / np.log10(14)

    # ROC
    df["ROC_5"] = (df["close"] - df["close"].shift(5)) / (df["close"].shift(5) + eps) * 100

    # OBV slope
    obv_direction = np.sign(df["close"].diff().fillna(0))
    obv = (df["volume"] * obv_direction).cumsum()
    df["OBV_slope_5"] = obv.diff(5) / (obv.shift(5).abs() + eps)

    # Fill missing
    for col in [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "J", "rsi_change",
        "bb_width", "NATR", "volatility_ratio", "adx_change", "VWAP_Dist",
        "volume_ratio", "BSP_15", "BSP_30", "VEV", "close_to_ma50",
        "Macro_Trend", "momentum_3", "wick_upper_ratio", "wick_lower_ratio",
        "body_ratio", "CCI", "CHOP", "ROC_5", "OBV_slope_5",
    ]:
        if col not in df:
            df[col] = 0

    return df.replace([np.inf, -np.inf], np.nan).dropna()
