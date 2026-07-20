# -*- coding: utf-8 -*-
"""
LiveDataFeed — 实时 Gate.io 数据源

替代 hibt_ticks.csv 的静态数据，每 tick 拉取最新 15m K 线。
- 缓存 60 秒，避免 API 限流
- 自动补齐历史数据（首次拉取 200 根 K 线用于特征计算）
- 支持回退到 CSV 文件
"""
import time
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict
from curl_cffi import requests

API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
SYMBOL_PAIRS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}

# 缓存
_cache: Dict[str, pd.DataFrame] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL = 60  # 秒


def fetch_klines(pair: str, interval: str = "15m", limit: int = 200) -> Optional[pd.DataFrame]:
    """从 Gate.io 拉取 K 线"""
    try:
        r = requests.get(API_URL, params={
            "currency_pair": pair, "interval": interval, "limit": limit,
        }, impersonate="chrome110", timeout=10, verify=False)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data or len(data) < 2:
            return None
    except Exception:
        return None

    df = pd.DataFrame(data, columns=["ts", "qv", "close", "high", "low", "open", "volume", "final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    return df


def get_live_data(symbol: str, interval: str = "15m", limit: int = 200) -> Optional[pd.DataFrame]:
    """
    获取实时数据（带缓存）。

    Returns:
        DataFrame with datetime index, columns: open, high, low, close, volume
    """
    now = time.time()
    cache_key = f"{symbol}_{interval}"
    if cache_key in _cache and (now - _cache_ts.get(cache_key, 0)) < CACHE_TTL:
        return _cache[cache_key]

    pair = SYMBOL_PAIRS.get(symbol)
    if not pair:
        return None

    df = fetch_klines(pair, interval=interval, limit=limit)
    if df is not None and len(df) > 0:
        _cache[cache_key] = df
        _cache_ts[cache_key] = now
    return df


def build_tick_dataframe(live_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    将各品种的实时 DataFrame 合并为 engine.tick() 期望的格式。

    engine.tick() 期望的 CSV 格式:
    columns: ts, symbol, open, high, low, close, volume

    Returns:
        DataFrame with columns: ts, symbol, open, high, low, close, volume, datetime
    """
    rows = []
    for symbol, df in live_data.items():
        if df is None or df.empty:
            continue
        for dt, row in df.iterrows():
            ts = int(dt.timestamp() * 1000)
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "datetime": dt,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_csv_fallback(csv_path: str) -> Optional[pd.DataFrame]:
    """从 CSV 文件加载数据（回退方案）"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        df = df.iloc[:, :7]
        df.columns = ["ts", "symbol", "open", "high", "low", "close", "volume"]
        if str(df.iloc[0, 1]).strip() == "symbol":
            df = df.iloc[1:].copy()
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts", "open", "high", "low", "close"])
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
        df = df.dropna(subset=["datetime"])
        return df
    except Exception:
        return None


def get_live_data_for_engine(symbols: list, interval: str = "15m", limit: int = 200,
                              csv_fallback: str = "") -> Dict[str, pd.DataFrame]:
    """
    为引擎获取所有品种的实时数据。

    Returns:
        Dict[symbol] → DataFrame
    """
    result = {}
    for sym in symbols:
        df = get_live_data(sym, interval=interval, limit=limit)
        if df is not None:
            result[sym] = df

    # 如果所有品种都失败，尝试 CSV 回退
    if not result and csv_fallback:
        csv_df = load_csv_fallback(csv_fallback)
        if csv_df is not None:
            for sym in symbols:
                ds = csv_df[csv_df["symbol"] == sym]
                if not ds.empty:
                    result[sym] = ds

    return result