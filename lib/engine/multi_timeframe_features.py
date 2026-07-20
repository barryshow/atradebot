# -*- coding: utf-8 -*-
"""
MultiTimeframeFeatureEngine — 多周期特征计算（安全版本）

所有 numpy 运算都受 try/except 保护，单个特征计算失败不影响其他特征。
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from .realtime_feed import RealtimeFeed, RealtimePrice


def compute_1m_features(df_1m: pd.DataFrame) -> dict:
    """从 1m closed K 线计算 Fast Entry 特征"""
    d = df_1m.copy()
    eps = 1e-10
    closes = d["close"].values.astype(float)
    volumes = d["volume"].values.astype(float)
    highs = d["high"].values.astype(float)
    lows = d["low"].values.astype(float)
    n = len(closes)
    f = {}
    if n < 5: return f

    def _safe(key, fn):
        try: f[key] = float(fn())
        except: f[key] = 0.0

    _safe("ret_1m", lambda: closes[-1] / closes[-2] - 1 if closes[-2] > 0 else 0)
    _safe("ret_3m", lambda: closes[-1] / closes[-4] - 1 if n >= 4 else 0)
    _safe("ret_5m", lambda: closes[-1] / closes[-6] - 1 if n >= 6 else 0)
    _safe("momentum_1m", lambda: closes[-1] / closes[-2] - 1)
    _safe("momentum_3m", lambda: closes[-1] / closes[-4] - 1 if n >= 4 else 0)
    _safe("price_acceleration", lambda: (closes[-1]/closes[-2]-1) - (closes[-2]/closes[-3]-1) if n >= 3 else 0)

    # RSI
    try:
        if n >= 16:
            deltas = np.diff(closes[-16:])
            g = np.maximum(deltas, 0); l = np.maximum(-deltas, 0)
            ag = np.mean(g); al = np.mean(l)
            f["rsi_1m"] = float(100 - 100 / (1 + ag / max(al, eps)))
    except: f["rsi_1m"] = 50.0

    # EMA slope
    try:
        if n >= 12:
            ema = pd.Series(closes).ewm(span=12, adjust=False).mean().values
            f["ema_slope_1m"] = float((ema[-1] - ema[-5]) / max(abs(ema[-5]), eps)) if n >= 5 else 0
    except: f["ema_slope_1m"] = 0.0

    # Volatility
    try:
        if n >= 16:
            rets = np.diff(closes[-16:]) / np.maximum(np.abs(closes[-16:-1]), eps)
            f["volatility_1m"] = float(np.std(rets))
    except: f["volatility_1m"] = 0.0

    # ATR
    try:
        if n >= 16:
            h = highs[-15:]; l = lows[-15:]; pc = closes[-16:-1]
            tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
            f["atr_1m"] = float(np.mean(tr) / max(closes[-1], eps))
    except: f["atr_1m"] = 0.0

    # Volume
    _safe("volume_1m", lambda: volumes[-1])
    _safe("volume_ratio_1m", lambda: volumes[-1] / max(np.mean(volumes[-5:]), eps))
    try:
        f["volume_trend_1m"] = float(np.mean(volumes[-5:]) / max(np.mean(volumes[-10:-5]), eps)) if n >= 10 else 1.0
    except: f["volume_trend_1m"] = 1.0

    # VWAP
    try:
        if n >= 14:
            tp = (highs[-14:] + lows[-14:] + closes[-14:]) / 3
            vwap = np.sum(volumes[-14:] * tp) / max(np.sum(volumes[-14:]), eps)
            f["vwap_deviation_1m"] = float(closes[-1] / vwap - 1) if vwap > 0 else 0
    except: f["vwap_deviation_1m"] = 0.0

    return f


def compute_5m_features(df_5m: pd.DataFrame) -> dict:
    """从 5m closed K 线计算中期特征"""
    d = df_5m.copy()
    eps = 1e-10
    closes = d["close"].values.astype(float)
    n = len(closes)
    f = {}
    if n < 3: return f

    def _safe(key, fn):
        try: f[key] = float(fn())
        except: f[key] = 0.0

    _safe("ret_5m_5m", lambda: closes[-1] / closes[-2] - 1 if closes[-2] > 0 else 0)
    _safe("ret_15m_5m", lambda: closes[-1] / closes[-4] - 1 if n >= 4 else 0)

    # EMA trend
    try:
        if n >= 12:
            ema = pd.Series(closes).ewm(span=12, adjust=False).mean().values
            f["ema_trend_5m"] = 1 if ema[-1] > ema[-3] else (-1 if ema[-1] < ema[-3] else 0)
    except: f["ema_trend_5m"] = 0

    # ADX
    try:
        if n >= 16:
            h = d["high"].values.astype(float)[-16:]; l = d["low"].values.astype(float)[-16:]; pc = closes[-17:-1]
            tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-pc), np.abs(l[1:]-pc)))
            up = np.maximum(h[1:]-h[:-1], 0); dn = np.maximum(l[:-1]-l[1:], 0)
            a = float(np.mean(tr)); pdi = 100*float(np.mean(up))/max(a,eps); ndi = 100*float(np.mean(dn))/max(a,eps)
            f["adx_5m"] = float(100*abs(pdi-ndi)/max(pdi+ndi, eps))
    except: f["adx_5m"] = 20.0

    # RSI
    try:
        if n >= 16:
            deltas = np.diff(closes[-16:])
            g = np.maximum(deltas, 0); l = np.maximum(-deltas, 0)
            f["rsi_5m"] = float(100 - 100 / (1 + np.mean(g) / max(np.mean(l), eps)))
    except: f["rsi_5m"] = 50.0

    # ATR
    try:
        if n >= 16:
            h = d["high"].values.astype(float)[-15:]; l = d["low"].values.astype(float)[-15:]; pc = closes[-16:-1]
            tr = np.maximum(h-l, np.maximum(np.abs(h-pc), np.abs(l-pc)))
            f["atr_5m"] = float(np.mean(tr) / max(closes[-1], eps))
    except: f["atr_5m"] = 0.0

    # BB
    try:
        if n >= 20:
            mid = float(np.mean(closes[-20:])); std = float(np.std(closes[-20:]))
            bb_upper = mid + 2*std; bb_lower = mid - 2*std
            f["bb_width_5m"] = float((bb_upper - bb_lower) / max(mid, eps))
            f["bb_pos_5m"] = float((closes[-1] - bb_lower) / max(bb_upper - bb_lower, eps))
    except: f["bb_width_5m"] = 0.0; f["bb_pos_5m"] = 0.5

    return f


def compute_fast_entry_features(
    symbol: str,
    realtime: Optional[RealtimePrice],
    df_1m: Optional[pd.DataFrame],
    df_5m: Optional[pd.DataFrame],
    slow_context: Optional[dict] = None,
) -> dict:
    """组合所有输入为 Fast Entry Model 的特征向量"""
    f = {}

    if realtime:
        f["price"] = realtime.price
        f["bid_ask_spread"] = (realtime.ask - realtime.bid) / max(realtime.price, 0.01) if realtime.ask > 0 and realtime.bid > 0 else 0
        f["change_24h"] = realtime.change_pct / 100.0 if realtime.change_pct else 0
    else:
        f["price"] = 0.0; f["bid_ask_spread"] = 0.0; f["change_24h"] = 0.0

    f1 = compute_1m_features(df_1m) if df_1m is not None else {}
    for k, v in f1.items(): f[k] = v

    f5 = compute_5m_features(df_5m) if df_5m is not None else {}
    for k, v in f5.items(): f[k] = v

    if slow_context:
        f["slow_probability"] = slow_context.get("probability", 0.50)
        f["slow_regime"] = _regime_to_float(slow_context.get("regime", "RANGE"))
        f["slow_trend_strength"] = slow_context.get("trend_strength", 0.0)
        f["slow_volatility"] = slow_context.get("volatility", 0.0)
    else:
        f["slow_probability"] = 0.50; f["slow_regime"] = 0.0
        f["slow_trend_strength"] = 0.0; f["slow_volatility"] = 0.0

    f["fast_slow_agreement"] = 1.0 if (f["slow_probability"] > 0.50) == (f.get("ret_1m", 0) > 0) else 0.0
    f["price_volatility_product"] = f.get("volatility_1m", 0) * abs(f.get("ret_1m", 0))

    return f


FAST_FEATURES = [
    "price", "bid_ask_spread", "change_24h",
    "ret_1m", "ret_3m", "ret_5m", "momentum_1m", "momentum_3m",
    "price_acceleration", "rsi_1m", "ema_slope_1m",
    "volatility_1m", "atr_1m", "volume_1m", "volume_ratio_1m", "volume_trend_1m",
    "vwap_deviation_1m",
    "ret_5m_5m", "ret_15m_5m", "ema_trend_5m",
    "adx_5m", "rsi_5m", "atr_5m", "bb_width_5m", "bb_pos_5m",
    "slow_probability", "slow_regime", "slow_trend_strength", "slow_volatility",
    "fast_slow_agreement", "price_volatility_product",
]


def _regime_to_float(regime: str) -> float:
    mapping = {"TREND_UP": 1.0, "TREND_DOWN": -1.0, "RANGE": 0.0,
               "HIGH_VOLATILITY": 0.5, "LOW_LIQUIDITY": -0.5, "EVENT_RISK": -1.0}
    return mapping.get(regime, 0.0)


def build_fast_feature_vector(features: dict) -> np.ndarray:
    vec = np.array([features.get(name, 0.0) for name in FAST_FEATURES], dtype=np.float64)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)