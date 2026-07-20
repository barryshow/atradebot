# -*- coding: utf-8 -*-
"""
MarketRegimeDetector — 市场状态分类器

不预测涨跌，只判断当前市场处于什么状态。
输出: TREND_UP / TREND_DOWN / RANGE / HIGH_VOLATILITY / LOW_LIQUIDITY / EVENT_RISK

特征:
- ATR / realized volatility
- ADX
- EMA slope (20/50)
- Bollinger Band width
- volume z-score
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass
from .models import MarketRegime


@dataclass
class RegimeConfig:
    """Regime 检测阈值"""
    adx_trend_threshold: float = 25.0         # ADX > 25 → 趋势
    adx_strong_trend: float = 35.0            # ADX > 35 → 强趋势
    bb_width_high: float = 0.04               # BB 宽度 > 4% → 高波动
    bb_width_low: float = 0.01                # BB 宽度 < 1% → 低波动
    vol_ratio_extreme: float = 2.5            # vol ratio > 2.5 → 极端波动
    ema_slope_min: float = 0.001              # EMA slope 显著性阈值
    volume_zscore_low: float = -2.0           # 低流动性
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class MarketRegimeDetector:
    """
    市场状态检测器。

    输入: 特征 DataFrame 的最近一行或多行
    输出: MarketRegime 对象
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.cfg = config or RegimeConfig()

    def detect(self, indicators: Dict) -> MarketRegime:
        """
        检测当前市场状态。

        Args:
            indicators: 指标字典，包含 ADX, RSI, BB_Pos, BB_width, ATR_pct,
                       volatility_ratio, MACD, price_vs_MA20, volume_ratio, etc.

        Returns:
            MarketRegime
        """
        adx = float(indicators.get("ADX", 20))
        rsi = float(indicators.get("RSI", 50))
        bb_width = float(indicators.get("bb_width", 0.02))
        vol_ratio = float(indicators.get("volatility_ratio", 1.0))
        atr_pct = float(indicators.get("ATR_pct", 0.003))
        price_vs_ma20 = float(indicators.get("price_vs_MA20", 0))
        macd = float(indicators.get("MACD", 0))
        volume_ratio = float(indicators.get("vol_ratio", 1.0))
        body_pct = float(indicators.get("body_pct", 0.3))

        # ── 1. 事件风险检测 ──
        if vol_ratio > self.cfg.vol_ratio_extreme:
            return MarketRegime(
                regime="HIGH_VOLATILITY",
                confidence=min(0.9, vol_ratio / 5.0),
                atr=round(atr_pct, 6),
                adx=round(adx, 2),
                volatility=round(vol_ratio, 3),
                ema_slope=round(price_vs_ma20, 6),
                bb_width=round(bb_width, 4),
                volume_zscore=round(volume_ratio - 1, 3),
                order_book_imbalance=0.0,
                details={"trigger": "extreme_volatility", "vol_ratio": vol_ratio},
            )

        # ── 2. 低流动性检测 ──
        if volume_ratio < 0.3:
            return MarketRegime(
                regime="LOW_LIQUIDITY",
                confidence=0.7,
                atr=round(atr_pct, 6),
                adx=round(adx, 2),
                volatility=round(vol_ratio, 3),
                ema_slope=round(price_vs_ma20, 6),
                bb_width=round(bb_width, 4),
                volume_zscore=round(volume_ratio - 1, 3),
                order_book_imbalance=0.0,
                details={"trigger": "low_volume", "volume_ratio": volume_ratio},
            )

        # ── 3. 趋势检测 ──
        if adx > self.cfg.adx_trend_threshold:
            # 有趋势：判断方向
            if price_vs_ma20 > self.cfg.ema_slope_min and macd > 0:
                confidence = min(0.9, adx / 60.0)
                return MarketRegime(
                    regime="TREND_UP",
                    confidence=round(confidence, 3),
                    atr=round(atr_pct, 6),
                    adx=round(adx, 2),
                    volatility=round(vol_ratio, 3),
                    ema_slope=round(price_vs_ma20, 6),
                    bb_width=round(bb_width, 4),
                    volume_zscore=round(volume_ratio - 1, 3),
                    order_book_imbalance=0.0,
                    details={"adx": adx, "price_vs_ma20": price_vs_ma20, "macd": macd},
                )
            elif price_vs_ma20 < -self.cfg.ema_slope_min and macd < 0:
                confidence = min(0.9, adx / 60.0)
                return MarketRegime(
                    regime="TREND_DOWN",
                    confidence=round(confidence, 3),
                    atr=round(atr_pct, 6),
                    adx=round(adx, 2),
                    volatility=round(vol_ratio, 3),
                    ema_slope=round(price_vs_ma20, 6),
                    bb_width=round(bb_width, 4),
                    volume_zscore=round(volume_ratio - 1, 3),
                    order_book_imbalance=0.0,
                    details={"adx": adx, "price_vs_ma20": price_vs_ma20, "macd": macd},
                )

        # ── 4. 高波动检测 ──
        if bb_width > self.cfg.bb_width_high or vol_ratio > 1.8:
            return MarketRegime(
                regime="HIGH_VOLATILITY",
                confidence=min(0.8, bb_width / 0.08),
                atr=round(atr_pct, 6),
                adx=round(adx, 2),
                volatility=round(vol_ratio, 3),
                ema_slope=round(price_vs_ma20, 6),
                bb_width=round(bb_width, 4),
                volume_zscore=round(volume_ratio - 1, 3),
                order_book_imbalance=0.0,
                details={"bb_width": bb_width, "vol_ratio": vol_ratio},
            )

        # ── 5. 默认：震荡 ──
        return MarketRegime(
            regime="RANGE",
            confidence=0.6,
            atr=round(atr_pct, 6),
            adx=round(adx, 2),
            volatility=round(vol_ratio, 3),
            ema_slope=round(price_vs_ma20, 6),
            bb_width=round(bb_width, 4),
            volume_zscore=round(volume_ratio - 1, 3),
            order_book_imbalance=0.0,
            details={"adx": adx, "rsi": rsi},
        )

    def detect_from_row(self, row: Dict) -> MarketRegime:
        """从特征行检测"""
        return self.detect(row)


# 全局单例
_regime_detector: Optional[MarketRegimeDetector] = None

def get_regime_detector() -> MarketRegimeDetector:
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector