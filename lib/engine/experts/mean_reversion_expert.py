# -*- coding: utf-8 -*-
"""
MeanReversionExpert — 均值回归专家模型

负责震荡和过度偏离行情中的方向预测。
特征: RSI, Bollinger z-score, VWAP deviation, short-term reversal, price acceleration
"""
import numpy as np
from typing import Optional, Dict
from ..models import ExpertPrediction


class MeanReversionExpert:
    """
    均值回归专家。

    逻辑: 在震荡行情中，价格倾向于回归均值。
    - RSI < 30 + BB_Pos < 0.2 → 超卖 → 倾向于 CALL (反弹)
    - RSI > 70 + BB_Pos > 0.8 → 超买 → 倾向于 PUT (回调)
    """

    def __init__(self):
        self.name = "mean_reversion"
        self.model_version = "v1.0"

    def predict(self, symbol: str, indicators: Dict, row: Dict) -> ExpertPrediction:
        rsi = float(indicators.get("RSI", 50))
        bb_pos = float(indicators.get("BB_Pos", 0.5))
        vwap_dist = float(indicators.get("VWAP_dist", 0))
        cci = float(indicators.get("CCI", 0))
        body_pct = float(row.get("body_pct", 0.3))

        # 均值回归信号
        reversion_score = 0.0

        # RSI 极端
        if rsi < 30:
            reversion_score += 0.25  # 超卖 → 倾向反弹
        elif rsi > 70:
            reversion_score -= 0.25  # 超买 → 倾向回调

        # BB 位置
        if bb_pos < 0.2:
            reversion_score += 0.25  # 下轨 → 反弹
        elif bb_pos > 0.8:
            reversion_score -= 0.25  # 上轨 → 回调

        # VWAP 偏离
        if vwap_dist < -0.02:
            reversion_score += 0.15
        elif vwap_dist > 0.02:
            reversion_score -= 0.15

        # CCI 极端
        if cci < -100:
            reversion_score += 0.15
        elif cci > 100:
            reversion_score -= 0.15

        # 震荡市确认: ADX 低时均值回归更可靠
        adx = float(indicators.get("ADX", 20))
        range_confidence = max(0.0, 1.0 - adx / 35.0)  # ADX 越低越可信

        # 转换为概率
        raw_prob = 0.50 + reversion_score * range_confidence * 0.55
        raw_prob = max(0.35, min(0.65, raw_prob))

        direction = 1 if raw_prob >= 0.50 else 2
        direction_str = "CALL" if direction == 1 else "PUT"

        if direction == 2:
            raw_prob = 1.0 - raw_prob
            if raw_prob < 0.50:
                raw_prob = 0.50 + (0.50 - raw_prob)

        return ExpertPrediction(
            expert_name=self.name,
            symbol=symbol,
            direction=direction,
            direction_str=direction_str,
            raw_probability=round(raw_prob, 4),
            calibrated_probability=round(raw_prob, 4),
            confidence=round(range_confidence, 3),
            features_used=["RSI", "BB_Pos", "VWAP_dist", "CCI", "ADX", "body_pct"],
            model_version=self.model_version,
        )