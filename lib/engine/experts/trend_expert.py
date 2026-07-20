# -*- coding: utf-8 -*-
"""
TrendExpert — 趋势行情专家模型

负责趋势行情中的方向预测。
特征: EMA slope, multi-timeframe returns, ADX, momentum, volume trend
"""
import numpy as np
from typing import Optional, Dict
from ..models import ExpertPrediction


class TrendExpert:
    """
    趋势专家。

    逻辑: 在趋势行情中，方向倾向于延续。
    - ADX > 25 且 price > MA20 → 倾向于 CALL
    - ADX > 25 且 price < MA20 → 倾向于 PUT
    """

    def __init__(self):
        self.name = "trend"
        self.model_version = "v1.0"

    def predict(self, symbol: str, indicators: Dict, row: Dict) -> ExpertPrediction:
        """
        预测方向概率。

        Args:
            symbol: 品种
            indicators: 指标字典
            row: 特征行

        Returns:
            ExpertPrediction
        """
        adx = float(indicators.get("ADX", 20))
        price_vs_ma20 = float(indicators.get("price_vs_MA20", 0))
        macd = float(indicators.get("MACD", 0))
        ma_trend = float(indicators.get("MA_trend", 0))
        ret_1 = float(row.get("ret_1", 0))
        ret_3 = float(row.get("ret_3", 0))
        ret_6 = float(row.get("ret_6", 0))

        # 趋势强度 (0-1)
        trend_strength = min(1.0, max(0.0, adx / 40.0))

        # 方向信号
        direction_score = 0.0
        if price_vs_ma20 > 0.001:
            direction_score += 0.3
        elif price_vs_ma20 < -0.001:
            direction_score -= 0.3

        if macd > 0:
            direction_score += 0.2
        elif macd < 0:
            direction_score -= 0.2

        if ma_trend > 0:
            direction_score += 0.15
        elif ma_trend < 0:
            direction_score -= 0.15

        # 短期动量
        if ret_1 > 0:
            direction_score += 0.1
        if ret_3 > 0:
            direction_score += 0.1
        if ret_6 > 0:
            direction_score += 0.05

        # 转换为概率
        # direction_score ∈ [-0.75, 0.75]
        # 基础概率 0.50，趋势越强方向越极端
        raw_prob = 0.50 + direction_score * trend_strength * 0.35
        raw_prob = max(0.35, min(0.65, raw_prob))

        direction = 1 if raw_prob >= 0.50 else 2
        direction_str = "CALL" if direction == 1 else "PUT"

        # 如果不是 CALL，调整概率
        if direction == 2:
            raw_prob = 1.0 - raw_prob
            # 确保 PUT 方向概率有意义
            if raw_prob < 0.50:
                raw_prob = 0.50 + (0.50 - raw_prob)

        return ExpertPrediction(
            expert_name=self.name,
            symbol=symbol,
            direction=direction,
            direction_str=direction_str,
            raw_probability=round(raw_prob, 4),
            calibrated_probability=round(raw_prob, 4),  # 后续通过 Meta 校准
            confidence=round(trend_strength, 3),
            features_used=["ADX", "price_vs_MA20", "MACD", "MA_trend", "ret_1", "ret_3", "ret_6"],
            model_version=self.model_version,
        )