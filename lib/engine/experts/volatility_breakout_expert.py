# -*- coding: utf-8 -*-
"""
VolatilityBreakoutExpert — 波动突破专家模型

负责波动突然放大时的方向预测。
特征: ATR expansion, realized volatility, volume spike, range breakout, Bollinger bandwidth expansion
"""
import numpy as np
from typing import Optional, Dict
from ..models import ExpertPrediction


class VolatilityBreakoutExpert:
    """
    波动突破专家。

    逻辑: 波动放大时，价格倾向于沿突破方向继续。
    - vol_ratio > 2.0 + price > BB上轨 → 向上突破 → CALL
    - vol_ratio > 2.0 + price < BB下轨 → 向下突破 → PUT
    """

    def __init__(self):
        self.name = "volatility_breakout"
        self.model_version = "v1.0"

    def predict(self, symbol: str, indicators: Dict, row: Dict) -> ExpertPrediction:
        vol_ratio = float(indicators.get("volatility_ratio", 1.0))
        bb_pos = float(indicators.get("BB_Pos", 0.5))
        bb_width = float(indicators.get("bb_width", 0.02))
        atr_pct = float(indicators.get("ATR_pct", 0.003))
        ret_1 = float(row.get("ret_1", 0))
        body_pct = float(row.get("body_pct", 0.3))
        volume_ratio = float(indicators.get("vol_ratio", 1.0))

        # 波动突破信号
        breakout_score = 0.0

        # 波动放大
        vol_expanding = vol_ratio > 1.5
        if not vol_expanding:
            # 无波动放大，返回中性
            return ExpertPrediction(
                expert_name=self.name,
                symbol=symbol,
                direction=1,
                direction_str="CALL",
                raw_probability=0.50,
                calibrated_probability=0.50,
                confidence=0.0,
                features_used=["volatility_ratio", "BB_Pos", "bb_width"],
                model_version=self.model_version,
            )

        # 波动 + 突破方向
        if ret_1 > 0.002 and bb_pos > 0.6:
            breakout_score += 0.3  # 上突破
        elif ret_1 < -0.002 and bb_pos < 0.4:
            breakout_score -= 0.3  # 下突破

        # 大实体确认
        if body_pct > 0.5:
            if ret_1 > 0:
                breakout_score += 0.15
            else:
                breakout_score -= 0.15

        # 放量确认
        if volume_ratio > 1.5:
            if ret_1 > 0:
                breakout_score += 0.1
            else:
                breakout_score -= 0.1

        # 波动强度
        vol_strength = min(1.0, vol_ratio / 3.0)

        # 转换为概率
        raw_prob = 0.50 + breakout_score * vol_strength * 0.4
        raw_prob = max(0.38, min(0.62, raw_prob))

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
            confidence=round(vol_strength, 3),
            features_used=["volatility_ratio", "BB_Pos", "bb_width", "body_pct", "volume_ratio"],
            model_version=self.model_version,
        )