# -*- coding: utf-8 -*-
"""
ModelUncertaintyEstimator — 模型不确定性估算器

三个不确定性来源:
1. Expert Disagreement: 多个 Expert 意见冲突程度
2. Calibration Degradation: 校准质量下降
3. Sample Insufficiency: 某个概率区间样本不足
"""
import numpy as np
from typing import Optional, Dict, List
from .models import ExpertPrediction


class ModelUncertaintyEstimator:
    """
    模型不确定性估算器。

    输出三个 margin:
    - uncertainty_margin: 专家意见冲突
    - calibration_margin: 校准质量下降
    - sample_uncertainty_margin: 样本不足
    """

    def __init__(self):
        self.max_uncertainty_margin = 0.05  # 最大不确定性折扣
        self.max_calibration_margin = 0.03
        self.max_sample_margin = 0.03

    def estimate_expert_uncertainty(self, predictions: List[ExpertPrediction]) -> float:
        """
        专家意见冲突导致的 uncertainty。

        当多个 Expert 意见严重冲突时，增加 uncertainty_penalty。

        Args:
            predictions: 专家预测列表

        Returns:
            uncertainty_margin (概率点)
        """
        if len(predictions) < 2:
            return 0.0

        probs = np.array([p.raw_probability for p in predictions])
        directions = np.array([p.direction for p in predictions])

        # 概率标准差
        prob_std = float(np.std(probs))

        # 方向分歧
        call_count = int(np.sum(directions == 1))
        put_count = int(np.sum(directions == 2))
        dir_disagreement = min(call_count, put_count) / len(directions)  # 0 (all agree) ~ 0.5 (split)

        # 不确定性 = 概率标准差归一化 + 方向分歧
        uncertainty = prob_std * 0.5 + dir_disagreement * 0.5
        return round(min(uncertainty, self.max_uncertainty_margin), 4)

    def estimate_calibration_uncertainty(
        self,
        recent_actual_wr: float,
        recent_predicted_prob: float,
        min_samples: int = 20,
    ) -> float:
        """
        校准质量下降导致的 margin。

        Args:
            recent_actual_wr: 近期实际胜率
            recent_predicted_prob: 近期平均预测概率
            min_samples: 最少样本数

        Returns:
            calibration_margin
        """
        delta = abs(recent_actual_wr - recent_predicted_prob)
        if delta > 0.10:
            return self.max_calibration_margin
        elif delta > 0.05:
            return self.max_calibration_margin * 0.5
        return 0.0

    def estimate_sample_uncertainty(self, sample_count: int, min_expected: int = 50) -> float:
        """
        样本不足导致的 margin。

        Args:
            sample_count: 该概率区间的样本数
            min_expected: 期望最小样本数

        Returns:
            sample_uncertainty_margin
        """
        if sample_count >= min_expected:
            return 0.0
        ratio = sample_count / max(min_expected, 1)
        return round(self.max_sample_margin * (1.0 - ratio), 4)


# 全局单例
_uncertainty_estimator: Optional[ModelUncertaintyEstimator] = None

def get_uncertainty_estimator() -> ModelUncertaintyEstimator:
    global _uncertainty_estimator
    if _uncertainty_estimator is None:
        _uncertainty_estimator = ModelUncertaintyEstimator()
    return _uncertainty_estimator