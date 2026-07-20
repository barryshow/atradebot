# -*- coding: utf-8 -*-
"""
ProbabilityCalibrator — 概率校准器

将模型原始预测概率校准为真实胜率。

核心原则:
1. 使用严格 Walk-Forward 数据进行校准，禁止使用训练集校准
2. 支持 Platt Scaling 和 Isotonic Regression
3. 按概率区间统计 Reliability Diagram
4. 计算 Brier Score 和 Expected Calibration Error (ECE)

用法:
    calibrator = ProbabilityCalibrator(method="isotonic")
    calibrator.fit(raw_probabilities, actual_outcomes)
    calibrated_prob = calibrator.calibrate(raw_prob)
"""
import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


class ProbabilityCalibrator:
    """
    概率校准器。

    方法:
    - "platt": Platt Scaling (Logistic Regression on raw prob)
    - "isotonic": Isotonic Regression (non-parametric)
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._calibrator = None
        self._fitted = False
        self._calibration_stats: Dict = {}

    def fit(self, raw_probabilities: np.ndarray, actual_outcomes: np.ndarray):
        """
        训练校准器。

        Args:
            raw_probabilities: 模型原始预测概率 (shape: n_samples,)
            actual_outcomes: 实际结果 0/1 (shape: n_samples,)
        """
        if len(raw_probabilities) < 10:
            return self

        raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64).reshape(-1, 1)
        actual_outcomes = np.asarray(actual_outcomes, dtype=np.float64)

        if self.method == "platt":
            self._calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self._calibrator.fit(raw_probabilities, actual_outcomes)
        elif self.method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True
            )
            self._calibrator.fit(raw_probabilities.ravel(), actual_outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self._fitted = True
        self._compute_calibration_stats(raw_probabilities.ravel(), actual_outcomes)
        return self

    def calibrate(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """校准概率"""
        if not self._fitted or self._calibrator is None:
            return np.asarray(raw_probabilities)

        raw_probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        if self.method == "platt":
            proba = self._calibrator.predict_proba(raw_probabilities.reshape(-1, 1))
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        elif self.method == "isotonic":
            return self._calibrator.predict(raw_probabilities)
        return raw_probabilities

    def _compute_calibration_stats(self, raw_probs: np.ndarray, outcomes: np.ndarray):
        """计算校准统计"""
        # Brier Score
        brier = np.mean((raw_probs - outcomes) ** 2)

        # ECE (Expected Calibration Error)
        prob_true, prob_pred = calibration_curve(
            outcomes, raw_probs, n_bins=10, strategy="uniform"
        )
        # Recompute bin counts matching the calibration_curve output bins
        bin_edges = np.linspace(0, 1, 11)
        bin_counts = np.histogram(raw_probs, bins=bin_edges)[0]
        # Only use bins that have data in calibration_curve
        n_bins = len(prob_true)
        if n_bins > 0:
            valid_counts = bin_counts[bin_counts > 0][:n_bins]
            if len(valid_counts) == n_bins:
                ece = float(np.sum(
                    np.abs(prob_pred - prob_true) * valid_counts / np.sum(valid_counts)
                ))
            else:
                ece = float(np.mean(np.abs(prob_pred - prob_true)))
        else:
            ece = 0.0

        # Log Loss
        eps = 1e-15
        log_loss = -np.mean(
            outcomes * np.log(np.clip(raw_probs, eps, 1 - eps))
            + (1 - outcomes) * np.log(np.clip(1 - raw_probs, eps, 1 - eps))
        )

        self._calibration_stats = {
            "brier_score": float(brier),
            "ece": float(ece),
            "log_loss": float(log_loss),
            "n_samples": len(raw_probs),
            "n_bins": len(prob_true),
            "calibration_curve": {
                "prob_pred": prob_pred.tolist(),
                "prob_true": prob_true.tolist(),
                "bin_counts": bin_counts.tolist(),
            },
        }

    def get_stats(self) -> Dict:
        return dict(self._calibration_stats)

    def is_fitted(self) -> bool:
        return self._fitted


class ReliabilityDiagram:
    """
    Reliability Diagram — 按概率区间统计预测 vs 实际胜率。

    用于生成 Dashboard 上的校准图表数据。
    """

    BUCKETS = [
        (0.50, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 0.58),
        (0.58, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.00),
    ]

    @classmethod
    def compute(
        cls,
        predicted_probabilities: np.ndarray,
        actual_outcomes: np.ndarray,
    ) -> Dict:
        """
        按概率区间统计。

        Returns:
            {
                "buckets": [
                    {"bucket": "50-52%", "count": N, "predicted_prob": X, "actual_win_rate": Y},
                    ...
                ],
                "overall_brier_score": float,
                "overall_ece": float,
            }
        """
        probs = np.asarray(predicted_probabilities)
        outcomes = np.asarray(actual_outcomes)

        buckets = []
        for low, high in cls.BUCKETS:
            mask = (probs >= low) & (probs < high)
            count = mask.sum()
            if count > 0:
                avg_pred = probs[mask].mean()
                actual_wr = outcomes[mask].mean()
                buckets.append({
                    "bucket": f"{int(low*100)}-{int(high*100)}%",
                    "count": int(count),
                    "predicted_prob": round(float(avg_pred), 4),
                    "actual_win_rate": round(float(actual_wr), 4),
                    "calibration_error": round(float(avg_pred - actual_wr), 4),
                })
            else:
                buckets.append({
                    "bucket": f"{int(low*100)}-{int(high*100)}%",
                    "count": 0,
                    "predicted_prob": 0.0,
                    "actual_win_rate": 0.0,
                    "calibration_error": 0.0,
                })

        # Brier
        brier = float(np.mean((probs - outcomes) ** 2))

        # ECE
        prob_true, prob_pred = calibration_curve(outcomes, probs, n_bins=10, strategy="uniform")
        ece = float(np.mean(np.abs(prob_pred - prob_true))) if len(prob_true) > 0 else 0.0

        return {
            "buckets": buckets,
            "overall_brier_score": round(brier, 4),
            "overall_ece": round(ece, 4),
        }


class WalkForwardCalibrator:
    """
    Walk-Forward 概率校准。

    使用时间序列数据，用历史数据训练校准器，对未来数据进行校准。
    严格防止 Look-Ahead Bias。
    """

    def __init__(self, method: str = "isotonic", min_samples: int = 100):
        self.method = method
        self.min_samples = min_samples
        self._calibrator: Optional[ProbabilityCalibrator] = None
        self._history_probs: List[float] = []
        self._history_outcomes: List[int] = []

    def update(self, raw_prob: float, outcome: Optional[int] = None):
        """
        添加观测并返回校准概率。

        Args:
            raw_prob: 模型原始概率
            outcome: 实际结果 (0/1)，None 表示尚未结算

        Returns:
            calibrated_prob: 校准后概率
        """
        self._history_probs.append(raw_prob)

        if outcome is not None:
            self._history_outcomes.append(outcome)
            # 定期重新拟合
            if len(self._history_outcomes) >= self.min_samples and len(self._history_outcomes) % 20 == 0:
                self._refit()

        return self._calibrate_single(raw_prob)

    def _refit(self):
        """用历史数据重新训练校准器"""
        if len(self._history_outcomes) < self.min_samples:
            return
        # 对齐：只用已结算的样本
        n = min(len(self._history_probs), len(self._history_outcomes))
        if n < self.min_samples:
            return
        probs = np.array(self._history_probs[:n])
        outcomes = np.array(self._history_outcomes[:n])
        self._calibrator = ProbabilityCalibrator(method=self.method)
        self._calibrator.fit(probs, outcomes)

    def _calibrate_single(self, raw_prob: float) -> float:
        if self._calibrator is None or not self._calibrator.is_fitted():
            return raw_prob
        return float(self._calibrator.calibrate(np.array([raw_prob]))[0])

    def is_ready(self) -> bool:
        """Check if calibrator has enough data to produce reliable calibrations."""
        return (
            self._calibrator is not None
            and self._calibrator.is_fitted()
            and len(self._history_outcomes) >= self.min_samples
        )

    def get_status(self) -> dict:
        """Get calibration status for diagnostics."""
        return {
            "ready": self.is_ready(),
            "samples": len(self._history_outcomes),
            "min_required": self.min_samples,
            "method": self.method,
            "stats": self._calibrator.get_stats() if self._calibrator and self._calibrator.is_fitted() else {},
        }

    def get_reliability_diagram(self) -> Dict:
        """获取 Reliability Diagram"""
        n = min(len(self._history_probs), len(self._history_outcomes))
        if n < 10:
            return {"buckets": [], "overall_brier_score": 0, "overall_ece": 0}
        return ReliabilityDiagram.compute(
            np.array(self._history_probs[:n]),
            np.array(self._history_outcomes[:n]),
        )