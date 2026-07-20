# -*- coding: utf-8 -*-
"""
ModelHealthMonitor — 模型自我监控

滚动统计 last 50/100/500 trades，检测模型退化。
触发条件: 实际胜率显著低于预测胜率 + Brier Score 恶化 + ECE 恶化
需要结合置信区间，避免正常随机连败导致频繁停机。
"""
import numpy as np
from typing import Optional, Dict, List, Tuple
from .models import ModelHealthReport
from .probability_calibrator import ReliabilityDiagram


class ModelHealthMonitor:
    """
    模型健康监控。

    监控窗口: 50, 100, 500 trades
    触发 MODEL_DEGRADED 需要多个条件同时满足。
    """

    def __init__(self):
        self.min_health_sample = 50
        self.win_rate_delta_threshold = 0.10    # 实际胜率低于预测 10% 触发
        self.brier_threshold = 0.30             # Brier > 0.30 触发
        self.confidence_interval_z = 1.96       # 95% 置信区间

    def check(
        self,
        trades: List[dict],  # [{"predicted_prob": float, "result": "WIN"/"LOSS", "pnl": float}, ...]
    ) -> ModelHealthReport:
        """
        检查模型健康状态。

        只在样本 >= min_health_sample 时才做判断。
        """
        if len(trades) < self.min_health_sample:
            return ModelHealthReport(
                window=0,
                trade_count=len(trades),
                is_degraded=False,
                degradation_reason="INSUFFICIENT_SAMPLES",
            )

        # 只取最近 min_health_sample 笔
        recent = trades[-self.min_health_sample:]
        probs = np.array([t["predicted_prob"] for t in recent])
        results = np.array([1 if t["result"] == "WIN" else 0 for t in recent])
        pnls = np.array([t["pnl"] for t in recent])

        actual_wr = float(np.mean(results))
        predicted_wr = float(np.mean(probs))
        wr_delta = predicted_wr - actual_wr  # 正值 = 过度自信

        # Brier Score
        brier = float(np.mean((probs - results) ** 2))

        # ECE
        diagram = ReliabilityDiagram.compute(probs, results)
        ece = diagram["overall_ece"]

        # EV
        r = 0.80  # net payout ratio
        ev = np.mean([p * r - (1 - p) for p in probs])

        # PnL
        actual_pnl = float(np.sum(pnls))
        total_staked = len(recent) * 3  # 假设每笔 3U
        roi = actual_pnl / total_staked if total_staked > 0 else 0.0

        # Max Drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl_val in pnls:
            cum_pnl += pnl_val
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

        # 置信区间: 实际胜率是否显著低于预测
        n = len(recent)
        se = np.sqrt(actual_wr * (1 - actual_wr) / n) if n > 0 else 0.0
        ci_lower = actual_wr - self.confidence_interval_z * se
        is_significantly_worse = predicted_wr > ci_lower + self.win_rate_delta_threshold

        # 退化判断
        is_degraded = False
        reasons = []

        if is_significantly_worse and wr_delta > self.win_rate_delta_threshold:
            reasons.append(f"WIN_RATE_DELTA: {wr_delta:.1%}")
        if brier > self.brier_threshold:
            reasons.append(f"BRIER_HIGH: {brier:.3f}")
        if ece > 0.15:
            reasons.append(f"ECE_HIGH: {ece:.3f}")

        if len(reasons) >= 2:
            is_degraded = True

        return ModelHealthReport(
            window=self.min_health_sample,
            trade_count=len(recent),
            actual_win_rate=round(actual_wr, 4),
            predicted_win_rate=round(predicted_wr, 4),
            win_rate_delta=round(wr_delta, 4),
            brier_score=round(brier, 4),
            expected_calibration_error=round(ece, 4),
            ev=round(ev, 4),
            actual_pnl=round(actual_pnl, 4),
            roi=round(roi, 4),
            max_drawdown=round(max_dd, 4),
            is_degraded=is_degraded,
            degradation_reason="; ".join(reasons) if reasons else "",
        )

    def check_multi_window(
        self,
        all_trades: List[dict],
    ) -> Dict[int, ModelHealthReport]:
        """多窗口检查"""
        windows = [50, 100, 500]
        reports = {}
        for w in windows:
            if len(all_trades) >= w:
                reports[w] = self.check(all_trades[-w:])
            else:
                reports[w] = ModelHealthReport(
                    window=w,
                    trade_count=len(all_trades),
                    is_degraded=False,
                    degradation_reason="INSUFFICIENT_SAMPLES",
                )
        return reports


# 全局单例
_monitor: Optional[ModelHealthMonitor] = None

def get_model_health_monitor() -> ModelHealthMonitor:
    global _monitor
    if _monitor is None:
        _monitor = ModelHealthMonitor()
    return _monitor