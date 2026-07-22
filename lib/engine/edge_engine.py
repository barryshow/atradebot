# -*- coding: utf-8 -*-
"""
EdgeEngine — 动态盈亏平衡 + Edge 计算引擎

核心公式:
    p_be = 1 / (1 + net_payout_ratio)
    raw_edge = calibrated_probability - break_even_probability
    conservative_probability = calibrated_probability - uncertainty_margin - calibration_margin - model_degradation_margin
    expected_roi = conservative_probability * net_payout_ratio - (1 - conservative_probability)
    effective_edge = conservative_probability - break_even_probability

两层计算:
    Layer 1: conservative_probability = calibrated_prob - all margins
    Layer 2: expected_roi = p * r - (1 - p)

只有 expected_roi > MIN_EXPECTED_ROI 且 effective_edge > MIN_EFFECTIVE_EDGE 才进入候选池。
"""
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from . import config
from .models import EdgeResult


@dataclass
class PayoutInfo:
    """动态赔付率信息"""
    symbol: str = ""
    expiry_minutes: int = 15
    net_payout_ratio: float = 0.0          # 净盈利比例
    total_payout_ratio: float = 0.0        # 总返还比例 (含本金)
    source: str = ""                       # "api" / "hardcoded" / "estimated"
    updated_at: float = 0.0                # 最后更新时间


class ContractDiscovery:
    """
    合约发现器 — 管理 HIBT 可用期限和赔付率。

    当前状态: HIBT API 不支持动态获取赔付率，使用硬编码值。
    但在架构上预留了 API 接口。
    """

    def __init__(self):
        # 硬编码的赔付率（当前唯一可用来源）
        self._hardcoded_payouts: Dict[str, Dict[str, float]] = {
            "BTCUSDT": {"net": 0.818, "total": 1.818},
            "ETHUSDT": {"net": 0.80, "total": 1.80},
            "SOLUSDT": {"net": 0.80, "total": 1.80},
        }
        self._available_expiries: list = config.AVAILABLE_EXPIRIES
        self._expiry_discovery_available: bool = not config.EXPIRY_DISCOVERY_UNAVAILABLE

    def get_available_expiries(self) -> list:
        """获取可用期限列表"""
        if not self._expiry_discovery_available:
            print(f"[ContractDiscovery] ⚠ EXPIRY_DISCOVERY_UNAVAILABLE — "
                  f"using configured expiries: {self._available_expiries}", flush=True)
        return self._available_expiries

    def get_payout(self, symbol: str, expiry_minutes: int = 15) -> PayoutInfo:
        """
        获取赔付率。

        Returns:
            PayoutInfo with net_payout_ratio and source
        """
        payouts = self._hardcoded_payouts.get(symbol, {"net": 0.80, "total": 1.80})
        return PayoutInfo(
            symbol=symbol,
            expiry_minutes=expiry_minutes,
            net_payout_ratio=payouts["net"],
            total_payout_ratio=payouts["total"],
            source="hardcoded",
            updated_at=0.0,
        )

    def update_payout_from_api(self, symbol: str, net_payout: float):
        """从 API 更新赔付率（未来可用）"""
        self._hardcoded_payouts[symbol] = {"net": net_payout, "total": 1.0 + net_payout}


class EdgeEngine:
    """
    Edge 计算引擎。

    输入: 校准概率 + 赔付率 + 各种 margin
    输出: EdgeResult (含 effective_edge, expected_roi, 是否通过)
    """

    def __init__(self):
        self.min_effective_edge = config.MIN_EFFECTIVE_EDGE
        self.min_expected_roi = config.MIN_EXPECTED_ROI
        self.default_uncertainty_margin = config.DEFAULT_UNCERTAINTY_MARGIN
        self.default_calibration_margin = config.DEFAULT_CALIBRATION_MARGIN
        self.default_degradation_margin = config.DEFAULT_DEGRADATION_MARGIN
        self.contract_discovery = ContractDiscovery()

    def compute(
        self,
        symbol: str,
        calibrated_probability: float,
        direction: str,
        direction_int: int,
        expiry_minutes: int = 15,
        entry_price: float = 0.0,
        uncertainty_margin: Optional[float] = None,
        calibration_margin: Optional[float] = None,
        model_degradation_margin: Optional[float] = None,
        sample_uncertainty_margin: Optional[float] = None,
        regime: str = "",
        expert_votes: Optional[Dict] = None,
    ) -> EdgeResult:
        """
        计算 Edge。

        Returns:
            EdgeResult with passed=True/False and reject_reason
        """
        # 获取赔付率
        payout = self.contract_discovery.get_payout(symbol, expiry_minutes)
        net_payout_ratio = payout.net_payout_ratio
        payout_source = payout.source  # "hardcoded" = CONFIG_ASSUMED, not verified
        payout_verified = (payout_source == "api")  # 只有API返回才是verified
        payout_flag = "CONFIG_ASSUMED" if payout_source in ("hardcoded", "estimated") else "VERIFIED_HIBT"

        # 使用默认值或传入值（None 表示未传，用默认值）
        unc_margin = uncertainty_margin if uncertainty_margin is not None else self.default_uncertainty_margin
        cal_margin = calibration_margin if calibration_margin is not None else self.default_calibration_margin
        deg_margin = model_degradation_margin if model_degradation_margin is not None else self.default_degradation_margin
        sample_margin = sample_uncertainty_margin if sample_uncertainty_margin is not None else 0.0

        # ── Layer 1: Conservative Probability ──
        total_margin = unc_margin + cal_margin + deg_margin + sample_margin
        conservative_prob = max(0.0, min(1.0, calibrated_probability - total_margin))

        # ── Layer 2: Expected ROI ──
        # 每投入 1U 的期望 ROI
        expected_roi = conservative_prob * net_payout_ratio - (1.0 - conservative_prob)

        # ── Break-even ──
        break_even_prob = 1.0 / (1.0 + net_payout_ratio)

        # ── Edge ──
        raw_edge = calibrated_probability - break_even_prob
        effective_edge = conservative_prob - break_even_prob
        probability_edge = calibrated_probability - break_even_prob

        # ── 判定 ──
        passed = True
        reject_reason = ""

        if effective_edge <= self.min_effective_edge:
            passed = False
            reject_reason = "LOW_EDGE" if effective_edge > 0 else "NO_EDGE"
        elif expected_roi <= self.min_expected_roi:
            passed = False
            reject_reason = "LOW_EDGE"  # Edge 存在但 ROI 不足

        return EdgeResult(
            symbol=symbol,
            expiry_minutes=expiry_minutes,
            direction=direction,
            direction_int=direction_int,
            entry_price=entry_price,
            calibrated_probability=round(calibrated_probability, 4),
            conservative_probability=round(conservative_prob, 4),
            payout_ratio=round(payout.total_payout_ratio, 4),
            net_payout_ratio=round(net_payout_ratio, 4),
            payout_source=payout.source,
            payout_verified=payout_verified,
            payout_flag=payout_flag,
            break_even_probability=round(break_even_prob, 4),
            probability_edge=round(probability_edge, 4),
            raw_edge=round(raw_edge, 4),
            effective_edge=round(effective_edge, 4),
            expected_roi=round(expected_roi, 4),
            edge_flag="SIMULATED_EDGE" if payout_flag == "CONFIG_ASSUMED" else "VERIFIED_EDGE",
            uncertainty_margin=round(unc_margin, 4),
            calibration_margin=round(cal_margin, 4),
            model_degradation_margin=round(deg_margin, 4),
            sample_uncertainty_margin=round(sample_margin, 4),
            passed=passed,
            reject_reason=reject_reason,
            regime=regime,
            expert_votes=expert_votes or {},
        )

    def compute_break_even(self, net_payout_ratio: float) -> float:
        """计算盈亏平衡概率"""
        return 1.0 / (1.0 + net_payout_ratio)


# 全局单例
_edge_engine: Optional[EdgeEngine] = None

def get_edge_engine() -> EdgeEngine:
    global _edge_engine
    if _edge_engine is None:
        _edge_engine = EdgeEngine()
    return _edge_engine