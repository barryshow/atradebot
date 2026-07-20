# -*- coding: utf-8 -*-
"""
OpportunityRanker — 候选交易机会排序器

所有候选机会统一进入排序器，按 Risk Adjusted EV 排序。
每个周期只选择最高质量机会。
相关性惩罚: BTC+ETH+SOL 同向 → 提高 correlation_risk_penalty。
"""
import numpy as np
from typing import Optional, List, Dict
from .models import Opportunity, EdgeResult


class OpportunityRanker:
    """
    机会排序器。

    排序得分 = Risk Adjusted EV
    risk_adjusted_ev = expected_roi * (1 - correlation_penalty) * regime_confidence
    """

    def __init__(self):
        self.correlation_penalty = 0.30  # 同向关联惩罚
        self.max_selected = 3  # 每周期最多选择

    def rank(self, opportunities: List[EdgeResult]) -> List[Opportunity]:
        """
        排序并选择最佳机会。

        Args:
            opportunities: EdgeResult 列表（只包含 passed=True 的）

        Returns:
            Opportunity 列表（按 rank_score 降序，selected 标记）
        """
        if not opportunities:
            return []

        opps = []

        # 检测相关性
        call_symbols = [e.symbol for e in opportunities if e.direction == "CALL"]
        put_symbols = [e.symbol for e in opportunities if e.direction == "PUT"]

        for edge in opportunities:
            # 相关性惩罚
            corr_penalty = 0.0
            if edge.direction == "CALL" and len(call_symbols) > 1:
                corr_penalty = self.correlation_penalty * (len(call_symbols) - 1) / 2
            elif edge.direction == "PUT" and len(put_symbols) > 1:
                corr_penalty = self.correlation_penalty * (len(put_symbols) - 1) / 2

            # Risk Adjusted EV
            risk_adjusted_ev = edge.expected_roi * (1.0 - corr_penalty)

            # 排序得分
            rank_score = risk_adjusted_ev * 100  # 放大到可读范围

            opp = Opportunity(
                symbol=edge.symbol,
                expiry_minutes=edge.expiry_minutes,
                direction=edge.direction,
                direction_int=edge.direction_int,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                uncertainty=edge.uncertainty_margin + edge.calibration_margin,
                regime=edge.regime,
                rank_score=round(rank_score, 4),
                selected=False,
                risk_adjusted_ev=round(risk_adjusted_ev, 4),
            )
            opps.append(opp)

        # 按 rank_score 降序排序
        opps.sort(key=lambda x: x.rank_score, reverse=True)

        # 选择 top N
        selected_count = 0
        selected_symbols = set()
        for opp in opps:
            if selected_count >= self.max_selected:
                break
            # 同品种只选一个方向
            if opp.symbol in selected_symbols:
                continue
            opp.selected = True
            selected_symbols.add(opp.symbol)
            selected_count += 1

        return opps


# 全局单例
_ranker: Optional[OpportunityRanker] = None

def get_opportunity_ranker() -> OpportunityRanker:
    global _ranker
    if _ranker is None:
        _ranker = OpportunityRanker()
    return _ranker