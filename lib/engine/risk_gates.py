# -*- coding: utf-8 -*-
"""
简化风控管道 - 激进模式
  L1: ML概率门槛 (放宽至0.30)
  L2: 极值翻转 (保留, 高盈亏比信号)
  L3: AI仅做信息展示 (不否决)
"""
from . import config
from .models import Prediction, RiskGateResult


def check_ml_probability(pred: Prediction) -> RiskGateResult:
    """Level 1: 宽松的最小胜率门槛"""
    return RiskGateResult(
        level=1,
        name="ML Probability",
        passed=pred.prob_win >= config.MIN_PROBABILITY,
        reason=f"Win prob {pred.prob_win:.3f} {'>=' if pred.prob_win >= config.MIN_PROBABILITY else '<'} {config.MIN_PROBABILITY}",
        details={"prob_win": pred.prob_win, "threshold": config.MIN_PROBABILITY},
    )


def check_extreme_reversal_flipper(pred: Prediction, bb_pos: float) -> Prediction:
    """
    Level 2: '神之手'极值翻转。
    ML说做多但价格在布林上轨(>0.70) → 翻转为做空
    ML说做空但价格在布林下轨(<0.30) → 翻转为做多
    避免追涨杀跌假突破，这是系统中唯一保留的物理门
    """
    should_flip_long_to_short = pred.direction == 1 and bb_pos > config.BB_EXTREME_HIGH
    should_flip_short_to_long = pred.direction == 2 and bb_pos < config.BB_EXTREME_LOW

    if should_flip_long_to_short or should_flip_short_to_long:
        new_dir = 2 if should_flip_long_to_short else 1
        return Prediction(
            symbol=pred.symbol,
            prob_long=1.0 - pred.prob_long,
            direction=new_dir,
            prob_win=pred.prob_win,
            flipped=True,
        )
    return pred


def check_confluence(direction: int, indicators: dict) -> RiskGateResult:
    """Level 2b: 多指标共振（仅供参考，不否决）"""
    score = confluence_score(direction, indicators)
    return RiskGateResult(
        level=2,
        name="Confluence",
        passed=True,  # 永远通过，只做展示
        reason=f"共振分={score:.2f}",
        details={"confluence_score": score, "threshold": config.CONFLUENCE_MIN},
    )


def confluence_score(direction: int, indicators: dict) -> float:
    """多指标共振打分 0.0~1.0"""
    score = 0.0
    adx = indicators.get("ADX", 0)
    rsi = indicators.get("RSI", 50)
    bsp5 = indicators.get("BSP_5", 0)
    macd_change = indicators.get("macd_hist_change", 0)
    bb_pos = indicators.get("BB_Pos", 0.5)

    if adx > 25:
        if direction == 1 and bb_pos < 0.5:
            score += 0.25
        elif direction == 2 and bb_pos > 0.5:
            score += 0.25

    if direction == 1 and 25 <= rsi <= 55:
        score += 0.20
    elif direction == 2 and 45 <= rsi <= 75:
        score += 0.20

    if (direction == 1 and bsp5 > 0.05) or (direction == 2 and bsp5 < -0.05):
        score += 0.25

    if (direction == 1 and macd_change > 0) or (direction == 2 and macd_change < 0):
        score += 0.15

    if (direction == 1 and bb_pos < 0.38) or (direction == 2 and bb_pos > 0.62):
        score += 0.15

    return min(score, 1.0)


def run_risk_pipeline(pred: Prediction, row, indicators: dict) -> tuple[list[RiskGateResult], Prediction, float]:
    """
    激进模式管道:
      L1: ML概率门 (放宽)
      L2a: 极值翻转 (高盈亏比信号)
      L2b: 共振参考 (不否决)
      不再检查: BB死区, ADX震荡, ADX极值
    """
    gates = []

    # L1: ML概率 (门槛已放宽到0.30)
    g1 = check_ml_probability(pred)
    gates.append(g1)
    if not g1.passed:
        return gates, pred, 0.0

    # L2a: 极值翻转
    pred = check_extreme_reversal_flipper(pred, indicators.get("BB_Pos", 0.5))

    # L2b: 共振分（只展示不否决）
    c_score = confluence_score(pred.direction, indicators)
    g_conf = check_confluence(pred.direction, indicators)
    gates.append(g_conf)

    return gates, pred, c_score
