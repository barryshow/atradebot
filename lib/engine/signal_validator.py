# -*- coding: utf-8 -*-
"""
SignalValidator — 信号验证器（流水线 L0~L2）

L0: Anti-Knife Filter (防接刀) — 最高优先级
L1: 硬性概率门槛 (≥62%)
L2: 极值翻转 + 概率重置 (Reversal Probability Override)

输入: 原始模型预测 Prediction + 指标数据
输出: TradeSignal | None (None 表示拦截)
"""
import numpy as np
from . import config
from .models import Prediction, TradeSignal, GateResult, PositionState


# ──────────────────────────────────────────────
# L0: 防接刀过滤（Anti-Knife Filter）
# ──────────────────────────────────────────────

def _check_anti_knife(direction: int, row: dict) -> GateResult:
    """
    检测剧烈单边行情，禁止逆势开仓。

    检测逻辑:
    - 取最近 N 根 K 线（由 feature_data 最后几行推算），判断连续大实体同向运动
    - body_ratio > 0.6（大实体）+ 同色连续 3 根 = 猛烈单边
    - CCI > 100 或 < -100 辅助确认极端

    返回:
        GateResult(passed=False, reason="Violent Momentum - Reject Counter-Trend")
        或 GateResult(passed=True)
    """
    body_ratio = float(row.get("body_pct", 0.5))
    cci = float(row.get("CCI", 0))
    is_green = int(row.get("is_green", 1))
    natr = float(row.get("NATR", 0.01))

    # NATR 极端波动标记（ATR 超过均值 2 倍以上）
    vol_ratio = float(row.get("volatility_ratio", 1.0))
    extreme_vol = vol_ratio > 2.0

    # 判断当前 K 线是否为大实体
    big_body = body_ratio >= config.ANTI_KNIFE_BODY_RATIO

    # 需要更多 K 线数据做连续判断，但当前 row 是最后一根聚合K线
    # 我们结合 CCI + body + vol 做保守估计
    # CCI > 100 且大实体阳线 → 猛烈上涨
    # CCI < -100 且大实体阴线 → 猛烈下跌

    if direction == 1:  # 想做多（CALL）
        # 但市场正在暴跌 → 接刀风险
        falling_hard = cci < -config.ANTI_KNIFE_CCI and big_body and is_green == 0
        # 或者 vol 异常 + CCI 极低
        falling_vol = cci < -config.ANTI_KNIFE_CCI * 0.8 and extreme_vol and is_green == 0
        if falling_hard or falling_vol:
            return GateResult(
                level=0, name="Anti-Knife",
                passed=False,
                reason=f"Violent Momentum - Reject Counter-Trend (CCI={cci:.0f}, body={body_ratio:.2f})",
            )

    elif direction == 2:  # 想做空（PUT）
        # 但市场正在暴涨 → 接刀风险
        surging_hard = cci > config.ANTI_KNIFE_CCI and big_body and is_green == 1
        surging_vol = cci > config.ANTI_KNIFE_CCI * 0.8 and extreme_vol and is_green == 1
        if surging_hard or surging_vol:
            return GateResult(
                level=0, name="Anti-Knife",
                passed=False,
                reason=f"Violent Momentum - Reject Counter-Trend (CCI={cci:.0f}, body={body_ratio:.2f})",
            )

    return GateResult(level=0, name="Anti-Knife", passed=True, reason="")


# ──────────────────────────────────────────────
# L1: 硬性概率门槛
# ──────────────────────────────────────────────

def _check_hard_threshold(pred: Prediction) -> GateResult:
    """模型原始预测 < 62% 直接拦截"""
    passed = pred.prob_win >= config.HARD_PROB_THRESHOLD
    return GateResult(
        level=1,
        name="Hard Threshold",
        passed=passed,
        reason=f"prob_win={pred.prob_win:.3f} {'>=' if passed else '<'} {config.HARD_PROB_THRESHOLD}",
    )


# ──────────────────────────────────────────────
# L2: 极值翻转 + 概率重置
# ──────────────────────────────────────────────

def _check_extreme_reversal(
    pred: Prediction, bb_pos: float, indicators: dict,
) -> tuple[GateResult, Prediction]:
    """
    保留布林带极值翻转策略，但剥离原模型假概率。

    - 做多 + BB_Pos > 0.72 → 翻转为做空, prob=0.55, is_reversal=True
    - 做空 + BB_Pos < 0.28 → 翻转为做多, prob=0.55, is_reversal=True
    - L0 防接刀已优先拦截暴跌中抄底、暴涨中摸顶，所以到这里的是"温和"极值
    """
    should_flip_long_to_short = pred.direction == 1 and bb_pos > config.BB_EXTREME_HIGH
    should_flip_short_to_long = pred.direction == 2 and bb_pos < config.BB_EXTREME_LOW

    if not (should_flip_long_to_short or should_flip_short_to_long):
        return (
            GateResult(level=2, name="Reversal", passed=True, reason=""),
            pred,
        )

    new_dir = 2 if should_flip_long_to_short else 1
    reversed_pred = Prediction(
        symbol=pred.symbol,
        prob_long=1.0 - pred.prob_long,
        direction=new_dir,
        prob_win=config.REVERSAL_PROB,   # 强制重置为 0.55
        flipped=True,
    )

    old_dir_str = "做多" if pred.direction == 1 else "做空"
    new_dir_str = "做空" if new_dir == 2 else "做多"
    reason = (
        f"Reversal: {old_dir_str}→{new_dir_str} @BB={bb_pos:.3f}, "
        f"prob reset {pred.prob_win:.3f}→{config.REVERSAL_PROB}"
    )

    return (
        GateResult(level=2, name="Reversal", passed=True, reason=reason),
        reversed_pred,
    )


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def validate_signal(
    symbol: str,
    pred: Prediction,
    row: dict,
    indicators: dict,
    current_price: float,
    existing_position: PositionState | None,
) -> tuple[TradeSignal | None, list[GateResult]]:
    """
    三阶段信号验证。

    返回:
        (TradeSignal | None, [GateResult, ...])
        None 表示信号被拦截，GateResult 列表记录每道门的结果
    """
    gates: list[GateResult] = []
    bb_pos = indicators.get("BB_Pos", 0.5)

    # ── L0: 防接刀 ──
    g0 = _check_anti_knife(pred.direction, row)
    gates.append(g0)
    if not g0.passed:
        return None, gates

    # ── L1: 硬性概率门槛 ──
    g1 = _check_hard_threshold(pred)
    gates.append(g1)
    if not g1.passed:
        return None, gates

    # ── L2: 极值翻转 + 概率重置 ──
    g2, final_pred = _check_extreme_reversal(pred, bb_pos, indicators)
    gates.append(g2)

    # 构建 TradeSignal
    dir_str = "做多(CALL)" if final_pred.direction == 1 else "做空(PUT)"

    # L5 的判断交给 RiskManager，此处只确定方向
    # action 由 RiskManager 决定，此处占位 "pending"
    signal = TradeSignal(
        symbol=symbol,
        direction=final_pred.direction,
        dir_str=dir_str,
        prob_win=final_pred.prob_win,
        original_prob=pred.prob_win,
        is_reversal=final_pred.flipped,
        action="pending",
        entry_price=current_price,
        indicators=indicators,
        confluence=0.0,        # RiskManager 会填
        row=row,
        flip_note="🔄 [概率重置] " if final_pred.flipped else "",
    )

    return signal, gates
