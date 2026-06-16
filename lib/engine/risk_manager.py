# -*- coding: utf-8 -*-
"""
RiskManager — 风险管理器（流水线 L3~L5）

L3: 共振分门槛 (≥0.65)
L4: 双重冷却 (Reject Cooldown 60s + Settlement Cooldown 60s)
L5: 持仓检查 + 加仓/反手逻辑

输入: TradeSignal (已通过 SignalValidator)
输出: TradeSignal | None (None 表示拦截)
"""
import time
from . import config
from .models import TradeSignal, GateResult, PositionState


# ──────────────────────────────────────────────
# L3: 共振分
# ──────────────────────────────────────────────

def _confluence_score(direction: int, indicators: dict) -> float:
    """多指标共振打分 0.0~1.0（与 risk_gates.py 一致）"""
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


def _check_confluence(signal: TradeSignal) -> GateResult:
    """共振分 < 0.65 拦截"""
    score = _confluence_score(signal.direction, signal.indicators)
    signal.confluence = score
    passed = score >= config.CONFLUENCE_MIN
    return GateResult(
        level=3,
        name="Confluence",
        passed=passed,
        reason=f"confluence={score:.2f} {'>=' if passed else '<'} {config.CONFLUENCE_MIN}",
    )


# ──────────────────────────────────────────────
# L4: 双重冷却
# ──────────────────────────────────────────────

def _check_cooldown(
    symbol: str,
    current_ts: int,
    last_reject_ts: dict[str, int],
    last_settlement_ts: dict[str, int],
) -> GateResult:
    """
    - Reject Cooldown: 被拒后 60 秒不重复计算
    - Settlement Cooldown: 结算完成后额外 60 秒不开新单
    """
    reject_elapsed = current_ts - last_reject_ts.get(symbol, 0)
    if reject_elapsed < config.REJECT_COOLDOWN_SEC * 1000:
        return GateResult(
            level=4, name="Reject Cooldown",
            passed=False,
            reason=f"reject cooldown {reject_elapsed//1000}s < {config.REJECT_COOLDOWN_SEC}s",
        )

    settle_elapsed = current_ts - last_settlement_ts.get(symbol, 0)
    if settle_elapsed < config.SETTLEMENT_COOLDOWN_SEC * 1000:
        return GateResult(
            level=4, name="Settlement Cooldown",
            passed=False,
            reason=f"settlement cooldown {settle_elapsed//1000}s < {config.SETTLEMENT_COOLDOWN_SEC}s",
        )

    return GateResult(level=4, name="Cooldown", passed=True, reason="")


# ──────────────────────────────────────────────
# L5: 持仓检查 + 加仓/反手
# ──────────────────────────────────────────────

def _check_position_management(
    signal: TradeSignal,
    position: PositionState | None,
    current_price: float,
) -> tuple[GateResult, str]:
    """
    同品种持仓状态机:

    状态1: 无持仓 → action="open"（正常开底仓）
    状态2: 同向信号 + 浮盈 → action="add"（加仓）
    状态3: 同向信号 + 亏损 → 拦截 (No Martingale)
    状态4: 反向信号 → action="close_and_open"（平仓→重置→独立评估）
    状态5: pending_close 状态 → 拦截（等结算完成）
    """
    if position is None:
        # 无持仓：正常开底仓
        return (
            GateResult(level=5, name="Position Mgmt", passed=True, reason="no position → open"),
            "open",
        )

    # 有持仓但 pending_close — 等待结算完成
    if position.pending_close:
        return (
            GateResult(level=5, name="Position Mgmt", passed=False,
                       reason="pending close, wait settlement"),
            "pending",
        )

    is_same_direction = position.direction == signal.direction

    if is_same_direction:
        # 同向信号
        # 计算未实现盈亏
        if signal.direction == 1:
            roi = (current_price - position.entry_price) / position.entry_price
        else:
            roi = (position.entry_price - current_price) / position.entry_price

        if roi >= config.ADD_POSITION_MIN_ROI:
            return (
                GateResult(level=5, name="Position Mgmt", passed=True,
                           reason=f"add position (roi={roi:.4f})"),
                "add",
            )
        else:
            return (
                GateResult(level=5, name="Position Mgmt", passed=False,
                           reason=f"No Martingale - position in loss (roi={roi:.4f})"),
                "pending",
            )
    else:
        # 反向信号 → 触发平仓反手
        return (
            GateResult(level=5, name="Position Mgmt", passed=True,
                       reason=f"reverse signal → close existing + open new"),
            "close_and_open",
        )


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def manage_signal(
    signal: TradeSignal,
    current_ts: int,
    last_reject_ts: dict[str, int],
    last_settlement_ts: dict[str, int],
    existing_position: PositionState | None,
    current_price: float,
) -> tuple[TradeSignal | None, list[GateResult]]:
    """
    通过 L3~L5 风控管道管理信号。

    返回:
        (TradeSignal | None, [GateResult, ...])
        None 表示拦截
    """
    gates: list[GateResult] = []

    # ── L3: 共振分门槛 ──
    g3 = _check_confluence(signal)
    gates.append(g3)
    if not g3.passed:
        return None, gates

    # ── L4: 双重冷却 ──
    g4 = _check_cooldown(signal.symbol, current_ts, last_reject_ts, last_settlement_ts)
    gates.append(g4)
    if not g4.passed:
        return None, gates

    # ── L5: 持仓管理 ──
    g5, action = _check_position_management(signal, existing_position, current_price)
    signal.action = action
    gates.append(g5)
    if not g5.passed:
        return None, gates

    return signal, gates
