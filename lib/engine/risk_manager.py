# -*- coding: utf-8 -*-
"""
RiskManager — 风险管理器（流水线 L3~L5）

L3: 共振分门槛 (≥0.65)
L4: 双重冷却 (Reject Cooldown 60s + Settlement Cooldown 60s)
L5: 持仓检查（开仓/反手/跳过，无加仓）

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
# L5: 持仓检查（简化版—去掉加仓）
# ──────────────────────────────────────────────

def _check_position_management(
    signal: TradeSignal,
    position: PositionState | None,
) -> tuple[GateResult, str]:
    """
    同品种持仓状态机（简化版）：

    HIBT 二元期权每单独立结算，持仓期间不应"加仓"。
    每笔交易独立决策，等结算后再重新评估。

    状态1: 无持仓 → action="open"（正常开底仓）
    状态2: 有持仓 + 同向信号 → 跳过（等结算，不追仓）
    状态3: 有持仓 + 反向信号 → action="close_and_open"（标记平仓，结算后自动开反向仓）
    状态4: pending_close → 跳过（等结算完成）
    """
    if position is None:
        return (
            GateResult(level=5, name="Position Mgmt", passed=True,
                       reason="no position → open"),
            "open",
        )

    # pending_close — 等结算
    if position.pending_close:
        return (
            GateResult(level=5, name="Position Mgmt", passed=False,
                       reason="pending close, wait settlement"),
            "pending",
        )

    # 有持仓 — 检查方向
    is_same_direction = position.direction == signal.direction

    if is_same_direction:
        # 同向：跳过，等结算后再决策（不再加仓）
        return (
            GateResult(level=5, name="Position Mgmt", passed=False,
                       reason="same direction, skip — wait settlement (no add)"),
            "pending",
        )
    else:
        # 反向：平仓反手
        return (
            GateResult(level=5, name="Position Mgmt", passed=True,
                       reason="reverse signal → close existing + open new"),
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
    g5, action = _check_position_management(signal, existing_position)
    signal.action = action
    gates.append(g5)
    if not g5.passed:
        return None, gates

    return signal, gates
