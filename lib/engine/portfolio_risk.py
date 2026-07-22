# -*- coding: utf-8 -*-
"""
PortfolioRiskManager — 组合风险管理

组合风险控制:
- max_single_trade_risk (单笔最大风险)
- max_total_open_exposure (最大总敞口)
- max_correlated_exposure (最大关联敞口)
- daily_stop (日亏损限制)
- weekly_drawdown_stop (周回撤限制)

Integer Kelly:
- Fractional Kelly + Hard Cap
- 整数约束 (≥3U, step=1)
- 小资金模式处理
"""
import time
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from . import config
from .models import EdgeResult


@dataclass
class PositionSizeResult:
    """仓位计算结果"""
    stake_usd: int = 0
    bet_fraction: float = 0.0
    kelly_fraction: float = 0.0
    target_fraction: float = 0.0
    allowed: bool = False
    reject_reason: str = ""


@dataclass
class PortfolioCheckResult:
    """组合检查结果"""
    allowed: bool = False
    reject_reason: str = ""
    max_open_exposure: float = 0.0
    current_exposure: float = 0.0
    correlated_exposure: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0


class PortfolioRiskManager:
    """
    组合风险 + Integer Kelly 资金管理。

    绝对禁止: Martingale, 输后翻倍, 连续亏损自动加仓
    """

    def __init__(self):
        self.min_order_usd = config.MIN_ORDER_USD
        self.order_step = config.ORDER_AMOUNT_STEP
        self.max_bet_fraction = config.MAX_BET_FRACTION
        self.max_total_exposure = config.MAX_TOTAL_EXPOSURE
        self.max_correlated_exposure = config.MAX_CORRELATED_EXPOSURE
        self.daily_stop = config.DAILY_STOP
        self.weekly_drawdown_stop = config.WEEKLY_DRAWDOWN_STOP
        self.kelly_fraction = config.KELLY_FRACTION
        self.small_account_max_bet_fraction = config.SMALL_ACCOUNT_MAX_BET_FRACTION

        # 追踪
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.open_positions: Dict[str, List[dict]] = {}  # symbol → [position_info]
        self.daily_reset_time: float = time.time()
        self.weekly_reset_time: float = time.time()

    def compute_position_size(
        self,
        edge_result: EdgeResult,
        equity: float,
    ) -> PositionSizeResult:
        """
        Integer Kelly 仓位计算。

        流程:
        kelly_fraction → target_fraction → raw_stake → integer_stake → min_order_check

        Returns:
            PositionSizeResult
        """
        if equity <= 0:
            return PositionSizeResult(reject_reason="ACCOUNT_TOO_SMALL_FOR_RISK_RULE")

        # ── 1. 理论 Kelly ──
        # kelly = (p * (1 + r) - 1) / r
        # 其中 r = net_payout_ratio
        r = edge_result.net_payout_ratio
        p = edge_result.conservative_probability
        kelly = max(0.0, (p * (1.0 + r) - 1.0) / r) if r > 0 else 0.0

        # ── 2. Fractional Kelly ──
        target_fraction = self.kelly_fraction * kelly

        # ── 3. Hard Cap ──
        effective_fraction = min(target_fraction, self.max_bet_fraction)

        # ── 4. Small Account Check ──
        min_possible_fraction = self.min_order_usd / equity
        if effective_fraction < min_possible_fraction:
            # 小账户：Kelly 理论下注比例低于最低下注额
            # ⚠️ 禁止自动突破风险上限: 如果 min_possible_fraction > MAX_BET_FRACTION
            #    直接拒绝，不为满足 3U 最低下注而突破风险上限
            if min_possible_fraction > self.max_bet_fraction:
                return PositionSizeResult(
                    stake_usd=0,
                    bet_fraction=round(min_possible_fraction, 4),
                    kelly_fraction=round(kelly, 4),
                    target_fraction=round(target_fraction, 4),
                    allowed=False,
                    reject_reason="ACCOUNT_TOO_SMALL_FOR_RISK_RULE",
                )
            if equity >= self.min_order_usd:
                effective_fraction = min_possible_fraction
            else:
                return PositionSizeResult(
                    stake_usd=0,
                    bet_fraction=round(min_possible_fraction, 4),
                    kelly_fraction=round(kelly, 4),
                    target_fraction=round(target_fraction, 4),
                    allowed=False,
                    reject_reason="ACCOUNT_TOO_SMALL_FOR_RISK_RULE",
                )

        # ── 5. 转换为 USDT ──
        raw_stake = equity * effective_fraction

        # ── 6. Integer Constraint ──
        integer_stake = int(math.floor(raw_stake))
        integer_stake = (integer_stake // self.order_step) * self.order_step

        # 小账户：如果 min_possible_fraction 刚好是有效下注比例（即 Kelly 为 0 时
        # 只用最低 3U 下注），浮点误差可能导致 raw_stake=2.9999 → floor=2 → 被拒。
        # 此时直接取 min_order_usd。
        if integer_stake < self.min_order_usd and raw_stake >= self.min_order_usd - 0.01:
            integer_stake = self.min_order_usd

        # ── 7. Minimum Order Check ──
        if integer_stake < self.min_order_usd:
            return PositionSizeResult(
                stake_usd=integer_stake,
                bet_fraction=round(effective_fraction, 4),
                kelly_fraction=round(kelly, 4),
                target_fraction=round(target_fraction, 4),
                allowed=False,
                reject_reason="ACCOUNT_TOO_SMALL_FOR_RISK_RULE",
            )

        return PositionSizeResult(
            stake_usd=integer_stake,
            bet_fraction=round(effective_fraction, 4),
            kelly_fraction=round(kelly, 4),
            target_fraction=round(target_fraction, 4),
            allowed=True,
            reject_reason="",
        )

    def check_portfolio_limits(
        self,
        symbol: str,
        direction: str,
        stake_usd: int,
        equity: float,
        active_positions: Optional[Dict] = None,
    ) -> PortfolioCheckResult:
        """
        组合限制检查。

        Returns:
            PortfolioCheckResult
        """
        # ── 日亏损检查 ──
        if equity > 0 and self.daily_pnl < -self.daily_stop * equity:
            return PortfolioCheckResult(
                allowed=False,
                reject_reason="DAILY_STOP",
                daily_pnl=self.daily_pnl,
            )

        # ── 周回撤检查 ──
        if equity > 0 and self.weekly_pnl < -self.weekly_drawdown_stop * equity:
            return PortfolioCheckResult(
                allowed=False,
                reject_reason="WEEKLY_DRAWDOWN",
                weekly_pnl=self.weekly_pnl,
            )

        # ── 总敞口检查 ──
        current_exposure = sum(
            sum(p.get("stake_usd", 0) for p in positions)
            for positions in (active_positions or {}).values()
        )
        new_exposure = current_exposure + stake_usd
        # 小账户：如果最低下单额已超过理论敞口上限，放宽到至少 min_order_usd
        effective_max_exposure = max(
            equity * self.max_total_exposure,
            self.min_order_usd * 1.0,  # 允许至少一单
        )
        if equity > 0 and new_exposure > effective_max_exposure:
            return PortfolioCheckResult(
                allowed=False,
                reject_reason="PORTFOLIO_LIMIT",
                current_exposure=current_exposure,
                max_open_exposure=effective_max_exposure,
            )

        # ── 关联敞口检查 ──
        correlated_exposure = sum(
            sum(p.get("stake_usd", 0) for p in positions if p.get("direction") == direction)
            for positions in (active_positions or {}).values()
        )
        # 小账户：放宽到至少允许一单
        effective_max_correlated = max(
            equity * self.max_correlated_exposure,
            self.min_order_usd * 1.0,
        )
        if equity > 0 and correlated_exposure + stake_usd > effective_max_correlated:
            return PortfolioCheckResult(
                allowed=False,
                reject_reason="CORRELATION_LIMIT",
                correlated_exposure=correlated_exposure,
            )

        return PortfolioCheckResult(
            allowed=True,
            current_exposure=current_exposure,
            correlated_exposure=correlated_exposure,
            daily_pnl=self.daily_pnl,
            weekly_pnl=self.weekly_pnl,
        )

    def record_pnl(self, pnl: float):
        """记录盈亏"""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        # 日重置
        now = time.time()
        if now - self.daily_reset_time > 86400:
            self.daily_pnl = pnl
            self.daily_reset_time = now

        # 周重置
        if now - self.weekly_reset_time > 604800:
            self.weekly_pnl = pnl
            self.weekly_reset_time = now

    def get_minimum_possible_risk(self, equity: float) -> float:
        """计算当前账户的最小可能风险比例"""
        return self.min_order_usd / equity if equity > 0 else float("inf")

    def check_account_size(self, equity: float) -> Tuple[bool, str]:
        """
        检查账户大小是否满足基本风险规则。

        Returns:
            (can_trade, message)
        """
        min_risk = self.get_minimum_possible_risk(equity)
        if min_risk > self.max_bet_fraction:
            if self.small_account_max_bet_fraction > 0 and min_risk <= self.small_account_max_bet_fraction:
                return True, f"SMALL_ACCOUNT_MODE: min_risk={min_risk:.1%}, using small_account_max={self.small_account_max_bet_fraction:.1%}"
            return False, f"ACCOUNT_TOO_SMALL: min_order={self.min_order_usd}U, equity={equity:.0f}U, min_possible_risk={min_risk:.1%} > max_bet_fraction={self.max_bet_fraction:.1%}"
        return True, "OK"


# 全局单例
_portfolio_risk: Optional[PortfolioRiskManager] = None

def get_portfolio_risk() -> PortfolioRiskManager:
    global _portfolio_risk
    if _portfolio_risk is None:
        _portfolio_risk = PortfolioRiskManager()
    return _portfolio_risk