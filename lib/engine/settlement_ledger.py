# -*- coding: utf-8 -*-
"""
SettlementLedger — 结算账本

与 TradeLedger 配合使用：
- TradeLedger 记录所有交易决策（下单时写入）
- SettlementLedger 记录结算结果（结算后更新）

职责：
1. 管理结算状态机：PENDING → CONFIRMED / ESTIMATED / DISPUTED
2. 记录结算价格、盈亏
3. 支持余额核对
4. 提供结算统计

存储: 与 TradeLedger 共享同一个 JSONL 文件（通过在 TradeLedger 中更新字段实现）
"""
import json
import os
import time
from typing import Optional
from dataclasses import dataclass, field, asdict
from .trade_ledger import TradeLedger, TradeRecord, get_trade_ledger


# ── 结算状态常量 ──
class SettlementStatus:
    PENDING = "PENDING"            # 等待结算
    CONFIRMED = "CONFIRMED"        # HIBT API 确认结算
    ESTIMATED = "ESTIMATED"        # 余额推算（无 API 确认）
    DISPUTED = "DISPUTED"          # 余额推算与预期不符
    REJECTED = "REJECTED"          # 订单被拒
    EXPIRED = "EXPIRED"            # 超时未确认


# ── 拒绝原因常量 ──
class RejectReason:
    NO_EDGE = "NO_EDGE"                              # 无概率优势
    LOW_EDGE = "LOW_EDGE"                            # 优势不足
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"            # 不确定性过高
    MODEL_DEGRADED = "MODEL_DEGRADED"                # 模型退化
    CORRELATION_LIMIT = "CORRELATION_LIMIT"          # 相关性限制
    DAILY_STOP = "DAILY_STOP"                        # 日亏损限制
    WEEKLY_DRAWDOWN = "WEEKLY_DRAWDOWN"              # 周回撤限制
    LOW_LIQUIDITY = "LOW_LIQUIDITY"                  # 低流动性
    EVENT_RISK = "EVENT_RISK"                        # 事件风险
    ACCOUNT_TOO_SMALL = "ACCOUNT_TOO_SMALL_FOR_RISK_RULE"  # 账户太小
    PORTFOLIO_LIMIT = "PORTFOLIO_LIMIT"              # 组合限制
    CONSECUTIVE_LOSS = "CONSECUTIVE_LOSS"            # 连续亏损
    SIGNAL_VALIDATION = "SIGNAL_VALIDATION"          # 信号验证失败
    ORDER_FAILED = "ORDER_FAILED"                    # API 下单失败
    CONFIG_DISABLED = "CONFIG_DISABLED"              # 配置禁用


@dataclass
class SettlementRecord:
    """结算记录（与 TradeRecord 关联）"""
    trade_id: str = ""                          # 关联 TradeLedger 的 trade_id
    settlement_id: str = ""                     # 结算记录 ID
    settlement_time_ms: int = 0                 # 结算时间
    settlement_status: str = SettlementStatus.PENDING
    settlement_method: str = ""                 # "api" / "balance_estimate" / "time_estimate" / "manual"

    # 结算价格
    entry_price: float = 0.0
    expiry_price: Optional[float] = None        # HIBT 结算指数价格
    expiry_price_source: str = ""               # "hibt_official" / "gate_io" / "estimated"

    # 盈亏
    stake_usd: int = 0
    net_payout_ratio: float = 0.0               # 实际净赔付率
    realized_pnl: Optional[float] = None
    result: str = ""                            # "WIN" / "LOSS" / "TIE"

    # 余额核对
    balance_before: Optional[float] = None
    balance_after: Optional[float] = None
    balance_change: Optional[float] = None

    # 元数据
    notes: str = ""
    created_at: str = ""
    confirmed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None or k in ("expiry_price", "balance_before", "balance_after", "balance_change", "confirmed_at")}

    @classmethod
    def from_dict(cls, d: dict) -> "SettlementRecord":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class SettlementLedger:
    """
    结算账本。

    注意：当前 HIBT API 不支持结算查询，所以结算数据主要来自：
    1. 时间推算 + 余额变化（ESTIMATED）
    2. 将来如果有结算 API，升级为 CONFIRMED

    结算状态机：
    PENDING → (时间到期) → ESTIMATED → (API确认) → CONFIRMED
    PENDING → (API直接返回) → CONFIRMED
    PENDING → (超时) → EXPIRED
    """

    def __init__(self, trade_ledger: Optional[TradeLedger] = None):
        self.trade_ledger = trade_ledger or get_trade_ledger()

    def estimate_settlement(
        self,
        trade_id: str,
        current_balance: float,
        pre_balance: float,
        expiry_price: Optional[float] = None,
    ) -> bool:
        """
        通过余额变化推算结算结果。

        这是当前唯一可用的结算方式（HIBT 无结算 API）。
        标记为 ESTIMATED 而非 CONFIRMED。
        """
        pnl = current_balance - pre_balance
        is_win = pnl > 0
        result = "TIE" if abs(pnl) < 0.001 else ("WIN" if is_win else "LOSS")

        # 更新 TradeLedger
        return self.trade_ledger.update_settlement(
            trade_id=trade_id,
            result=result,
            realized_pnl=round(pnl, 4),
            expiry_price=expiry_price,
            settlement_status=SettlementStatus.ESTIMATED,
        )

    def confirm_settlement(
        self,
        trade_id: str,
        result: str,
        realized_pnl: float,
        expiry_price: float,
        settlement_method: str = "api",
    ) -> bool:
        """
        通过 HIBT API 确认结算（未来可用时）。
        """
        return self.trade_ledger.update_settlement(
            trade_id=trade_id,
            result=result,
            realized_pnl=realized_pnl,
            expiry_price=expiry_price,
            settlement_status=SettlementStatus.CONFIRMED,
        )

    def mark_expired(self, trade_id: str) -> bool:
        """标记超时未确认的结算"""
        return self.trade_ledger._update_field(trade_id, "settlement_status", SettlementStatus.EXPIRED)

    def mark_disputed(self, trade_id: str, notes: str = "") -> bool:
        """标记结算异常"""
        self.trade_ledger._update_field(trade_id, "settlement_status", SettlementStatus.DISPUTED)
        if notes:
            self.trade_ledger._update_field(trade_id, "notes", notes)
        return True

    def get_pending(self) -> list[TradeRecord]:
        """获取所有待结算的交易"""
        return self.trade_ledger.get_pending_settlements()

    def reconcile_balance(self, current_balance: float) -> dict:
        """
        余额核对：检查所有 pending 订单的预期余额是否与实际一致。

        返回:
            {"matched": bool, "expected": float, "actual": float, "diff": float, "pending_pnl": float}
        """
        pending = self.get_pending()
        all_records = self.trade_ledger.load_all()
        settled = [
            r for r in all_records
            if r.settlement_status in (SettlementStatus.CONFIRMED, SettlementStatus.ESTIMATED)
            and r.entry_time_ms > int((time.time() - 86400) * 1000)
        ]

        total_pnl = sum(r.realized_pnl or 0 for r in settled)
        total_staked = sum(r.stake_usd for r in pending)

        expected = current_balance - total_pnl + total_staked  # 粗略估算
        return {
            "matched": abs(expected - current_balance) < 0.01,
            "expected": round(expected, 4),
            "actual": round(current_balance, 4),
            "diff": round(current_balance - expected, 4),
            "pending_count": len(pending),
            "pending_staked": total_staked,
        }

    def get_settlement_stats(self, days: int = 7) -> dict:
        """获取近期结算统计"""
        from_time = int((time.time() - days * 86400) * 1000)
        records = self.trade_ledger.query(
            from_time_ms=from_time,
        )
        settled = [r for r in records if r.result in ("WIN", "LOSS", "TIE")]
        pending = [r for r in records if r.settlement_status == SettlementStatus.PENDING]

        total = len(settled)
        wins = sum(1 for r in settled if r.result == "WIN")
        losses = sum(1 for r in settled if r.result == "LOSS")
        ties = sum(1 for r in settled if r.result == "TIE")

        total_pnl = sum(r.realized_pnl or 0 for r in settled)
        total_staked = sum(r.stake_usd for r in settled)

        return {
            "period_days": days,
            "settled": total,
            "pending": len(pending),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            "total_pnl": round(total_pnl, 4),
            "total_staked": total_staked,
            "roi": round(total_pnl / total_staked, 4) if total_staked > 0 else 0.0,
            "avg_pnl_per_trade": round(total_pnl / total, 4) if total > 0 else 0.0,
            "estimated_count": sum(1 for r in settled if r.settlement_status == SettlementStatus.ESTIMATED),
            "confirmed_count": sum(1 for r in settled if r.settlement_status == SettlementStatus.CONFIRMED),
        }


# ── 全局单例 ──
_settlement_ledger_instance: Optional[SettlementLedger] = None


def get_settlement_ledger() -> SettlementLedger:
    global _settlement_ledger_instance
    if _settlement_ledger_instance is None:
        _settlement_ledger_instance = SettlementLedger()
    return _settlement_ledger_instance