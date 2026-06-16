# -*- coding: utf-8 -*-
"""
OrderExecutor — 订单执行器（流水线末端）

职责:
1. 根据 TradeSignal.action 执行操作：
   - "open" → 调 place_order 开底仓 (3U)
   - "add"  → 调 place_order 加仓 (3U)
   - "close_and_open" → 标记 pending_close，等待结算后开反向仓
   - "close" → 仅标记 pending_close（被动等待结算）

2. 维护 PositionBook（品种持仓账簿）
   - 每个品种跟踪 direction, entry_price, amount, open_time
   - 浮盈计算
   - pending_close 状态管理

3. 重要限制:
   - HIBT 不支持主动平仓 API（定时二元期权，15分钟到期自动结算）
   - "平仓"实现为标记 pending_close → 等结算 → 执行反向开仓
"""
import time
from typing import Optional
from . import config
from .models import TradeSignal, PositionState
from .exchange import place_order


class PositionBook:
    """品种持仓账簿"""

    def __init__(self):
        # symbol -> PositionState
        self._positions: dict[str, PositionState] = {}

    def get(self, symbol: str) -> Optional[PositionState]:
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def open_position(self, signal: TradeSignal, current_ts: int, entry_bar_ts):
        """开仓（或加仓）"""
        existing = self._positions.get(signal.symbol)
        if existing and existing.direction == signal.direction:
            # 加仓：累积投入金额
            existing.amount += config.BET_MIN
            # 加权平均持仓价
            total = existing.amount
            existing.entry_price = (
                existing.entry_price * (existing.amount - config.BET_MIN) / total
                + signal.entry_price * config.BET_MIN / total
            )
            existing.open_time_ms = current_ts
            existing.pending_close = False
        else:
            self._positions[signal.symbol] = PositionState(
                symbol=signal.symbol,
                direction=signal.direction,
                amount=config.BET_MIN,
                entry_price=signal.entry_price,
                open_time_ms=current_ts,
                entry_bar_ts=entry_bar_ts,
            )

    def mark_pending_close(self, symbol: str):
        """标记品种为等待平仓状态"""
        pos = self._positions.get(symbol)
        if pos:
            pos.pending_close = True

    def remove(self, symbol: str):
        """结算完成后清除持仓记录"""
        self._positions.pop(symbol, None)

    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """更新未实现盈亏（仅用于日志和通知）"""
        pos = self._positions.get(symbol)
        if not pos:
            return
        if pos.direction == 1:
            pos.unrealized_roi = (current_price - pos.entry_price) / pos.entry_price
        else:
            pos.unrealized_roi = (pos.entry_price - current_price) / pos.entry_price
        pos.unrealized_pnl = pos.unrealized_roi * pos.amount

    @property
    def active_count(self) -> int:
        return len([p for p in self._positions.values() if not p.pending_close])

    @property
    def all_positions(self) -> dict:
        return dict(self._positions)


class OrderExecutor:
    """
    订单执行器，按 TradeSignal.action 执行不同策略。

    与 engine.py 关系：
    - engine 的 _check_settlement() 检测到余额变化后，
      调用 executor.on_settlement() 清理持仓状态
    - engine 调用 executor.execute() 执行订单
    """

    def __init__(self):
        self.positions = PositionBook()
        # 品种冷却跟踪
        self.last_settlement_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_reject_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}

    def execute(self, signal: TradeSignal, current_ts: int, entry_bar_ts) -> bool:
        """
        执行交易信号。

        返回 True 表示成功下单, False 表示失败（API拒单）。
        """
        action = signal.action

        if action == "open":
            return self._open_position(signal, current_ts, entry_bar_ts)

        elif action == "add":
            return self._add_position(signal, current_ts, entry_bar_ts)

        elif action == "close_and_open":
            # 标记 pending_close → 等待结算 → 结算后自动执行反向开仓
            self.positions.mark_pending_close(signal.symbol)
            # 暂存反向信号，结算后使用
            self._pending_reverse_signal = signal
            return True

        elif action == "close":
            self.positions.mark_pending_close(signal.symbol)
            return True

        return False

    def _place(self, signal: TradeSignal, amount: float) -> bool:
        """统一下单接口"""
        result = place_order(signal.symbol, signal.direction, amount, config.HOLD_MINUTES)
        return result.ok

    def _open_position(self, signal: TradeSignal, current_ts: int, entry_bar_ts) -> bool:
        """开底仓（3U）"""
        if self.positions.has_position(signal.symbol):
            # 安全保护：不应该发生
            return False

        ok = self._place(signal, config.BET_MIN)
        if ok:
            self.positions.open_position(signal, current_ts, entry_bar_ts)
        return ok

    def _add_position(self, signal: TradeSignal, current_ts: int, entry_bar_ts) -> bool:
        """加仓（追加 3U）"""
        ok = self._place(signal, config.BET_MIN)
        if ok:
            self.positions.open_position(signal, current_ts, entry_bar_ts)
        return ok

    def on_settlement(self, symbol: str):
        """
        结算完成回调（由 engine._check_settlement 调用）。

        如果有暂存的反向信号，结算后自动执行反向开仓。
        """
        self.positions.remove(symbol)
        self.last_settlement_ts[symbol] = int(time.time() * 1000)

        # 检查是否有 pending 的反向信号
        if hasattr(self, '_pending_reverse_signal') and self._pending_reverse_signal is not None:
            rev = self._pending_reverse_signal
            if rev.symbol == symbol:
                # 执行反向开仓（作为全新信号 — 以底仓开）
                ok = self._place(rev, config.BET_MIN)
                if ok:
                    self.positions.open_position(rev, int(time.time() * 1000), None)
                self._pending_reverse_signal = None

    def flush_pending_reverse(self, symbol: str, current_ts: int):
        """
        如果 pending_close 后结算已完成，但反向信号还没执行（例如 on_settlement 漏了），
        这里作为兜底清理。
        """
        pos = self.positions.get(symbol)
        if pos and pos.pending_close:
            # 检查是否已经结算（余额变化已确认）
            has_settled = current_ts - self.last_settlement_ts.get(symbol, 0) < 1000
            # 如果结算时间戳刚更新过，说明刚才结算了
            if has_settled:
                self.on_settlement(symbol)
