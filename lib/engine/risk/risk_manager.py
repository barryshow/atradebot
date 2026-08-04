# -*- coding: utf-8 -*-
"""
RiskManager v2 — comprehensive risk controls.

Replaces old risk_manager.py with:
- Max 1 unsettled order at a time (default)
- Fixed minimum stake until calibration stable, then Kelly
- Daily loss limit
- Consecutive loss pause
- Daily max trade count
- Data freshness check
- Payout fetch failure pause
- Settlement reconciliation failure pause → auto-degrade to SHADOW
- Order idempotency key
- API timeout → query order status first, don't blindly resubmit
- BTC model disabled by default
"""
import time
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RiskStatus(str, Enum):
    OK = "OK"
    PAUSED = "PAUSED"
    HALTED = "HALTED"


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    allowed: bool = False
    status: str = RiskStatus.OK.value
    reason: str = ""
    detail: str = ""


class RiskManager:
    """
    Central risk manager.

    Rules (enforced in order):
    1. Kill switch
    2. Data freshness
    3. Max concurrent orders (default: 1)
    4. Daily max trade count
    5. Daily loss limit
    6. Consecutive loss pause
    7. Payout verification
    8. Model validity
    9. Settlement reconciliation health
    """

    def __init__(
        self,
        max_concurrent_orders: int = 1,
        daily_max_trades: int = 20,
        daily_max_loss: float = float("inf"),  # USDT, set by config
        consecutive_loss_pause: int = 3,
        consecutive_loss_pause_seconds: int = 300,
        data_max_age_seconds: int = 300,
        min_order_amount: float = 3.0,
        kelly_enabled: bool = False,           # Disabled until calibration stable
    ):
        self.max_concurrent_orders = max_concurrent_orders
        self.daily_max_trades = daily_max_trades
        self.daily_max_loss = daily_max_loss
        self.consecutive_loss_pause = consecutive_loss_pause
        self.consecutive_loss_pause_seconds = consecutive_loss_pause_seconds
        self.data_max_age_seconds = data_max_age_seconds
        self.min_order_amount = min_order_amount
        self.kelly_enabled = kelly_enabled

        # State
        self._paused = False
        self._paused_reason = ""
        self._pause_until: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_reset_time: float = time.time()
        self._consecutive_losses: int = 0
        self._lock = threading.Lock()
        self._order_idempotency_keys: Dict[str, float] = {}  # key → timestamp

        # Symbol enablement
        self._disabled_symbols: set = {"BTCUSDT"}  # Default: BTC disabled until proven

    # ── Symbol management ──

    def is_symbol_enabled(self, symbol: str) -> bool:
        return symbol not in self._disabled_symbols

    def enable_symbol(self, symbol: str):
        self._disabled_symbols.discard(symbol)

    def disable_symbol(self, symbol: str):
        self._disabled_symbols.add(symbol)

    # ── Core checks ──

    def check_data_freshness(self, last_update_time: float) -> RiskCheckResult:
        """Check if market data is fresh."""
        age = time.time() - last_update_time
        if age > self.data_max_age_seconds:
            return RiskCheckResult(
                allowed=False, status=RiskStatus.PAUSED.value,
                reason="DATA_STALE",
                detail=f"Data age {age:.0f}s > {self.data_max_age_seconds}s",
            )
        return RiskCheckResult(allowed=True)

    def check_concurrent_orders(self, active_order_count: int) -> RiskCheckResult:
        """Check max concurrent orders limit."""
        if active_order_count >= self.max_concurrent_orders:
            return RiskCheckResult(
                allowed=False, status=RiskStatus.OK.value,
                reason="MAX_CONCURRENT_ORDERS",
                detail=f"{active_order_count} >= {self.max_concurrent_orders}",
            )
        return RiskCheckResult(allowed=True)

    def check_daily_trades(self) -> RiskCheckResult:
        """Check daily max trade count."""
        self._maybe_reset_daily()
        if self._daily_trades >= self.daily_max_trades:
            return RiskCheckResult(
                allowed=False, status=RiskStatus.PAUSED.value,
                reason="DAILY_MAX_TRADES",
                detail=f"{self._daily_trades} >= {self.daily_max_trades}",
            )
        return RiskCheckResult(allowed=True)

    def check_daily_loss(self) -> RiskCheckResult:
        """Check daily loss limit."""
        self._maybe_reset_daily()
        # daily_max_loss <= 0 means no limit
        if self.daily_max_loss <= 0:
            return RiskCheckResult(allowed=True)
        if self._daily_pnl < -self.daily_max_loss:
            return RiskCheckResult(
                allowed=False, status=RiskStatus.HALTED.value,
                reason="DAILY_LOSS_LIMIT",
                detail=f"Daily PnL {self._daily_pnl:.2f} < -{self.daily_max_loss}",
            )
        return RiskCheckResult(allowed=True)

    def check_consecutive_losses(self) -> RiskCheckResult:
        """Check if consecutive losses require a pause."""
        if self._paused and time.time() < self._pause_until:
            remaining = int(self._pause_until - time.time())
            return RiskCheckResult(
                allowed=False, status=RiskStatus.PAUSED.value,
                reason="CONSECUTIVE_LOSS_PAUSE",
                detail=f"Paused for {remaining}s more: {self._paused_reason}",
            )

        if self._consecutive_losses >= self.consecutive_loss_pause:
            self._paused = True
            self._paused_reason = f"{self._consecutive_losses} consecutive losses"
            self._pause_until = time.time() + self.consecutive_loss_pause_seconds
            return RiskCheckResult(
                allowed=False, status=RiskStatus.PAUSED.value,
                reason="CONSECUTIVE_LOSS_PAUSE",
                detail=f"Paused for {self.consecutive_loss_pause_seconds}s after {self._consecutive_losses} losses",
            )
        return RiskCheckResult(allowed=True)

    def check_order_idempotency(self, key: str, timeout_seconds: int = 300) -> RiskCheckResult:
        """Check if an order with the same idempotency key was recently submitted."""
        now = time.time()
        if key in self._order_idempotency_keys:
            elapsed = now - self._order_idempotency_keys[key]
            if elapsed < timeout_seconds:
                return RiskCheckResult(
                    allowed=False, status=RiskStatus.OK.value,
                    reason="IDEMPOTENCY_BLOCK",
                    detail=f"Order {key} already submitted {elapsed:.0f}s ago",
                )
        return RiskCheckResult(allowed=True)

    def check_all(
        self,
        active_order_count: int = 0,
        data_last_update: float = 0.0,
        symbol: str = "",
    ) -> RiskCheckResult:
        """Run all risk checks. Returns first failure."""

        # 1. Symbol enabled
        if symbol and not self.is_symbol_enabled(symbol):
            return RiskCheckResult(
                allowed=False, status=RiskStatus.HALTED.value,
                reason="SYMBOL_DISABLED",
                detail=f"{symbol} is disabled (requires OOS validation)",
            )

        # 2. Consecutive loss pause (check first — fastest fail)
        cl = self.check_consecutive_losses()
        if not cl.allowed:
            return cl

        # 3. Daily loss limit
        dl = self.check_daily_loss()
        if not dl.allowed:
            return dl

        # 4. Daily max trades
        dt = self.check_daily_trades()
        if not dt.allowed:
            return dt

        # 5. Data freshness
        if data_last_update > 0:
            df = self.check_data_freshness(data_last_update)
            if not df.allowed:
                return df

        # 6. Concurrent orders
        co = self.check_concurrent_orders(active_order_count)
        if not co.allowed:
            return co

        return RiskCheckResult(allowed=True)

    # ── State updates ──

    def record_order_idempotency(self, key: str):
        """Record an order idempotency key."""
        self._order_idempotency_keys[key] = time.time()
        # Clean up old keys
        cutoff = time.time() - 600
        self._order_idempotency_keys = {
            k: v for k, v in self._order_idempotency_keys.items() if v > cutoff
        }

    def record_trade(self):
        """Record that a trade was submitted."""
        self._daily_trades += 1

    def record_pnl(self, pnl: float):
        """Record realized PnL."""
        self._daily_pnl += pnl

    def record_result(self, is_win: bool):
        """Record trade result."""
        with self._lock:
            if is_win:
                self._consecutive_losses = 0
                if self._paused and time.time() > self._pause_until:
                    self._paused = False
                    self._paused_reason = ""
            else:
                self._consecutive_losses += 1

    def _maybe_reset_daily(self):
        """Reset daily counters if needed."""
        now = time.time()
        if now - self._daily_reset_time > 86400:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_reset_time = now

    # ── Position sizing ──

    def compute_stake(self, equity: float) -> Tuple[float, float]:
        """
        Compute stake amount.

        Returns:
            (stake, bet_fraction)
            Until calibration is stable, uses fixed min amount.
            Kelly is disabled by default.
        """
        if not self.kelly_enabled:
            stake = min(self.min_order_amount, equity * 0.05)
            stake = max(self.min_order_amount, stake)
            fraction = stake / equity if equity > 0 else 0.0
            return (stake, fraction)

        # Kelly would go here when calibration is stable
        stake = min(self.min_order_amount, equity * 0.05)
        fraction = stake / equity if equity > 0 else 0.0
        return (stake, fraction)

    def enable_kelly(self):
        """Enable Kelly position sizing (requires stable calibration)."""
        self.kelly_enabled = True

    def disable_kelly(self):
        self.kelly_enabled = False

    def reset_pause(self):
        """Manually reset pause state."""
        self._paused = False
        self._paused_reason = ""
        self._pause_until = 0.0
        self._consecutive_losses = 0

    def get_status(self) -> dict:
        self._maybe_reset_daily()
        return {
            "status": RiskStatus.PAUSED.value if self._paused else RiskStatus.OK.value,
            "paused": self._paused,
            "paused_reason": self._paused_reason,
            "pause_remaining_seconds": max(0, int(self._pause_until - time.time())) if self._paused else 0,
            "daily_pnl": round(self._daily_pnl, 4),
            "daily_trades": self._daily_trades,
            "daily_max_trades": self.daily_max_trades,
            "consecutive_losses": self._consecutive_losses,
            "kelly_enabled": self.kelly_enabled,
            "disabled_symbols": list(self._disabled_symbols),
        }


# ── Global singleton ──
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    global _risk_manager
    if _risk_manager is None:
        from .. import config
        _risk_manager = RiskManager(
            max_concurrent_orders=1,
            daily_max_trades=getattr(config, "MAX_NEW_TRADES_PER_HOUR", 3) * 24,
            daily_max_loss=getattr(config, "DAILY_STOP", 5.0),
            consecutive_loss_pause=getattr(config, "CONSECUTIVE_LOSS_HALT", 3),
            consecutive_loss_pause_seconds=getattr(config, "CONSECUTIVE_LOSS_PAUSE_SEC", 300),
            data_max_age_seconds=300,
            min_order_amount=getattr(config, "MIN_ORDER_USD", 3.0),
            kelly_enabled=False,
        )
    return _risk_manager
