# -*- coding: utf-8 -*-
"""
RunMode — PAPER / SHADOW / LIVE 运行模式管理。

默认 SHADOW。LIVE 需多重条件人工开启，异常自动降级。
"""
import os
import time
import threading
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass, field


class RunMode(str, Enum):
    PAPER = "PAPER"       # 历史数据回测
    SHADOW = "SHADOW"     # 实时行情，记录但不提交订单
    LIVE = "LIVE"         # 真实下单


# ── Global Kill Switch ──
KILL_SWITCH = threading.Event()


def kill():
    """Activate the global kill switch — stops all trading immediately."""
    KILL_SWITCH.set()


def reset_kill():
    """Reset kill switch (for restart)."""
    KILL_SWITCH.clear()


def is_killed() -> bool:
    return KILL_SWITCH.is_set()


@dataclass
class LiveGateResult:
    """Result of LIVE gate check."""
    passed: bool = False
    hard_blocks: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    all_checks: Dict[str, bool] = field(default_factory=dict)

    @property
    def reasons(self) -> List[str]:
        return self.hard_blocks + self.soft_warnings


class LiveGate:
    """
    LIVE 门控 — 集中检查所有 LIVE 前置条件。

    LIVE 必须满足 ALL:
    1. ENABLE_LIVE_TRADING=true (环境变量)
    2. 管理页面人工确认 (admin_confirmed=True)
    3. HIBT 逐单结算可用
    4. 实时 payout 可用
    5. 概率校准器 ready
    6. 行情数据新鲜
    7. 模型版本有效
    8. 无未结算订单
    9. 未触发风控暂停
    """

    def __init__(self):
        self.admin_confirmed = False  # 需管理页面设置
        self._settlement_available: Optional[bool] = None
        self._payout_available: Optional[bool] = None

    def confirm_admin(self):
        self.admin_confirmed = True

    def revoke_admin(self):
        self.admin_confirmed = False

    def check(
        self,
        calibrator_ready: bool = False,
        data_fresh: bool = True,
        model_valid: bool = True,
        has_pending_orders: bool = False,
        risk_paused: bool = False,
        settlement_available: bool = False,
        payout_available: bool = False,
    ) -> LiveGateResult:
        """Check all LIVE preconditions. Returns LiveGateResult."""

        env_enabled = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
        checks = {}
        hard_blocks = []
        soft_warnings = []

        def _check(key: str, passed: bool, is_hard: bool, msg: str):
            checks[key] = passed
            if not passed:
                if is_hard:
                    hard_blocks.append(msg)
                else:
                    soft_warnings.append(msg)

        _check("env_enabled", env_enabled, True, "ENABLE_LIVE_TRADING != true")
        _check("admin_confirmed", self.admin_confirmed, True, "Admin not confirmed")
        _check("kill_switch", not is_killed(), True, "KILL_SWITCH active")
        _check("settlement_available", settlement_available, True,
               "HIBT settlement data unavailable — cannot verify trade results")
        _check("payout_available", payout_available, True,
               "Real-time payout unavailable — using assumed values not allowed in LIVE")
        _check("calibrator_ready", calibrator_ready, True,
               "Probability calibrator not ready")
        _check("data_fresh", data_fresh, True,
               "Market data is stale")
        _check("model_valid", model_valid, True,
               "Model version invalid or degraded")
        _check("no_pending_orders", not has_pending_orders, False,
               "Has pending unsettled orders")
        _check("risk_ok", not risk_paused, True,
               "Risk manager has paused trading")

        return LiveGateResult(
            passed=len(hard_blocks) == 0,
            hard_blocks=hard_blocks,
            soft_warnings=soft_warnings,
            all_checks=checks,
        )


class RunModeManager:
    """
    运行模式管理器。

    规则:
    - 默认 SHADOW
    - 程序重启自动恢复为 SHADOW
    - LIVE 需要 ENABLE_LIVE_TRADING=true + admin_confirmed + gate check
    - 异常自动降级到 SHADOW
    """

    def __init__(self, mode: RunMode = RunMode.SHADOW):
        self._mode = RunMode.PAPER if os.getenv("BACKTEST_MODE", "false").lower() == "true" else RunMode.SHADOW
        if mode != RunMode.SHADOW:
            self._mode = mode
        self._live_gate = LiveGate()
        self._degraded_from_live = False
        self._degrade_reason = ""
        self._degrade_time: float = 0.0
        self._live_start_time: float = 0.0
        self._symbol_mode: Dict[str, str] = {}  # symbol → "SHADOW_ACTIVE" / "OBSERVE_ONLY" / "DISABLED"

    @property
    def mode(self) -> RunMode:
        return self._mode

    @property
    def is_paper(self) -> bool:
        return self._mode == RunMode.PAPER

    @property
    def is_shadow(self) -> bool:
        return self._mode == RunMode.SHADOW

    @property
    def is_live(self) -> bool:
        return self._mode == RunMode.LIVE

    @property
    def can_place_orders(self) -> bool:
        """Can we submit real orders?"""
        return self._mode == RunMode.LIVE and not is_killed()

    def attempt_live(self, **gate_checks) -> LiveGateResult:
        """Attempt to enter LIVE mode. Returns gate check result."""
        result = self._live_gate.check(**gate_checks)
        if result.passed:
            self._mode = RunMode.LIVE
            self._degraded_from_live = False
            self._live_start_time = time.time()
        return result

    def set_shadow(self):
        """Force SHADOW mode."""
        if self._mode == RunMode.LIVE:
            self._degraded_from_live = True
            self._degrade_time = time.time()
        self._mode = RunMode.SHADOW

    def set_paper(self):
        self._mode = RunMode.PAPER

    def degrade(self, reason: str):
        """Auto-degrade to SHADOW on failure."""
        if self._mode == RunMode.LIVE:
            self._degraded_from_live = True
            self._degrade_reason = reason
            self._degrade_time = time.time()
            self._mode = RunMode.SHADOW

    def get_live_gate(self) -> LiveGate:
        return self._live_gate

    def confirm_live_admin(self):
        self._live_gate.confirm_admin()

    def revoke_live_admin(self):
        self._live_gate.revoke_admin()
        if self._mode == RunMode.LIVE:
            self.degrade("admin_revoked")

    def set_symbol_mode(self, symbol: str, mode: str):
        """Set per-symbol mode: SHADOW_ACTIVE / OBSERVE_ONLY / DISABLED."""
        self._symbol_mode[symbol] = mode

    def get_symbol_mode(self, symbol: str) -> str:
        from .. import config
        return self._symbol_mode.get(
            symbol,
            config.SHADOW_SYMBOL_MODE.get(symbol, "DISABLED")
        )

    def is_symbol_active(self, symbol: str) -> bool:
        return self.get_symbol_mode(symbol) == "SHADOW_ACTIVE"

    def get_status(self) -> dict:
        return {
            "mode": self._mode.value,
            "is_killed": is_killed(),
            "degraded_from_live": self._degraded_from_live,
            "degrade_reason": self._degrade_reason,
            "live_uptime_seconds": int(time.time() - self._live_start_time) if self._live_start_time > 0 and self._mode == RunMode.LIVE else 0,
            "symbol_modes": dict(self._symbol_mode),
        }


# ── Global singleton ──
_run_mode_manager: Optional[RunModeManager] = None


def get_run_mode_manager() -> RunModeManager:
    global _run_mode_manager
    if _run_mode_manager is None:
        _run_mode_manager = RunModeManager()
    return _run_mode_manager
