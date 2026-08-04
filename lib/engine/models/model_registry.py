# -*- coding: utf-8 -*-
"""
ModelRegistry — manage model versions, activation, and OOS validation.
"""
import os
import json
import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """Metadata for a single model version."""
    model_name: str = ""
    model_type: str = ""            # "fast_entry" / "slow_ensemble" / "calibrator"
    symbol: str = ""
    file_path: str = ""
    feature_count: int = 0
    training_start: str = ""
    training_end: str = ""
    oos_auc: float = 0.0
    oos_brier: float = 0.0
    oos_logloss: float = 0.0
    oos_trades: int = 0
    oos_win_rate: float = 0.0
    oos_ev: float = 0.0
    oos_pnl: float = 0.0
    is_active: int = 0              # 0=disabled, 1=shadow_only, 2=live_ready
    requires_manual_promotion: bool = True
    notes: str = ""


class ModelRegistry:
    """Central registry for all model versions."""

    # Symbols that require OOS validation before LIVE
    SYMBOLS_REQUIRING_OOS = {"BTCUSDT"}  # BTC disabled by default

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._db = None  # Lazy loaded

    def register(
        self,
        model_name: str,
        model_type: str,
        symbol: str,
        file_path: str,
        feature_count: int = 0,
        training_start: str = "",
        training_end: str = "",
        is_active: int = 1,          # Default: shadow_only
        requires_manual_promotion: bool = True,
    ):
        """Register a model."""
        self._models[model_name] = ModelInfo(
            model_name=model_name,
            model_type=model_type,
            symbol=symbol,
            file_path=file_path,
            feature_count=feature_count,
            training_start=training_start,
            training_end=training_end,
            is_active=is_active,
            requires_manual_promotion=requires_manual_promotion,
        )

    def get(self, model_name: str) -> Optional[ModelInfo]:
        return self._models.get(model_name)

    def get_for_symbol(self, symbol: str, model_type: str = "fast_entry") -> Optional[ModelInfo]:
        """Get the active model for a symbol."""
        for m in self._models.values():
            if m.symbol == symbol and m.model_type == model_type and m.is_active > 0:
                return m
        return None

    def record_oos_results(
        self,
        model_name: str,
        oos_auc: float,
        oos_brier: float,
        oos_logloss: float,
        oos_trades: int,
        oos_win_rate: float,
        oos_ev: float,
        oos_pnl: float,
    ):
        """Record OOS validation results."""
        if model_name not in self._models:
            return
        m = self._models[model_name]
        m.oos_auc = oos_auc
        m.oos_brier = oos_brier
        m.oos_logloss = oos_logloss
        m.oos_trades = oos_trades
        m.oos_win_rate = oos_win_rate
        m.oos_ev = oos_ev
        m.oos_pnl = oos_pnl

        # Auto-promote if OOS passes
        if (
            oos_auc >= 0.60
            and oos_brier <= 0.25
            and oos_trades >= 100
            and oos_win_rate >= 0.50
        ):
            if not m.requires_manual_promotion:
                m.is_active = 2  # live_ready
            m.notes = "OOS validation passed"

    def can_live(self, symbol: str) -> bool:
        """Check if a symbol's model is LIVE-ready."""
        if symbol in self.SYMBOLS_REQUIRING_OOS:
            fast = self.get_for_symbol(symbol, "fast_entry")
            slow = self.get_for_symbol(symbol, "slow_ensemble")
            if fast is None or fast.is_active < 2:
                return False
            if slow is None or slow.is_active < 2:
                return False
            return True
        return True

    def is_valid(self, model_name: str) -> bool:
        """Check if model exists and is active."""
        m = self._models.get(model_name)
        return m is not None and m.is_active > 0 and os.path.exists(m.file_path)

    def get_version_string(self, symbol: str) -> str:
        """Get a version string for this symbol's active models."""
        fast = self.get_for_symbol(symbol, "fast_entry")
        slow = self.get_for_symbol(symbol, "slow_ensemble")
        return f"fast={fast.model_name if fast else 'none'},slow={slow.model_name if slow else 'none'}"

    def list_all(self) -> List[ModelInfo]:
        return list(self._models.values())

    def list_active(self) -> List[ModelInfo]:
        return [m for m in self._models.values() if m.is_active > 0]

    def list_live_ready(self) -> List[ModelInfo]:
        return [m for m in self._models.values() if m.is_active >= 2]


# ── Global singleton ──
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
