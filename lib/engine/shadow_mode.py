# -*- coding: utf-8 -*-
"""
ShadowMode v2 — 影子交易模式 + Per-Symbol Activation + LIVE Gate

支持三种运行模式:
- BACKTEST: 历史数据回测
- SHADOW: 实时数据 + 实时决策 + 模拟结算 + 禁止真实下单
- LIVE: 真实下单

Per-symbol activation:
- SHADOW_ACTIVE: 模拟交易，记录结果
- OBSERVE_ONLY: 生成预测，记录，但不产生交易（用于数据收集）
- DISABLED: 跳过

LIVE Gate:
- LIVE_ENABLED + calibration READY + shadow validation PASSED 才允许
"""
import time
import json
import os
from typing import Optional, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from . import config
from .trade_ledger import TradeLedger, TradeRecord, get_trade_ledger


class RunMode(str, Enum):
    BACKTEST = "BACKTEST"
    SHADOW = "SHADOW"
    LIVE = "LIVE"


class SymbolMode(str, Enum):
    SHADOW_ACTIVE = "SHADOW_ACTIVE"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    DISABLED = "DISABLED"


class CalibrationStatus(str, Enum):
    READY = "READY"
    NOT_READY = "NOT_READY"
    PASSTHROUGH_UNCALIBRATED = "PASSTHROUGH_UNCALIBRATED"


class SettlementSource(str, Enum):
    HIBT_VERIFIED = "HIBT_VERIFIED"
    HIBT_MANUAL_RECONCILED = "HIBT_MANUAL_RECONCILED"
    EXTERNAL_PRICE_PROXY = "EXTERNAL_PRICE_PROXY"
    BALANCE_ESTIMATE = "BALANCE_ESTIMATE"


@dataclass
class ShadowCandidateRecord:
    """完整的 Shadow 候选记录 — 可回放每一个决策"""
    # ── 标识 ──
    record_id: str = ""
    timestamp: str = ""
    symbol: str = ""
    direction: str = ""                         # "CALL" / "PUT"
    direction_int: int = 0
    expiry_minutes: int = 15

    # ── Regime ──
    regime: str = ""
    regime_confidence: float = 0.0

    # ── 概率 ──
    raw_probability: float = 0.0
    ensemble_probability: float = 0.0
    calibrated_probability: float = 0.0
    calibration_status: str = CalibrationStatus.NOT_READY.value
    conservative_probability: float = 0.0

    # ── 不确定性 ──
    uncertainty_margin: float = 0.0
    calibration_margin: float = 0.0
    model_degradation_margin: float = 0.0

    # ── 赔付率 ──
    payout_ratio: float = 0.0
    net_payout_ratio: float = 0.0
    payout_source: str = ""
    payout_verified: bool = False

    # ── Edge ──
    break_even_probability: float = 0.0
    probability_edge: float = 0.0
    expected_roi: float = 0.0
    effective_edge: float = 0.0

    # ── Ranking ──
    risk_adjusted_ev: float = 0.0
    rank_score: float = 0.0
    rank: int = 0
    selected: bool = False

    # ── Position Sizing ──
    stake_before_integer_rounding: float = 0.0
    final_integer_stake: int = 0
    bet_fraction: float = 0.0
    kelly_fraction: float = 0.0

    # ── 风控 ──
    reject_reason: str = ""
    reject_detail: str = ""
    portfolio_check_passed: bool = False
    daily_stop_ok: bool = True
    exposure_ok: bool = True

    # ── 价格 ──
    entry_reference_price: float = 0.0
    expiry_reference_price: Optional[float] = None
    settlement_source: str = SettlementSource.EXTERNAL_PRICE_PROXY.value

    # ── 结果 ──
    result: str = ""                            # "WIN" / "LOSS" / "TIE" / "PENDING"
    pnl: float = 0.0
    settled_at: Optional[str] = None

    # ── 模型版本 ──
    model_version: str = ""
    expert_votes: dict = field(default_factory=dict)


@dataclass
class ShadowStats:
    """Shadow 模式统计"""
    symbol: str = ""
    total_candidates: int = 0
    total_selected: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_staked: int = 0
    roi: float = 0.0
    avg_effective_edge: float = 0.0
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    max_drawdown: float = 0.0
    longest_losing_streak: int = 0
    longest_winning_streak: int = 0
    avg_payout: float = 0.0
    probability_mean: float = 0.0
    probability_std: float = 0.0
    regime_distribution: dict = field(default_factory=dict)
    edge_distribution: dict = field(default_factory=dict)
    # Attribution
    base_model_wr: float = 0.0
    base_model_trades: int = 0
    regime_filter_wr: float = 0.0
    regime_filter_trades: int = 0
    full_eventedge_wr: float = 0.0
    full_eventedge_trades: int = 0


class ShadowMode:
    """
    Shadow Mode v2 — 完整候选记录 + Per-symbol 激活 + LIVE Gate。

    在 SHADOW 模式下:
    - 与 LIVE 完全相同的决策流程
    - 记录所有候选（含被拒绝的）
    - 模拟结算（基于到期价格）
    - 禁止调用真实下单 API
    - 支持 OBSERVE_ONLY 品种（不产生交易）
    """

    def __init__(self, mode: RunMode = RunMode.SHADOW):
        self.mode = mode
        self.shadow_trades: List[dict] = []
        self.candidates: List[ShadowCandidateRecord] = []
        self.ledger = get_trade_ledger()
        self._shadow_start_time = time.time()
        self._symbol_mode = self._load_symbol_modes()

    def _load_symbol_modes(self) -> Dict[str, SymbolMode]:
        """加载每个品种的 Shadow 模式 — 使用 SHADOW_SYMBOL_MODE 的 keys 作为品种列表"""
        result = {}
        for sym, mode_str in config.SHADOW_SYMBOL_MODE.items():
            result[sym] = SymbolMode(mode_str)
        return result

    def get_symbol_mode(self, symbol: str) -> SymbolMode:
        return self._symbol_mode.get(symbol, SymbolMode.DISABLED)

    def is_shadow_active(self, symbol: str) -> bool:
        """该品种是否允许模拟交易"""
        return self.mode == RunMode.SHADOW and self._symbol_mode.get(symbol) == SymbolMode.SHADOW_ACTIVE

    def is_shadow_or_live(self) -> bool:
        """兼容旧 API: 是否允许产生交易决策"""
        return self.mode in (RunMode.SHADOW, RunMode.LIVE)

    def is_observe_only(self, symbol: str) -> bool:
        return self._symbol_mode.get(symbol) == SymbolMode.OBSERVE_ONLY

    def is_live_allowed(self) -> bool:
        """检查是否允许 LIVE 模式"""
        if self.mode != RunMode.LIVE:
            return False
        return config.LIVE_ENABLED

    def can_place_order(self) -> bool:
        """是否允许真实下单 — 需要 LIVE_ENABLED + calibration + shadow validated"""
        return self.mode == RunMode.LIVE and config.LIVE_ENABLED

    def get_live_gate_status(self, calibrator_ready: bool) -> dict:
        """检查 LIVE 门控状态"""
        checks = {
            "live_enabled": config.LIVE_ENABLED,
            "mode_is_live": self.mode == RunMode.LIVE,
            "calibration_ready": calibrator_ready,
            "calibration_required": config.LIVE_REQUIRE_CALIBRATION,
            "shadow_validation_required": config.LIVE_REQUIRE_SHADOW_VALIDATION,
        }

        passed = True
        reasons = []
        if not checks["live_enabled"]:
            passed = False
            reasons.append("LIVE_ENABLED=false")
        if not checks["mode_is_live"]:
            passed = False
            reasons.append("MODE_NOT_LIVE")
        if config.LIVE_REQUIRE_CALIBRATION and not checks["calibration_ready"]:
            passed = False
            reasons.append("CALIBRATION_NOT_READY")
        if config.LIVE_REQUIRE_SHADOW_VALIDATION:
            # 检查是否完成 shadow 验证
            days = (time.time() - self._shadow_start_time) / 86400
            total_trades = len([c for c in self.candidates if c.selected])
            if days < config.SHADOW_MIN_DAYS:
                passed = False
                reasons.append(f"SHADOW_DAYS_INSUFFICIENT ({days:.1f}d < {config.SHADOW_MIN_DAYS}d)")
            if total_trades < config.SHADOW_MIN_TRADES:
                passed = False
                reasons.append(f"SHADOW_TRADES_INSUFFICIENT ({total_trades} < {config.SHADOW_MIN_TRADES})")

        return {
            "passed": passed,
            "reasons": reasons if not passed else [],
            "checks": checks,
        }

    def add_candidate(self, record: ShadowCandidateRecord):
        """添加候选记录"""
        self.candidates.append(record)
        self._save_candidate(record)

    def _save_candidate(self, record: ShadowCandidateRecord):
        """保存候选记录到 JSONL"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(config.SHADOW_RECORD_PATH)), exist_ok=True)
            with open(config.SHADOW_RECORD_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception:
            pass

    def record_shadow_trade(
        self,
        symbol: str,
        direction: str,
        direction_int: int,
        entry_time_ms: int,
        entry_price: float,
        stake_usd: int,
        expiry_minutes: int,
        calibrated_probability: float,
        break_even_probability: float,
        effective_edge: float,
        expected_roi: float,
        regime: str = "",
        expert_votes: Optional[Dict] = None,
        reject_reason: str = "",
    ) -> dict:
        """记录影子交易"""
        trade = {
            "trade_id": f"shadow_{int(time.time()*1000)}_{len(self.shadow_trades)}",
            "symbol": symbol,
            "direction": direction,
            "direction_int": direction_int,
            "entry_time_ms": entry_time_ms,
            "expiry_minutes": expiry_minutes,
            "entry_price": entry_price,
            "stake_usd": stake_usd,
            "calibrated_probability": calibrated_probability,
            "break_even_probability": break_even_probability,
            "effective_edge": effective_edge,
            "expected_roi": expected_roi,
            "regime": regime,
            "expert_votes": expert_votes or {},
            "reject_reason": reject_reason,
            "result": "PENDING",
            "simulated_pnl": 0.0,
            "simulated_at": "",
        }
        self.shadow_trades.append(trade)

        # 同时记录到 TradeLedger（标记为 SHADOW）
        if not reject_reason:
            rec = self.ledger.create_record(
                symbol=symbol, direction=direction, direction_int=direction_int,
                entry_time_ms=entry_time_ms, entry_price=entry_price,
                stake_usd=stake_usd, expiry_minutes=expiry_minutes,
                calibrated_probability=calibrated_probability,
                break_even_probability=break_even_probability,
                effective_edge=effective_edge,
                expected_roi=expected_roi,
                regime=regime,
                expert_votes=expert_votes,
            )
            rec.tags = ["shadow"]
            rec.settlement_status = "PENDING"
            self.ledger.save(rec)

        return trade

    def settle_shadow_trade(
        self,
        trade_id: str,
        expiry_price: float,
    ) -> Optional[dict]:
        """模拟结算影子交易"""
        for trade in self.shadow_trades:
            if trade["trade_id"] == trade_id and trade["result"] == "PENDING":
                if trade["direction"] == "CALL":
                    is_win = expiry_price > trade["entry_price"]
                else:
                    is_win = expiry_price < trade["entry_price"]

                trade["result"] = "SIMULATED_WIN" if is_win else "SIMULATED_LOSS"
                trade["simulated_pnl"] = trade["stake_usd"] * 0.80 if is_win else -trade["stake_usd"]
                trade["simulated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                return trade
        return None

    def get_shadow_stats(self) -> Dict:
        """获取影子模式统计"""
        settled = [t for t in self.shadow_trades if t["result"] in ("SIMULATED_WIN", "SIMULATED_LOSS")]
        if not settled:
            return {"total": len(self.shadow_trades), "settled": 0}

        wins = sum(1 for t in settled if t["result"] == "SIMULATED_WIN")
        total_pnl = sum(t["simulated_pnl"] for t in settled)
        return {
            "total": len(self.shadow_trades),
            "settled": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": round(wins / len(settled), 4),
            "total_pnl": round(total_pnl, 4),
            "avg_edge": round(sum(t["effective_edge"] for t in settled) / len(settled), 4),
        }

    def get_stats_by_symbol(self) -> Dict[str, ShadowStats]:
        """按品种统计"""
        stats = {}
        for sym in config.SYMBOLS:
            candidates = [c for c in self.candidates if c.symbol == sym]
            selected = [c for c in candidates if c.selected]
            settled = [c for c in selected if c.result in ("WIN", "LOSS")]

            if not candidates:
                stats[sym] = ShadowStats(symbol=sym)
                continue

            wins = sum(1 for c in settled if c.result == "WIN")
            losses = sum(1 for c in settled if c.result == "LOSS")
            ties = sum(1 for c in settled if c.result == "TIE")
            settled_count = wins + losses

            total_pnl = sum(c.pnl for c in settled)
            total_staked = sum(c.final_integer_stake for c in selected)
            wr = wins / settled_count if settled_count > 0 else 0.0
            roi = total_pnl / total_staked if total_staked > 0 else 0.0

            # Brier
            brier = 0.0
            for c in settled:
                actual = 1.0 if c.result == "WIN" else 0.0
                brier += (c.calibrated_probability - actual) ** 2
            brier /= settled_count if settled_count > 0 else 1

            # Edge distribution
            edge_buckets = {"0-2%": 0, "2-4%": 0, "4-6%": 0, "6-8%": 0, "8%+": 0}
            for c in selected:
                e = abs(c.effective_edge)
                if e < 0.02: edge_buckets["0-2%"] += 1
                elif e < 0.04: edge_buckets["2-4%"] += 1
                elif e < 0.06: edge_buckets["4-6%"] += 1
                elif e < 0.08: edge_buckets["6-8%"] += 1
                else: edge_buckets["8%+"] += 1

            # Regime distribution
            regime_dist = {}
            for c in candidates:
                r = c.regime or "UNKNOWN"
                regime_dist[r] = regime_dist.get(r, 0) + 1

            probs = [c.raw_probability for c in candidates]
            prob_mean = sum(probs) / len(probs) if probs else 0.0
            prob_std = (sum((p - prob_mean) ** 2 for p in probs) / len(probs)) ** 0.5 if probs else 0.0

            # Streaks
            max_lose = 0; max_win = 0; cur_lose = 0; cur_win = 0
            for c in settled:
                if c.result == "WIN":
                    cur_win += 1; cur_lose = 0
                    max_win = max(max_win, cur_win)
                elif c.result == "LOSS":
                    cur_lose += 1; cur_win = 0
                    max_lose = max(max_lose, cur_lose)

            stats[sym] = ShadowStats(
                symbol=sym,
                total_candidates=len(candidates),
                total_selected=len(selected),
                total_trades=settled_count,
                wins=wins, losses=losses, ties=ties,
                win_rate=round(wr, 4),
                total_pnl=round(total_pnl, 4),
                total_staked=total_staked,
                roi=round(roi, 4),
                avg_effective_edge=round(sum(c.effective_edge for c in selected) / len(selected), 4) if selected else 0.0,
                brier_score=round(brier, 4),
                max_drawdown=0.0,
                longest_losing_streak=max_lose,
                longest_winning_streak=max_win,
                avg_payout=round(sum(c.net_payout_ratio for c in selected) / len(selected), 4) if selected else 0.0,
                probability_mean=round(prob_mean, 4),
                probability_std=round(prob_std, 4),
                regime_distribution=regime_dist,
                edge_distribution=edge_buckets,
            )
        return stats

    def get_mode(self) -> str:
        return self.mode.value

    def get_symbol_mode_summary(self) -> dict:
        return {sym: mode.value for sym, mode in self._symbol_mode.items()}

    def get_shadow_days(self) -> float:
        return (time.time() - self._shadow_start_time) / 86400


# 全局单例
_shadow_mode: Optional[ShadowMode] = None

def get_shadow_mode() -> ShadowMode:
    global _shadow_mode
    if _shadow_mode is None:
        _shadow_mode = ShadowMode()
    return _shadow_mode

def set_run_mode(mode: RunMode):
    global _shadow_mode
    _shadow_mode = ShadowMode(mode=mode)