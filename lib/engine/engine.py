# -*- coding: utf-8 -*-
"""
ATradeBot 引擎 v4 — EventEdge V2 全流水线

核心流水线:
  Contract Discovery → Market Data → Feature Engine → Regime Detection
  → Expert Models (3x) → Meta Model → Probability Calibration
  → Dynamic Break-even → Edge Calculation → Uncertainty Filter
  → Opportunity Ranking → Portfolio Risk (Integer Kelly) → Order Execution

旧版 v3 流水线已被替换:
  SignalValidator(L0-L2) → RiskManager(L3-L5) → AI → OrderExecutor

支持三种模式: BACKTEST / SHADOW / LIVE
"""
import time
import json
import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from . import config
from .exchange import fetch_balance, place_order
from .notifier import notify_trade, notify_result
from .models import Prediction, TradeSignal, GateResult, EdgeResult, MarketRegime, EnsemblePrediction, ExpertPrediction
from .order_executor import OrderExecutor

# ── EventEdge V2 imports ──
from .regime_detector import MarketRegimeDetector, get_regime_detector
from .experts import ExpertManager, get_expert_manager
from .edge_engine import EdgeEngine, get_edge_engine
from .uncertainty import ModelUncertaintyEstimator, get_uncertainty_estimator
from .portfolio_risk import PortfolioRiskManager, get_portfolio_risk
from .opportunity_ranker import OpportunityRanker, get_opportunity_ranker
from .shadow_mode import ShadowMode, RunMode, get_shadow_mode, set_run_mode, CalibrationStatus, SettlementSource, ShadowCandidateRecord
from .trade_ledger import TradeLedger, TradeRecord, get_trade_ledger
from .settlement_ledger import SettlementLedger, get_settlement_ledger, RejectReason
from .model_health import ModelHealthMonitor, get_model_health_monitor
from .probability_calibrator import WalkForwardCalibrator


def emit(event_type: str, payload: dict):
    event = {"type": event_type, "ts": int(time.time() * 1000), "payload": payload}
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _calc_30m_features(data_feat: pd.DataFrame) -> dict | None:
    """特征计算 (在 FEATURE_INTERVAL_MIN 聚合K线上计算, 保持与训练特征一致)"""
    d = data_feat.copy()
    eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)
    d["ret_1"] = d["close"].pct_change(1).fillna(0)
    d["ret_3"] = d["close"].pct_change(3).fillna(0)
    d["ret_6"] = d["close"].pct_change(6).fillna(0)
    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    d["MACD"] = (2 * (macd - macd.ewm(span=9, adjust=False).mean())).fillna(0)
    d["MACD_hist"] = d["MACD"].fillna(0)
    delta = d["close"].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = (100 - (100 / (1 + gain / loss))).fillna(50)
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std().fillna(0)
    d["BB_Pos"] = ((d["close"] - (mid - 2 * std)) / (4 * std + eps)).clip(0, 1)
    d["BB_width"] = (((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)).fillna(0)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"] - d["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR_pct"] = (tr.rolling(14).mean() / (d["close"] + eps)).fillna(0)
    up = d["high"] - d["high"].shift(1)
    dn = d["low"].shift(1) - d["low"]
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=d.index)
    tr14 = tr.rolling(14).sum().replace(0, eps)
    pdi = 100 * pdm.rolling(14).sum() / tr14
    ndi = 100 * ndm.rolling(14).sum() / tr14
    d["ADX"] = (100 * abs(pdi - ndi) / (pdi + ndi + eps)).rolling(14).mean().fillna(20)
    d["MA10"] = d["close"].rolling(10).mean().bfill()
    d["MA20"] = d["close"].rolling(20).mean().bfill()
    d["MA50"] = d["close"].rolling(50).mean().bfill()
    d["price_vs_MA20"] = ((d["close"] - d["MA20"]) / (d["MA20"] + eps)).fillna(0)
    d["price_vs_MA50"] = ((d["close"] - d["MA50"]) / (d["MA50"] + eps)).fillna(0)
    d["MA_trend"] = np.sign(d["MA10"] - d["MA20"]).fillna(0)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vwap = (d["volume"] * tp).cumsum() / (d["volume"].cumsum() + eps)
    d["VWAP_dist"] = ((d["close"] - vwap) / (vwap + eps)).fillna(0)
    d["vol_ratio"] = (d["volume"] / (d["volume"].rolling(5).mean() + eps)).fillna(1)
    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"] * obv_dir).cumsum()
    d["OBV_trend"] = np.sign(obv - obv.shift(5)).fillna(0)
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["CCI"] = ((tp - tp_sma) / (0.015 * tp_mad + eps)).fillna(0)
    atr14 = tr.rolling(14).sum()
    d["CHOP"] = (100 * np.log10(atr14 / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + eps)) / np.log10(14)).fillna(50)
    body = (d["close"] - d["open"]).abs()
    d["body_pct"] = (body / (d["high"] - d["low"] + eps)).fillna(0.5)
    d["is_green"] = (d["close"] > d["open"]).astype(int)
    d["volatility_ratio"] = d["ATR_pct"] / (d["ATR_pct"].rolling(20).mean() + eps).fillna(1)
    result = d.replace([np.inf, -np.inf], np.nan).dropna()
    return result.iloc[-1].to_dict() if not result.empty else None


@dataclass
class CandidateOpportunity:
    """候选交易机会 — 从各品种生成后统一进入 Ranker 排序"""
    symbol: str = ""
    direction_str: str = ""
    direction_int: int = 0
    current_price: float = 0.0
    current_bar_ts: object = None
    current_ts: int = 0
    # Pipeline outputs
    regime: Optional[MarketRegime] = None
    ensemble: Optional[EnsemblePrediction] = None
    predictions: list = field(default_factory=list)
    edge: Optional[EdgeResult] = None
    indicators: dict = field(default_factory=dict)
    row: dict = field(default_factory=dict)
    # Will be populated after ranking
    position: Optional[object] = None  # PositionSizeResult


class TradingEngine:
    def __init__(self, run_mode: str = "live"):
        self.executor = OrderExecutor()
        self.reset_state()

        # ── EventEdge V2 模块 ──
        self.regime_detector = get_regime_detector()
        self.expert_manager = get_expert_manager()
        self.edge_engine = get_edge_engine()
        self.uncertainty = get_uncertainty_estimator()
        self.portfolio_risk = get_portfolio_risk()
        self.ranker = get_opportunity_ranker()
        self.trade_ledger = get_trade_ledger()
        self.settlement_ledger = get_settlement_ledger()
        self.model_health = get_model_health_monitor()
        self.calibrator = WalkForwardCalibrator(method="isotonic", min_samples=50)

        # 运行模式
        mode = RunMode.LIVE if run_mode == "live" else (RunMode.SHADOW if run_mode == "shadow" else RunMode.BACKTEST)
        self.shadow = ShadowMode(mode=mode)

        # 健康追踪
        self._health_trades: list = []  # [{"predicted_prob": float, "result": str, "pnl": float}, ...]
        self._last_health_check = time.time()

    def set_run_mode(self, mode: str):
        """设置运行模式: live / shadow / backtest"""
        m = RunMode.LIVE if mode == "live" else (RunMode.SHADOW if mode == "shadow" else RunMode.BACKTEST)
        self.shadow = ShadowMode(mode=m)

    def reset_state(self):
        self.running = False
        self.paused = False
        self.balance = 0.0
        self.start_balance = 0.0
        self.active_trades: list[dict] = []
        self.last_trade_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_reject_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_signal_bar_ts: dict[str, object] = {s: None for s in config.SYMBOLS}
        self.recent_results: list[bool] = []
        self.consecutive_losses = 0
        self.halted = False
        self.pause_until = 0
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        self._warmup_done: dict[str, bool] = {s: False for s in config.SYMBOLS}
        self.executor = OrderExecutor()

    def start(self):
        self.reset_state()
        self.running = True
        run_mode = self.shadow.get_mode()

        # ── Calibration Status Check ──
        cal_ready = self.calibrator.is_ready()
        cal_status = "READY" if cal_ready else "NOT_READY"
        if not cal_ready:
            emit("calibration_status", {"status": "NOT_READY", "msg": "PASSTHROUGH_UNCALIBRATED"})

        # ── LIVE Gate Check ──
        live_gate = self.shadow.get_live_gate_status(cal_ready)
        if self.shadow.is_live_allowed() and not live_gate["passed"]:
            emit("log", {"msg": f"LIVE_VALIDATION_GATE_NOT_PASSED: {'; '.join(live_gate['reasons'])}"})
            emit("log", {"msg": "LIVE 被拒绝，引擎将降级至 SHADOW 模式"})
            set_run_mode(RunMode.SHADOW)
            self.shadow = ShadowMode(mode=RunMode.SHADOW)
            run_mode = "SHADOW"

        # ── Per-symbol activation summary ──
        sym_modes = self.shadow.get_symbol_mode_summary()
        active = [s for s, m in sym_modes.items() if m == "SHADOW_ACTIVE"]
        observe = [s for s, m in sym_modes.items() if m == "OBSERVE_ONLY"]
        disabled = [s for s, m in sym_modes.items() if m == "DISABLED"]

        emit("status", {
            "state": "running", "run_mode": run_mode,
            "msg": f"EventEdge V2 引擎启动 ({run_mode})",
            "calibration": cal_status,
            "live_gate": live_gate,
            "symbol_modes": sym_modes,
        })

        self.balance = fetch_balance()
        if self.balance < 0:
            self.balance = 0.0

        # ── SHADOW 模式: 使用模拟余额 (不影响真实账户) ──
        if run_mode == "SHADOW" and self.balance < config.MIN_ORDER_USD:
            shadow_equity = max(500.0, config.MIN_ORDER_USD * 10)
            emit("log", {"msg": f"SHADOW 模式: 真实余额 {self.balance:.2f}U < 最低 {config.MIN_ORDER_USD}U, "
                         f"使用模拟余额 {shadow_equity:.0f}U 进行 Shadow 交易"})
            self.balance = shadow_equity
            self._shadow_simulated_balance = True
        else:
            self._shadow_simulated_balance = False

        self.start_balance = self.balance
        self.start_balance = self.balance
        emit("balance_update", {"balance": self.balance})
        self._warmup_done = {s: False for s in config.SYMBOLS}

        can_trade, msg = self.portfolio_risk.check_account_size(self.balance)
        if not can_trade:
            emit("log", {"msg": f"账户风险: {msg}"})

        emit("log", {
            "msg": f"EventEdge V2 启动! 余额{self.balance:.2f}U | "
                   f"模式{run_mode} | 底仓{config.MIN_ORDER_USD}U | "
                   f"MaxBet{config.MAX_BET_FRACTION:.0%} | "
                   f"Kelly{config.KELLY_FRACTION:.0%} | "
                   f"MinEdge{config.MIN_EFFECTIVE_EDGE:.1%} | "
                   f"Calibration:{cal_status} | "
                   f"Active:{','.join(active) if active else 'none'} | "
                   f"Observe:{','.join(observe) if observe else 'none'} | "
                   f"LIVE:{'ENABLED' if live_gate['passed'] else 'DISABLED'}"
        })

    def stop(self):
        self.running = False
        self.paused = False
        emit("status", {"state": "stopped"})

    def pause(self):
        self.paused = True
        emit("status", {"state": "paused"})

    def resume(self):
        self.paused = False
        self.halted = False
        self.pause_until = 0
        self.consecutive_losses = 0
        emit("status", {"state": "running"})

    # ═══════════════════════════════════════════════════════════
    # EventEdge V2 核心流水线
    # ═══════════════════════════════════════════════════════════

    def _generate_candidate_v2(self, symbol: str, full_df: pd.DataFrame, current_ts: int) -> Optional[CandidateOpportunity]:
        """
        EventEdge V2 Stage 1-5: 生成候选交易机会
        Regime → Experts → Meta → Uncertainty → Edge → Candidate

        返回 CandidateOpportunity 或 None（如果任何阶段不通过）
        注意：此阶段不执行下单，也不做 Portfolio Risk 检查
        """
        # ── 全局风控 ──
        if self.halted or current_ts < self.pause_until:
            return None
        if self.consecutive_losses >= config.CONSECUTIVE_LOSS_HALT:
            self.halted = True
            emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 暂停!"})
            return None

        # ── 数据准备 ──
        df_s = full_df[full_df["symbol"] == symbol].copy()
        if len(df_s) < 200:
            return None
        df_s.set_index("datetime", inplace=True)

        candle_min = config.CANDLE_INTERVAL_MIN
        detect_data = df_s.resample(f"{candle_min}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(detect_data) < 10:
            return None

        current_bar_ts = detect_data.index[-1]

        if not self._warmup_done.get(symbol, False):
            self.last_signal_bar_ts[symbol] = current_bar_ts
            self._warmup_done[symbol] = True
            emit("log", {"msg": f"预热完成 {symbol}: {current_bar_ts}"})
            return None

        if self.last_signal_bar_ts.get(symbol) == current_bar_ts:
            return None

        feat_min = config.FEATURE_INTERVAL_MIN
        feature_data = df_s.resample(f"{feat_min}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(feature_data) < 200:
            return None

        row = _calc_30m_features(feature_data)
        if row is None:
            return None

        # ── 构建指标字典 ──
        indicators = {
            "ADX": float(row.get("ADX", 20)),
            "RSI": float(row.get("RSI", 50)),
            "BB_Pos": float(row.get("BB_Pos", 0.5)),
            "bb_width": float(row.get("BB_width", 0.02)),
            "volatility_ratio": float(row.get("volatility_ratio", 1.0)),
            "ATR_pct": float(row.get("ATR_pct", 0.003)),
            "price_vs_MA20": float(row.get("price_vs_MA20", 0)),
            "MACD": float(row.get("MACD", 0)),
            "MA_trend": float(row.get("MA_trend", 0)),
            "VWAP_dist": float(row.get("VWAP_dist", 0)),
            "vol_ratio": float(row.get("vol_ratio", 1.0)),
            "CCI": float(row.get("CCI", 0)),
        }
        current_price = float(row.get("close", 0))

        emit("features", {"symbol": symbol, "indicators": indicators})

        # ═══════════════════════════════════════════════════
        # Stage 1: Regime Detection
        # ═══════════════════════════════════════════════════
        regime = self.regime_detector.detect(indicators)
        emit("regime_update", {
            "symbol": symbol,
            "regime": regime.regime,
            "confidence": regime.confidence,
            "adx": regime.adx,
            "volatility": regime.volatility,
        })

        # ═══════════════════════════════════════════════════
        # Stage 2: Expert Models
        # ═══════════════════════════════════════════════════
        predictions = self.expert_manager.predict_all(symbol, indicators, row)
        emit("expert_votes", {
            "symbol": symbol,
            "votes": {p.expert_name: {"prob": p.raw_probability, "dir": p.direction_str}
                       for p in predictions},
        })

        # ═══════════════════════════════════════════════════
        # Stage 3: Meta Model
        # ═══════════════════════════════════════════════════
        ensemble = self.expert_manager.ensemble(predictions, regime)
        emit("prediction", {
            "symbol": symbol,
            "prob_long": ensemble.ensemble_probability,
            "direction": 1 if ensemble.direction == 1 else 2,
            "prob_win": ensemble.ensemble_probability,
            "is_reversal": False,
            "expert_votes": ensemble.expert_votes,
        })

        # ═══════════════════════════════════════════════════
        # Stage 4: Uncertainty
        # ═══════════════════════════════════════════════════
        uncertainty_margin = self.uncertainty.estimate_expert_uncertainty(predictions)

        # ═══════════════════════════════════════════════════
        # Stage 5: Edge Calculation
        # ═══════════════════════════════════════════════════
        direction_str = "CALL" if ensemble.direction == 1 else "PUT"
        edge = self.edge_engine.compute(
            symbol=symbol,
            calibrated_probability=ensemble.ensemble_probability,
            direction=direction_str,
            direction_int=ensemble.direction,
            expiry_minutes=config.HOLD_MINUTES,
            entry_price=current_price,
            uncertainty_margin=uncertainty_margin,
            regime=regime.regime,
            expert_votes=ensemble.expert_votes,
        )
        emit("edge_calculation", {
            "symbol": symbol,
            "calibrated_probability": edge.calibrated_probability,
            "conservative_probability": edge.conservative_probability,
            "break_even_probability": edge.break_even_probability,
            "effective_edge": edge.effective_edge,
            "expected_roi": edge.expected_roi,
            "passed": edge.passed,
            "reject_reason": edge.reject_reason,
        })

        if not edge.passed:
            reject_reason = edge.reject_reason or "LOW_EDGE"
            emit("trade_rejected", {
                "symbol": symbol,
                "reason": reject_reason,
                "edge": edge.effective_edge,
                "roi": edge.expected_roi,
            })
            # 记录拒绝到 Ledger
            rec = self.trade_ledger.create_record(
                symbol=symbol, direction=direction_str, direction_int=ensemble.direction,
                entry_time_ms=current_ts, entry_price=current_price,
                stake_usd=0, expiry_minutes=config.HOLD_MINUTES,
                raw_probability=ensemble.ensemble_probability,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                regime=regime.regime,
                expert_votes=ensemble.expert_votes,
                reject_reason=reject_reason,
                reject_detail=f"edge={edge.effective_edge:.4f}, roi={edge.expected_roi:.4f}",
            )
            self.trade_ledger.save(rec)
            return None

        return CandidateOpportunity(
            symbol=symbol,
            direction_str=direction_str,
            direction_int=ensemble.direction,
            current_price=current_price,
            current_bar_ts=current_bar_ts,
            current_ts=current_ts,
            regime=regime,
            ensemble=ensemble,
            predictions=predictions,
            edge=edge,
            indicators=indicators,
            row=row,
        )

    def _execute_opportunity(self, candidate: CandidateOpportunity) -> bool:
        """
        EventEdge V2 Stage 6-8: 对已排序选中的候选机会执行下单
        Portfolio Risk → Model Health → Order Execution / Shadow Record

        Returns True if order was placed (or shadow recorded), False otherwise.
        """
        symbol = candidate.symbol
        direction_str = candidate.direction_str
        direction_int = candidate.direction_int
        edge = candidate.edge
        ensemble = candidate.ensemble
        regime = candidate.regime
        indicators = candidate.indicators
        current_price = candidate.current_price
        current_ts = candidate.current_ts
        current_bar_ts = candidate.current_bar_ts

        # ═══════════════════════════════════════════════════
        # Stage 6: Portfolio Risk + Integer Kelly
        # ═══════════════════════════════════════════════════
        pos = self.portfolio_risk.compute_position_size(edge, self.balance)

        if not pos.allowed:
            emit("trade_rejected", {
                "symbol": symbol,
                "reason": pos.reject_reason,
                "bet_fraction": pos.bet_fraction,
            })
            rec = self.trade_ledger.create_record(
                symbol=symbol, direction=direction_str, direction_int=direction_int,
                entry_time_ms=current_ts, entry_price=current_price,
                stake_usd=pos.stake_usd, expiry_minutes=config.HOLD_MINUTES,
                raw_probability=ensemble.ensemble_probability,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                regime=regime.regime,
                expert_votes=ensemble.expert_votes,
                reject_reason=pos.reject_reason,
                reject_detail=f"stake={pos.stake_usd}, fraction={pos.bet_fraction:.4f}",
            )
            self.trade_ledger.save(rec)
            return False

        stake_usd = pos.stake_usd

        # ═══════════════════════════════════════════════════
        # Stage 6b: Portfolio Limits Check
        # ═══════════════════════════════════════════════════
        active_positions = {
            t.get("symbol", ""): [{"direction": t.get("direction_str", direction_str), "stake_usd": t.get("amount", 0)}]
            for t in self.active_trades
        }
        port_check = self.portfolio_risk.check_portfolio_limits(
            symbol=symbol,
            direction=direction_str,
            stake_usd=stake_usd,
            equity=self.balance,
            active_positions=active_positions,
        )
        if not port_check.allowed:
            emit("trade_rejected", {
                "symbol": symbol,
                "reason": port_check.reject_reason,
                "detail": f"exposure={port_check.current_exposure:.2f}",
            })
            rec = self.trade_ledger.create_record(
                symbol=symbol, direction=direction_str, direction_int=direction_int,
                entry_time_ms=current_ts, entry_price=current_price,
                stake_usd=stake_usd, expiry_minutes=config.HOLD_MINUTES,
                raw_probability=ensemble.ensemble_probability,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                regime=regime.regime,
                expert_votes=ensemble.expert_votes,
                reject_reason=port_check.reject_reason,
                reject_detail=f"exposure={port_check.current_exposure:.2f}",
            )
            self.trade_ledger.save(rec)
            return False

        # ═══════════════════════════════════════════════════
        # Stage 7: Model Health Check
        # ═══════════════════════════════════════════════════
        if len(self._health_trades) >= 50:
            health = self.model_health.check(self._health_trades)
            emit("model_health", {
                "is_degraded": health.is_degraded,
                "actual_win_rate": health.actual_win_rate,
                "predicted_win_rate": health.predicted_win_rate,
                "win_rate_delta": health.win_rate_delta,
                "brier_score": health.brier_score,
                "window": health.window,
            })
            if health.is_degraded and self.shadow.is_live_allowed():
                emit("log", {"msg": f"⚠️ 模型退化! {health.degradation_reason} — 切换到 SHADOW"})
                set_run_mode(RunMode.SHADOW)
                self.shadow = ShadowMode(mode=RunMode.SHADOW)

        # ═══════════════════════════════════════════════════
        # Stage 8: Order Execution (LIVE) or Shadow Record
        # ═══════════════════════════════════════════════════
        # ── Per-symbol activation check ──
        if not self.shadow.is_shadow_active(symbol) and not self.shadow.can_place_order():
            if self.shadow.is_observe_only(symbol):
                emit("shadow_trade", {
                    "symbol": symbol, "direction": direction_str,
                    "entryPrice": current_price, "amount": 0,
                    "effectiveEdge": edge.effective_edge,
                    "expectedRoi": edge.expected_roi,
                    "regime": regime.regime,
                    "mode": "OBSERVE_ONLY",
                })
                emit("log", {"msg": f"[OBSERVE] {symbol} {direction_str} "
                             f"Edge={edge.effective_edge:.2%} ROI={edge.expected_roi:.2%} (no trade)"})
            return True

        if self.shadow.can_place_order():
            # ── LIVE 模式: 真实下单 ──
            result = place_order(symbol, direction_int, stake_usd, config.HOLD_MINUTES)
            if result.ok:
                # 先保存到 TradeLedger 获取唯一 trade_id
                rec = self.trade_ledger.create_record(
                    symbol=symbol, direction=direction_str, direction_int=direction_int,
                    entry_time_ms=current_ts, entry_price=current_price,
                    stake_usd=stake_usd, expiry_minutes=config.HOLD_MINUTES,
                    raw_probability=ensemble.ensemble_probability,
                    calibrated_probability=edge.calibrated_probability,
                    break_even_probability=edge.break_even_probability,
                    effective_edge=edge.effective_edge,
                    expected_roi=edge.expected_roi,
                    net_payout_ratio=edge.net_payout_ratio,
                    payout_source=edge.payout_source,
                    regime=regime.regime,
                    expert_votes=ensemble.expert_votes,
                )
                self.trade_ledger.save(rec)

                # 记录 active_trade 时附带 trade_id（贯穿全链路）
                self.active_trades.append({
                    "symbol": symbol, "dir": direction_int,
                    "direction_str": direction_str,
                    "start_ts": current_ts, "amount": stake_usd,
                    "entry": current_price, "pre_balance": self.balance,
                    "trade_id": rec.trade_id,
                })
                self.last_trade_ts[symbol] = current_ts
                self.last_signal_bar_ts[symbol] = current_bar_ts

                emit("trade_executed", {
                    "symbol": symbol, "direction": direction_str,
                    "trade_id": rec.trade_id,
                    "entryPrice": current_price, "amount": stake_usd,
                    "rawProbability": ensemble.ensemble_probability,
                    "calibratedProbability": edge.calibrated_probability,
                    "effectiveEdge": edge.effective_edge,
                    "expectedRoi": edge.expected_roi,
                    "balance": self.balance,
                    "regime": regime.regime,
                })
                notify_trade(
                    symbol, f"做多(CALL)" if direction_int == 1 else f"做空(PUT)",
                    current_price, stake_usd,
                    ensemble.ensemble_probability, indicators,
                    f"Edge={edge.effective_edge:.2%} ROI={edge.expected_roi:.2%}",
                    self.balance, len(self.active_trades), False,
                )

                self.balance = fetch_balance()
                if self.balance < 0:
                    self.balance = self.balance - stake_usd
                emit("balance_update", {"balance": self.balance})
                return True
            else:
                emit("trade_rejected", {
                    "symbol": symbol, "reason": "ORDER_FAILED",
                    "detail": result.msg,
                })
                return False
        else:
            # ── SHADOW_ACTIVE: 模拟下单 ──
            self.shadow.record_shadow_trade(
                symbol=symbol, direction=direction_str, direction_int=direction_int,
                entry_time_ms=current_ts, entry_price=current_price,
                stake_usd=stake_usd, expiry_minutes=config.HOLD_MINUTES,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                regime=regime.regime,
                expert_votes=ensemble.expert_votes,
            )
            emit("shadow_trade", {
                "symbol": symbol, "direction": direction_str,
                "entryPrice": current_price, "amount": stake_usd,
                "effectiveEdge": edge.effective_edge,
                "expectedRoi": edge.expected_roi,
                "regime": regime.regime,
                "mode": "SHADOW_ACTIVE",
            })
            emit("log", {"msg": f"[SHADOW] {symbol} {direction_str} "
                         f"{stake_usd}U Edge={edge.effective_edge:.2%} ROI={edge.expected_roi:.2%}"})
            return True

    def _check_settlement(self):
        """
        通过时间推算判断订单是否已结算。

        结算后:
        1. 通过 trade_id 精确匹配 TradeLedger 记录
        2. 从 TradeLedger 读取真实概率（不再硬编码 0.50）
        3. 更新 ModelHealth + Expert 表现
        """
        if not self.active_trades:
            return

        current_balance = fetch_balance()
        if current_balance < 0:
            return

        # ── SHADOW 模拟余额: 不要用真实余额做结算 ──
        if getattr(self, '_shadow_simulated_balance', False):
            current_balance = self.balance

        for i in range(len(self.active_trades) - 1, -1, -1):
            t = self.active_trades[i]
            elapsed_ms = time.time() * 1000 - t["start_ts"]
            settle_threshold_ms = config.HOLD_MINUTES * 60000 + 30000

            if elapsed_ms < settle_threshold_ms:
                continue

            pnl = current_balance - t["pre_balance"]
            is_win = pnl > 0
            is_tie = abs(pnl) < 0.001

            if is_win:
                self.total_wins += 1
            elif not is_tie:
                self.total_losses += 1
            self.total_pnl += pnl
            if not is_tie:
                self._record_result(is_win)
            result = "tie" if is_tie else ("win" if is_win else "loss")

            emit("trade_result", {
                "symbol": t["symbol"], "result": result, "pnl": round(pnl, 4),
                "entryPrice": t["entry"], "dir": t["dir"],
                "trade_id": t.get("trade_id", ""),
            })
            notify_result(t["symbol"], is_win, pnl)

            # ── 从 TradeLedger 读取真实概率 ──
            trade_id = t.get("trade_id", "")
            raw_prob = 0.50
            cal_prob = 0.50
            cons_prob = 0.50
            eff_edge = 0.0
            if trade_id:
                # 从 TradeLedger 查询该笔交易的概率
                records = self.trade_ledger.query()
                for rec in records:
                    if rec.trade_id == trade_id:
                        raw_prob = rec.raw_probability
                        cal_prob = rec.calibrated_probability
                        cons_prob = rec.conservative_probability
                        eff_edge = rec.effective_edge
                        break

            # ── 更新 SettlementLedger（使用精确 trade_id）──
            self.settlement_ledger.estimate_settlement(
                trade_id=trade_id if trade_id else "",
                current_balance=current_balance,
                pre_balance=t["pre_balance"],
            )

            # ── 更新 Expert 表现 ──
            direction_str = t.get("direction_str", "")
            if direction_str:
                win_result = "WIN" if is_win else ("TIE" if is_tie else "LOSS")
                for expert_name in ["trend", "mean_reversion", "volatility_breakout"]:
                    if win_result in ("WIN", "LOSS"):
                        self.expert_manager.update_performance(
                            expert_name, is_win=(win_result == "WIN")
                        )

            # ── 更新健康追踪（使用真实概率）──
            self._health_trades.append({
                "predicted_prob": cal_prob,       # 校准后概率
                "raw_probability": raw_prob,      # 原始概率
                "calibrated_probability": cal_prob,
                "conservative_probability": cons_prob,
                "effective_edge": eff_edge,
                "result": "TIE" if is_tie else ("WIN" if is_win else "LOSS"),
                "pnl": pnl,
                "trade_id": trade_id,
            })
            if len(self._health_trades) > 500:
                self._health_trades = self._health_trades[-500:]

            # 更新 Portfolio PnL
            self.portfolio_risk.record_pnl(pnl)

            self.active_trades.pop(i)
            self.balance = current_balance
            emit("balance_update", {"balance": current_balance})
            emit("log", {"msg": f"结算 {t['symbol']}: {'平' if is_tie else ('赢' if is_win else '输')} {abs(pnl):.2f}U | 余额→{current_balance:.2f}U"})

            # 通知 OrderExecutor
            self.executor.on_settlement(t["symbol"])

            # ── 定期健康检查（多窗口）──
            if time.time() - self._last_health_check > 300:
                multi = self.model_health.check_multi_window(self._health_trades)
                for window, health in multi.items():
                    emit("model_health", {
                        "is_degraded": health.is_degraded,
                        "actual_win_rate": health.actual_win_rate,
                        "predicted_win_rate": health.predicted_win_rate,
                        "win_rate_delta": health.win_rate_delta,
                        "brier_score": health.brier_score,
                        "ece": health.expected_calibration_error,
                        "window": health.window,
                        "trade_count": health.trade_count,
                        "degradation_reason": health.degradation_reason,
                    })
                self._last_health_check = time.time()

    def _record_result(self, is_win: bool):
        self.recent_results.append(is_win)
        if len(self.recent_results) > config.RECENT_WINDOW:
            self.recent_results = self.recent_results[-config.RECENT_WINDOW:]
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.CONSECUTIVE_LOSS_HALT:
                self.halted = True
                emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 暂停! 需手动恢复"})
            elif self.consecutive_losses >= 3:
                pause_ms = config.CONSECUTIVE_LOSS_PAUSE_SEC * 1000
                self.pause_until = int(time.time() * 1000) + pause_ms
                emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 冷冻{pause_ms//1000}秒"})

    def tick(self):
        """
        EventEdge V2 主循环:

        1. _check_settlement()
        2. 加载数据 (优先 Gate.io 实时 API, 回退 CSV)
        3. 发送 K 线
        4. 对每个品种: _generate_candidate_v2() → 收集候选
        5. 所有候选 → OpportunityRanker.rank() → 排序选择
        6. 对选中的候选: _execute_opportunity() → 下单
        """
        current_ts = int(time.time() * 1000)

        try:
            self._check_settlement()

            shadow_symbols = list(self.shadow.get_symbol_mode_summary().keys())

            # ── 数据加载: 优先 Gate.io 实时 API, 回退 CSV ──
            from .live_data import get_live_data_for_engine, build_tick_dataframe, load_csv_fallback

            live_data = get_live_data_for_engine(
                shadow_symbols, interval="15m", limit=200,
                csv_fallback=config.CSV_FILE,
            )

            if live_data:
                full_df = build_tick_dataframe(live_data)
            else:
                # 回退到 CSV
                full_df = load_csv_fallback(config.CSV_FILE)

            if full_df is None or len(full_df) < 50:
                return

            # 发送最新 K 线
            for s in shadow_symbols:
                ds = full_df[full_df["symbol"] == s]
                if not ds.empty:
                    last = ds.iloc[-1]
                    emit("candle_update", {
                        "symbol": s, "ts": int(last["ts"]),
                        "open": float(last["open"]), "high": float(last["high"]),
                        "low": float(last["low"]), "close": float(last["close"]),
                        "volume": float(last["volume"]) if len(last) > 6 else 0,
                    })

            # ═══════════════════════════════════════════════════
            # Phase 1: Generate Candidates (all symbols, including OBSERVE_ONLY)
            # ═══════════════════════════════════════════════════
            candidates: list[CandidateOpportunity] = []
            for s in shadow_symbols:
                sym_mode = self.shadow.get_symbol_mode(s)
                if sym_mode.value == "DISABLED":
                    continue
                candidate = self._generate_candidate_v2(s, full_df, current_ts)
                if candidate is not None:
                    candidates.append(candidate)

            if not candidates:
                return

            # ── Record ALL candidates (including rejected) to Shadow ──
            cal_ready = self.calibrator.is_ready()
            cal_status = CalibrationStatus.READY.value if cal_ready else CalibrationStatus.PASSTHROUGH_UNCALIBRATED.value
            for c in candidates:
                edge = c.edge
                ensemble = c.ensemble
                regime = c.regime
                is_tradeable = self.shadow.is_shadow_active(c.symbol) or self.shadow.can_place_order()
                is_observe = self.shadow.is_observe_only(c.symbol)

                record = ShadowCandidateRecord(
                    record_id=f"sc_{int(time.time()*1000)}_{c.symbol}_{c.direction_str}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    symbol=c.symbol,
                    direction=c.direction_str,
                    direction_int=c.direction_int,
                    expiry_minutes=config.HOLD_MINUTES,
                    regime=regime.regime if regime else "",
                    regime_confidence=regime.confidence if regime else 0.0,
                    raw_probability=round(ensemble.ensemble_probability, 4) if ensemble else 0.0,
                    ensemble_probability=round(ensemble.ensemble_probability, 4) if ensemble else 0.0,
                    calibrated_probability=round(edge.calibrated_probability, 4) if edge else 0.0,
                    calibration_status=cal_status,
                    conservative_probability=round(edge.conservative_probability, 4) if edge else 0.0,
                    uncertainty_margin=round(edge.uncertainty_margin, 4) if edge else 0.0,
                    calibration_margin=round(edge.calibration_margin, 4) if edge else 0.0,
                    model_degradation_margin=round(edge.model_degradation_margin, 4) if edge else 0.0,
                    payout_ratio=round(edge.payout_ratio, 4) if edge else 0.0,
                    net_payout_ratio=round(edge.net_payout_ratio, 4) if edge else 0.0,
                    payout_source=edge.payout_source if edge else "hardcoded",
                    payout_verified=False,
                    break_even_probability=round(edge.break_even_probability, 4) if edge else 0.0,
                    probability_edge=round(edge.probability_edge, 4) if edge else 0.0,
                    expected_roi=round(edge.expected_roi, 4) if edge else 0.0,
                    effective_edge=round(edge.effective_edge, 4) if edge else 0.0,
                    risk_adjusted_ev=0.0,
                    rank_score=0.0,
                    rank=0,
                    selected=False,
                    stake_before_integer_rounding=0.0,
                    final_integer_stake=0,
                    bet_fraction=0.0,
                    kelly_fraction=0.0,
                    reject_reason=edge.reject_reason if edge and not edge.passed else "",
                    reject_detail="",
                    portfolio_check_passed=False,
                    daily_stop_ok=True,
                    exposure_ok=True,
                    entry_reference_price=c.current_price,
                    expiry_reference_price=None,
                    settlement_source=SettlementSource.EXTERNAL_PRICE_PROXY.value,
                    result="PENDING",
                    pnl=0.0,
                    model_version="v3_lgbm",
                    expert_votes=ensemble.expert_votes if ensemble else {},
                )
                self.shadow.add_candidate(record)

            # ═══════════════════════════════════════════════════
            # Phase 2: Opportunity Ranking (only SHADOW_ACTIVE + LIVE symbols)
            # ═══════════════════════════════════════════════════
            tradeable = [c for c in candidates if self.shadow.is_shadow_active(c.symbol) or self.shadow.can_place_order()]
            edges = [c.edge for c in tradeable if c.edge is not None]

            if not edges:
                # 只有 OBSERVE 候选，记录但不交易
                emit("opportunity_ranked", {
                    "total_candidates": len(candidates),
                    "tradeable": 0,
                    "observe_only": len(candidates),
                    "selected": 0,
                })
                return

            ranked = self.ranker.rank(edges)
            selected = [o for o in ranked if o.selected]

            if selected:
                emit("opportunity_ranked", {
                    "total_candidates": len(candidates),
                    "tradeable": len(tradeable),
                    "ranked": len(ranked),
                    "selected": len(selected),
                    "best": selected[0].symbol if selected else "",
                    "best_score": selected[0].rank_score if selected else 0,
                })

            # ═══════════════════════════════════════════════════
            # Phase 3: Execute selected opportunities
            # ═══════════════════════════════════════════════════
            for opp in selected:
                # 找到对应的 CandidateOpportunity
                matching = [c for c in tradeable if c.symbol == opp.symbol and c.direction_str == opp.direction]
                if not matching:
                    continue
                candidate = matching[0]

                # 更新 candidate 的 position（已通过 ranker 选择）
                pos = self.portfolio_risk.compute_position_size(candidate.edge, self.balance)
                candidate.position = pos

                self._execute_opportunity(candidate)

        except Exception as e:
            emit("error", {"msg": f"Error: {str(e)[:100]}"})

    def get_status(self) -> dict:
        total = self.total_wins + self.total_losses
        wr = f"{(self.total_wins / total * 100):.1f}%" if total > 0 else "0.0%"
        state = "halted" if self.halted else (
            "paused" if self.paused else (
                "running" if self.running else "stopped"
            )
        )
        if self.pause_until > int(time.time() * 1000):
            state = "cooling"
        profit = self.balance - self.start_balance if self.start_balance > 0 else 0
        cal_ready = self.calibrator.is_ready()
        health_data = self._health_trades
        return {
            "state": state, "balance": self.balance,
            "wins": self.total_wins, "losses": self.total_losses,
            "winRate": wr, "activeTrades": len(self.active_trades),
            "maxConcurrentTrades": 999,
            "consecutiveLosses": self.consecutive_losses,
            "currentBet": config.MIN_ORDER_USD,
            "betMode": "eventedge_v2_kelly",
            "profit": round(profit, 2),
            "runMode": self.shadow.get_mode(),
            "calibrationReady": cal_ready,
            "healthTradeCount": len(health_data),
        }