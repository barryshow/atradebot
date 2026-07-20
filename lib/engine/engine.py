# -*- coding: utf-8 -*-
"""
ATradeBot 引擎 v5 — EventEdge V2 + Realtime Fast Entry

Realtime Feed (Gate.io 1m/5m)
  → Multi-Timeframe Features
  → Slow Context (15m LightGBM, every 15m)
  → Fast Entry (1m LightGBM, every 5s scan)
  → Ensemble + Edge + Ranker + Risk → Order

每笔交易仍是 15 分钟 HIBT Event Contract 到期。
"""
import time, json, sys, os, joblib, numpy as np, pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from . import config
from .exchange import fetch_balance, place_order
from .notifier import notify_trade, notify_result
from .models import (Prediction, TradeSignal, GateResult, EdgeResult,
                     MarketRegime, EnsemblePrediction, ExpertPrediction)
from .order_executor import OrderExecutor
from .regime_detector import MarketRegimeDetector, get_regime_detector
from .experts import ExpertManager, get_expert_manager
from .edge_engine import EdgeEngine, get_edge_engine
from .uncertainty import ModelUncertaintyEstimator, get_uncertainty_estimator
from .portfolio_risk import PortfolioRiskManager, get_portfolio_risk
from .opportunity_ranker import OpportunityRanker, get_opportunity_ranker
from .shadow_mode import (ShadowMode, RunMode, get_shadow_mode, set_run_mode,
                          CalibrationStatus, SettlementSource, ShadowCandidateRecord)
from .trade_ledger import TradeLedger, TradeRecord, get_trade_ledger
from .settlement_ledger import SettlementLedger, get_settlement_ledger, RejectReason
from .model_health import ModelHealthMonitor, get_model_health_monitor
from .probability_calibrator import WalkForwardCalibrator
from .realtime_feed import RealtimeFeed, get_realtime_feed
from .multi_timeframe_features import (
    compute_fast_entry_features, FAST_FEATURES, build_fast_feature_vector)


def emit(event_type: str, payload: dict):
    event = {"type": event_type, "ts": int(time.time() * 1000), "payload": payload}
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _fuse_probabilities(fast_prob: float, slow_prob: float, features: dict) -> float:
    """融合 Fast 和 Slow 概率"""
    if abs(fast_prob - slow_prob) > 0.10:
        w = 0.6  # 冲突时偏向 Fast
    else:
        w = 0.5
    return w * fast_prob + (1 - w) * slow_prob


@dataclass
class CandidateOpportunity:
    symbol: str = ""
    direction_str: str = ""
    direction_int: int = 0
    current_price: float = 0.0
    current_bar_ts: object = None
    current_ts: int = 0
    regime: Optional[MarketRegime] = None
    ensemble: Optional[EnsemblePrediction] = None
    predictions: list = field(default_factory=list)
    edge: Optional[EdgeResult] = None
    indicators: dict = field(default_factory=dict)
    row: dict = field(default_factory=dict)
    position: Optional[object] = None


class TradingEngine:
    def __init__(self, run_mode: str = "live", smoke_test: bool = False):
        self.executor = OrderExecutor()
        self.reset_state()
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
        mode = RunMode.LIVE if run_mode == "live" else (RunMode.SHADOW if run_mode == "shadow" else RunMode.BACKTEST)
        self.shadow = ShadowMode(mode=mode)
        self._smoke_test = smoke_test
        self._smoke_order_count = 0
        self._smoke_max_orders = 1
        self._realtime_feed: Optional[RealtimeFeed] = None
        self._fast_models: Dict[str, object] = {}
        self._fast_scalers: Dict[str, object] = {}
        self._fast_model_loaded = False
        self._slow_context: Dict[str, dict] = {}
        self._last_slow_update: Dict[str, float] = {}
        self._last_fast_scan = 0.0
        self._fast_scan_count = 0
        self._hourly_trade_count: Dict[str, int] = {}
        self._hourly_trade_window_start = time.time()
        self._last_trade_time: Dict[str, float] = {}
        self._cooldown_seconds = config.SIGNAL_COOLDOWN_SECONDS
        self._health_trades: list = []
        self._last_health_check = time.time()
        self._sys_funnel: dict[str, int] = {}
        self._strat_funnel: dict[str, dict[str, int]] = {}
        self._funnel_last_report = time.time()
        self._last_decision_bar: dict[str, object] = {}
        self._bars_evaluated_total: dict[str, int] = {}

    def set_run_mode(self, mode: str):
        m = RunMode.LIVE if mode == "live" else (RunMode.SHADOW if mode == "shadow" else RunMode.BACKTEST)
        self.shadow = ShadowMode(mode=m)

    def reset_state(self):
        self.running = False; self.paused = False
        self.balance = 0.0; self.start_balance = 0.0
        self.active_trades: list[dict] = []
        self.last_trade_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_reject_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_signal_bar_ts: dict[str, object] = {s: None for s in config.SYMBOLS}
        self.recent_results: list[bool] = []
        self.consecutive_losses = 0; self.halted = False; self.pause_until = 0
        self.total_pnl = 0.0; self.total_wins = 0; self.total_losses = 0
        self._warmup_done: dict[str, bool] = {s: False for s in config.SYMBOLS}
        self._shadow_simulated_balance = False
        self.executor = OrderExecutor()

    # ── Funnel helpers ──
    def _sys_funnel_count(self, stage: str):
        self._sys_funnel[stage] = self._sys_funnel.get(stage, 0) + 1

    def _strat_funnel_count(self, symbol: str, stage: str):
        if symbol not in self._strat_funnel:
            self._strat_funnel[symbol] = {}
        self._strat_funnel[symbol][stage] = self._strat_funnel[symbol].get(stage, 0) + 1

    def _emit_funnel_report(self):
        now = time.time()
        if now - self._funnel_last_report < 60:
            return
        self._funnel_last_report = now
        emit("funnel", {"type": "system",
            "ticks_total": self._sys_funnel.get("ticks_total", 0),
            "fast_scans": self._fast_scan_count,
            "data_stale": self._sys_funnel.get("data_stale", 0)})
        for sym in sorted(self._strat_funnel.keys()):
            f = self._strat_funnel[sym]
            emit("funnel", {"type": "strategy", "symbol": sym,
                "edge_rejected": f.get("edge_rejected", 0),
                "edge_passed": f.get("edge_passed", 0),
                "cooldown_rejected": f.get("cooldown_rejected", 0),
                "symbol_max_rejected": f.get("symbol_max_rejected", 0),
                "max_active_rejected": f.get("max_active_rejected", 0),
                "max_hourly_rejected": f.get("max_hourly_rejected", 0),
                "candidate_generated": f.get("candidate_generated", 0),
                "shadow_trade": f.get("shadow_trade", 0),
                "observe_only": f.get("observe_only", 0)})

    # ── Start / Stop ──
    def start(self):
        self.reset_state()
        self.running = True
        run_mode = self.shadow.get_mode()

        shadow_symbols = list(self.shadow.get_symbol_mode_summary().keys())
        self._realtime_feed = RealtimeFeed(shadow_symbols, scan_interval=config.FAST_SCAN_INTERVAL_SECONDS)
        self._realtime_feed.start()
        self._load_fast_models()

        cal_ready = self.calibrator.is_ready()
        cal_status = "READY" if cal_ready else "NOT_READY"
        if not cal_ready:
            emit("calibration_status", {"status": "NOT_READY", "msg": "PASSTHROUGH_UNCALIBRATED"})

        live_gate = self.shadow.get_live_gate_status(cal_ready, data_fresh=True,
            balance_ok=(self.balance >= config.MIN_ORDER_USD))
        if self.shadow.is_live_allowed() and not live_gate["passed"]:
            emit("log", {"msg": f"LIVE_VALIDATION_GATE_NOT_PASSED: {'; '.join(live_gate['reasons'])}"})
            set_run_mode(RunMode.SHADOW); self.shadow = ShadowMode(mode=RunMode.SHADOW)
            run_mode = "SHADOW"

        sym_modes = self.shadow.get_symbol_mode_summary()
        active = [s for s, m in sym_modes.items() if m == "SHADOW_ACTIVE"]

        emit("status", {"state": "running", "run_mode": run_mode,
            "calibration": cal_status, "live_gate": live_gate, "symbol_modes": sym_modes,
            "fast_scan_interval": config.FAST_SCAN_INTERVAL_SECONDS,
            "fast_model_loaded": self._fast_model_loaded})

        self.balance = fetch_balance()
        if self.balance < 0: self.balance = 0.0
        if run_mode == "SHADOW" and self.balance < config.MIN_ORDER_USD:
            self.balance = max(500.0, config.MIN_ORDER_USD * 10)
            self._shadow_simulated_balance = True
        else:
            self._shadow_simulated_balance = False
        self.start_balance = self.balance

        emit("log", {"msg": f"EventEdge V2 Fast Entry | 余额{self.balance:.0f}U | "
            f"扫描{config.FAST_SCAN_INTERVAL_SECONDS}s | Cooldown{self._cooldown_seconds}s | "
            f"MaxTrades/h{config.MAX_NEW_TRADES_PER_HOUR} | "
            f"Active:{','.join(active) if active else 'none'} | "
            f"FastModel:{'OK' if self._fast_model_loaded else 'NONE'} | "
            f"LIVE:{'ENABLED' if live_gate['passed'] else 'DISABLED'}"})

    def stop(self):
        self.running = False; self.paused = False
        if self._realtime_feed: self._realtime_feed.stop()
        emit("status", {"state": "stopped"})

    def pause(self):
        self.paused = True
        emit("status", {"state": "paused"})

    def resume(self):
        self.paused = False; self.halted = False
        self.pause_until = 0; self.consecutive_losses = 0
        emit("status", {"state": "running"})

    def _load_fast_models(self):
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        if not os.path.isdir(model_dir): model_dir = os.path.join(os.getcwd(), "models")
        loaded = 0
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            path = os.path.join(model_dir, f"{sym.lower()}_fast_entry.pkl")
            if os.path.exists(path):
                try:
                    bundle = joblib.load(path)
                    self._fast_models[sym] = bundle["model"]
                    self._fast_scalers[sym] = bundle["scaler"]
                    loaded += 1
                except Exception: pass
        self._fast_model_loaded = loaded > 0

    # ═══════════════════════════════════════════════════════════
    # Fast Entry Scan (每 5s)
    # ═══════════════════════════════════════════════════════════

    def _run_fast_entry_scan(self):
        now = time.time()
        if now - self._last_fast_scan < config.FAST_SCAN_INTERVAL_SECONDS:
            return
        self._last_fast_scan = now
        self._fast_scan_count += 1

        if self._realtime_feed is None:
            return
        if now - self._hourly_trade_window_start > 3600:
            self._hourly_trade_count = {}
            self._hourly_trade_window_start = now

        candidates = []
        for sym in self._realtime_feed.symbols:
            sym_mode = self.shadow.get_symbol_mode(sym)
            if sym_mode.value == "DISABLED":
                continue

            df_1m = self._realtime_feed.get_klines(sym, "1m")
            df_5m = self._realtime_feed.get_klines(sym, "5m")
            rt = self._realtime_feed.get_realtime_price(sym)
            if df_1m is None or len(df_1m) < 50 or rt is None:
                continue

            self._update_slow_context(sym, df_1m)

            fast_features = compute_fast_entry_features(
                sym, rt, df_1m, df_5m, slow_context=self._slow_context.get(sym))

            fast_prob = 0.50
            if self._fast_model_loaded and sym in self._fast_models:
                try:
                    vec = build_fast_feature_vector(fast_features).reshape(1, -1)
                    vec_s = self._fast_scalers[sym].transform(vec)
                    proba = self._fast_models[sym].predict_proba(vec_s)
                    pos_idx = 1 if 1 in self._fast_models[sym].classes_ else 0
                    fast_prob = float(proba[0, pos_idx])
                except Exception:
                    fast_prob = 0.50

            slow_ctx = self._slow_context.get(sym, {})
            slow_prob = slow_ctx.get("probability", 0.50)
            ensemble_prob = _fuse_probabilities(fast_prob, slow_prob, fast_features)

            direction = 1 if ensemble_prob >= 0.50 else 2
            direction_str = "CALL" if direction == 1 else "PUT"
            if direction == 2:
                ensemble_prob = max(0.35, 1.0 - ensemble_prob)

            edge = self.edge_engine.compute(
                symbol=sym, calibrated_probability=ensemble_prob,
                direction=direction_str, direction_int=direction,
                expiry_minutes=config.HOLD_MINUTES, entry_price=rt.price,
                uncertainty_margin=0.02, regime=slow_ctx.get("regime", "RANGE"))

            cooldown_key = f"{sym}_{direction_str}"
            in_cooldown = (now - self._last_trade_time.get(cooldown_key, 0)) < self._cooldown_seconds
            # 每品种独立限制: 最多 1 个活跃合约
            active_for_symbol = sum(1 for t in self.active_trades if t.get("symbol") == sym)
            max_per_symbol = 1
            # 全局上限
            total_active = len(self.active_trades)
            max_global = config.MAX_ACTIVE_EVENT_CONTRACTS
            hourly_count = self._hourly_trade_count.get(sym, 0)
            max_hourly = config.MAX_NEW_TRADES_PER_HOUR

            status = "EDGE_PASSED" if edge.passed else "NO_EDGE"
            if in_cooldown: status = "COOLDOWN"
            if active_for_symbol >= max_per_symbol: status = "SYMBOL_AT_MAX"
            if total_active >= max_global: status = "MAX_ACTIVE_TRADES"
            if hourly_count >= max_hourly: status = "MAX_HOURLY_TRADES"

            # 同品种反方向持仓检查
            has_opposite = any(
                t.get("symbol") == sym and t.get("dir") != direction
                for t in self.active_trades
            )
            if has_opposite and not config.ALLOW_OPPOSITE_OVERLAP:
                status = "OPPOSITE_OVERLAP"

            emit("fast_scan", {"symbol": sym, "direction": direction_str,
                "fast_prob": round(fast_prob, 4), "slow_prob": round(slow_prob, 4),
                "ensemble_prob": round(ensemble_prob, 4),
                "effective_edge": round(edge.effective_edge, 4),
                "break_even": round(edge.break_even_probability, 4),
                "status": status, "cooldown": in_cooldown,
                "active_contracts": total_active,
                "symbol_active": active_for_symbol,
                "hourly_trades": hourly_count})

            if not edge.passed:
                self._strat_funnel_count(sym, "edge_rejected"); continue
            if in_cooldown:
                self._strat_funnel_count(sym, "cooldown_rejected"); continue
            if active_for_symbol >= max_per_symbol:
                self._strat_funnel_count(sym, "symbol_max_rejected"); continue
            if has_opposite and not config.ALLOW_OPPOSITE_OVERLAP:
                self._strat_funnel_count(sym, "opposite_rejected"); continue
            if total_active >= max_global:
                self._strat_funnel_count(sym, "max_active_rejected"); continue
            if hourly_count >= max_hourly:
                self._strat_funnel_count(sym, "max_hourly_rejected"); continue

            self._strat_funnel_count(sym, "edge_passed")

            ensemble = EnsemblePrediction(symbol=sym, direction=direction,
                ensemble_probability=round(ensemble_prob, 4),
                calibrated_probability=round(ensemble_prob, 4),
                conservative_probability=round(edge.conservative_probability, 4),
                regime=slow_ctx.get("regime", "RANGE"))

            regime = MarketRegime(regime=slow_ctx.get("regime", "RANGE"), confidence=0.5)
            candidate = CandidateOpportunity(symbol=sym, direction_str=direction_str,
                direction_int=direction, current_price=rt.price,
                current_ts=int(now * 1000), regime=regime,
                ensemble=ensemble, edge=edge,
                indicators=fast_features, row=fast_features)
            candidates.append(candidate)
            self._strat_funnel_count(sym, "candidate_generated")

        if not candidates:
            return

        tradeable = [c for c in candidates if self.shadow.is_shadow_active(c.symbol) or self.shadow.can_place_order()]
        edges = [c.edge for c in tradeable if c.edge is not None]
        if not edges:
            return

        ranked = self.ranker.rank(edges)
        selected = [o for o in ranked if o.selected]

        for opp in selected:
            matching = [c for c in tradeable if c.symbol == opp.symbol and c.direction_str == opp.direction]
            if not matching: continue
            candidate = matching[0]
            pos = self.portfolio_risk.compute_position_size(candidate.edge, self.balance)
            candidate.position = pos
            self._execute_opportunity(candidate)
            cooldown_key = f"{opp.symbol}_{opp.direction}"
            self._last_trade_time[cooldown_key] = now
            self._hourly_trade_count[opp.symbol] = self._hourly_trade_count.get(opp.symbol, 0) + 1

    def _update_slow_context(self, sym: str, df_1m: pd.DataFrame):
        if df_1m is None or len(df_1m) < 15: return
        last_bar = df_1m.index[-1]
        last_update = self._last_slow_update.get(sym, 0)
        if hasattr(last_bar, 'timestamp'):
            bar_ts = last_bar.timestamp()
            if bar_ts - last_update < 840: return
            self._last_slow_update[sym] = bar_ts
        indicators = {"ADX": 20.0, "RSI": 50.0, "BB_Pos": 0.5, "bb_width": 0.02,
            "volatility_ratio": 1.0, "ATR_pct": 0.003, "price_vs_MA20": 0.0,
            "MACD": 0.0, "MA_trend": 0.0, "VWAP_dist": 0.0,
            "vol_ratio": 1.0, "CCI": 0.0}
        row = {"ret_1": 0.0, "ret_3": 0.0, "ret_6": 0.0, "body_pct": 0.3}
        try:
            predictions = self.expert_manager.predict_all(sym, indicators, row)
            regime = self.regime_detector.detect(indicators)
            ensemble = self.expert_manager.ensemble(predictions, regime)
            self._slow_context[sym] = {"probability": ensemble.ensemble_probability,
                "regime": regime.regime,
                "trend_strength": float(indicators.get("ADX", 20)) / 40.0,
                "volatility": float(indicators.get("volatility_ratio", 1.0))}
        except Exception:
            self._slow_context[sym] = {"probability": 0.50, "regime": "RANGE",
                "trend_strength": 0.0, "volatility": 0.5}

    # ═══════════════════════════════════════════════════════════
    # Order Execution (shared by Fast Entry + old 15m pipeline)
    # ═══════════════════════════════════════════════════════════

    def _execute_opportunity(self, candidate: CandidateOpportunity) -> bool:
        symbol = candidate.symbol
        direction_str = candidate.direction_str
        direction_int = candidate.direction_int
        edge = candidate.edge
        ensemble = candidate.ensemble
        regime = candidate.regime
        indicators = candidate.indicators
        current_price = candidate.current_price
        current_ts = candidate.current_ts

        pos = self.portfolio_risk.compute_position_size(edge, self.balance)
        if not pos.allowed:
            self._strat_funnel_count(symbol, "portfolio_rejected")
            return False
        stake_usd = pos.stake_usd

        active_positions = {t.get("symbol", ""): [{"direction": t.get("direction_str", direction_str),
            "stake_usd": t.get("amount", 0)}] for t in self.active_trades}
        port_check = self.portfolio_risk.check_portfolio_limits(
            symbol=symbol, direction=direction_str, stake_usd=stake_usd,
            equity=self.balance, active_positions=active_positions)
        if not port_check.allowed:
            return False

        if self.shadow.can_place_order():
            if self._smoke_test and self._smoke_order_count >= self._smoke_max_orders:
                return False
            result = place_order(symbol, direction_int, stake_usd, config.HOLD_MINUTES)
            if result.ok:
                lifecycle = "ORDER_ACCEPTED" if result.order_id else "ORDER_REQUESTED"
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
                    regime=regime.regime if regime else "",
                    expert_votes=ensemble.expert_votes if ensemble else {})
                if result.order_id: rec.hibt_order_id = result.order_id
                rec.settlement_status = lifecycle
                self.trade_ledger.save(rec)

                self.active_trades.append({
                    "symbol": symbol, "dir": direction_int, "direction_str": direction_str,
                    "start_ts": current_ts, "amount": stake_usd,
                    "entry": current_price, "pre_balance": self.balance,
                    "trade_id": rec.trade_id, "hibt_order_id": result.order_id,
                    "lifecycle": lifecycle,
                    "open_price_verified": result.open_price is not None,
                    "predicted_entry_price": current_price,
                    "hibt_open_price": result.open_price})

                if self._smoke_test:
                    self._smoke_order_count += 1
                    if self._smoke_order_count >= self._smoke_max_orders:
                        emit("log", {"msg": "SMOKE_TEST: limit reached"})

                emit("trade_executed", {"symbol": symbol, "direction": direction_str,
                    "trade_id": rec.trade_id, "hibt_order_id": result.order_id,
                    "entryPrice": current_price, "amount": stake_usd,
                    "rawProbability": ensemble.ensemble_probability,
                    "calibratedProbability": edge.calibrated_probability,
                    "effectiveEdge": edge.effective_edge,
                    "expectedRoi": edge.expected_roi,
                    "balance": self.balance, "lifecycle": lifecycle,
                    "open_price_verified": result.open_price is not None,
                    "predicted_entry_price": current_price,
                    "hibt_open_price": result.open_price})

                self.balance = fetch_balance()
                if self.balance < 0: self.balance = self.balance - stake_usd
                emit("balance_update", {"balance": self.balance})
                return True
            else:
                emit("trade_rejected", {"symbol": symbol, "reason": "ORDER_FAILED", "detail": result.msg})
                return False
        else:
            # SHADOW_ACTIVE
            self._strat_funnel_count(symbol, "shadow_trade")
            self.shadow.record_shadow_trade(
                symbol=symbol, direction=direction_str, direction_int=direction_int,
                entry_time_ms=current_ts, entry_price=current_price,
                stake_usd=stake_usd, expiry_minutes=config.HOLD_MINUTES,
                calibrated_probability=edge.calibrated_probability,
                break_even_probability=edge.break_even_probability,
                effective_edge=edge.effective_edge,
                expected_roi=edge.expected_roi,
                regime=regime.regime if regime else "",
                expert_votes=ensemble.expert_votes if ensemble else {})
            emit("shadow_trade", {"symbol": symbol, "direction": direction_str,
                "entryPrice": current_price, "amount": stake_usd,
                "effectiveEdge": edge.effective_edge, "expectedRoi": edge.expected_roi,
                "mode": "SHADOW_ACTIVE"})
            return True

    # ═══════════════════════════════════════════════════════════
    # Settlement
    # ═══════════════════════════════════════════════════════════

    def _check_settlement(self):
        if not self.active_trades: return
        current_balance = fetch_balance()
        if current_balance < 0: return
        if getattr(self, '_shadow_simulated_balance', False):
            current_balance = self.balance

        for i in range(len(self.active_trades) - 1, -1, -1):
            t = self.active_trades[i]
            elapsed_ms = time.time() * 1000 - t["start_ts"]
            settle_ms = config.HOLD_MINUTES * 60000 + 30000
            if elapsed_ms < settle_ms: continue

            pnl = current_balance - t["pre_balance"]
            is_win = pnl > 0; is_tie = abs(pnl) < 0.001
            if is_win: self.total_wins += 1
            elif not is_tie: self.total_losses += 1
            self.total_pnl += pnl
            if not is_tie: self._record_result(is_win)
            result = "tie" if is_tie else ("win" if is_win else "loss")

            trade_id = t.get("trade_id", "")
            if trade_id:
                self.settlement_ledger.estimate_settlement(trade_id=trade_id,
                    current_balance=current_balance, pre_balance=t["pre_balance"])
                self.trade_ledger._update_field(trade_id, "settlement_status", "SETTLED_UNVERIFIED")

            emit("trade_result", {"symbol": t["symbol"], "result": result,
                "pnl": round(pnl, 4), "trade_id": trade_id,
                "settlement": "SETTLED_UNVERIFIED",
                "hibt_order_id": t.get("hibt_order_id"),
                "open_price_verified": t.get("open_price_verified", False),
                "predicted_entry_price": t.get("predicted_entry_price"),
                "hibt_open_price": t.get("hibt_open_price")})

            self._health_trades.append({"predicted_prob": 0.50, "result": "WIN" if is_win else ("TIE" if is_tie else "LOSS"), "pnl": pnl})
            if len(self._health_trades) > 500: self._health_trades = self._health_trades[-500:]
            self.portfolio_risk.record_pnl(pnl)
            self.active_trades.pop(i)
            self.balance = current_balance
            emit("balance_update", {"balance": current_balance})

    def _record_result(self, is_win: bool):
        self.recent_results.append(is_win)
        if len(self.recent_results) > config.RECENT_WINDOW:
            self.recent_results = self.recent_results[-config.RECENT_WINDOW:]
        if is_win: self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.CONSECUTIVE_LOSS_HALT:
                self.halted = True
            elif self.consecutive_losses >= 3:
                self.pause_until = int(time.time() * 1000) + config.CONSECUTIVE_LOSS_PAUSE_SEC * 1000

    # ═══════════════════════════════════════════════════════════
    # Main Tick
    # ═══════════════════════════════════════════════════════════

    def tick(self):
        self._sys_funnel_count("ticks_total")
        try:
            self._check_settlement()
            self._run_fast_entry_scan()
            self._emit_funnel_report()
        except Exception as e:
            emit("error", {"msg": f"Error: {str(e)[:100]}"})

    def get_status(self) -> dict:
        total = self.total_wins + self.total_losses
        wr = f"{(self.total_wins / total * 100):.1f}%" if total > 0 else "0.0%"
        state = "halted" if self.halted else ("paused" if self.paused else ("running" if self.running else "stopped"))
        if self.pause_until > int(time.time() * 1000): state = "cooling"
        profit = self.balance - self.start_balance if self.start_balance > 0 else 0
        cal_ready = self.calibrator.is_ready()
        return {"state": state, "balance": self.balance,
            "wins": self.total_wins, "losses": self.total_losses,
            "winRate": wr, "activeTrades": len(self.active_trades),
            "maxConcurrentTrades": config.MAX_ACTIVE_EVENT_CONTRACTS,
            "consecutiveLosses": self.consecutive_losses,
            "currentBet": config.MIN_ORDER_USD, "betMode": "fast_entry_kelly",
            "profit": round(profit, 2), "runMode": self.shadow.get_mode(),
            "calibrationReady": cal_ready,
            "healthTradeCount": len(self._health_trades),
            "liveGate": self.shadow.get_live_gate_status(cal_ready),
            "symbolModes": self.shadow.get_symbol_mode_summary(),
            "fastScanCount": self._fast_scan_count,
            "fastScanInterval": config.FAST_SCAN_INTERVAL_SECONDS,
            "fastModelLoaded": self._fast_model_loaded}