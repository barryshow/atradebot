# -*- coding: utf-8 -*-
"""
ATradeBot Engine v6 — EV-based Decision Engine + Reliable Settlement.

Architecture:
  RealtimeFeed (Gate.io 1m/5m)
    ├─ Fast Model (1m LightGBM) → fast_prob
    ├─ Slow Model (15m LightGBM) → slow_prob
    ├─ Probability Calibrator → calibrated_prob
    ├─ Payout Fetcher (HIBT API) → payout_call_net, payout_put_net
    └─ DecisionEngine.evaluate() → CALL / PUT / ABSTAIN
         ├─ RiskManager checks
         ├─ RunModeManager (PAPER/SHADOW/LIVE)
         └─ TradeLedger (SQLite)

Key changes from v5:
  - No more "prob >= 0.50 → CALL" — uses EV formula
  - No more balance-based settlement guessing
  - Default SHADOW mode, LIVE requires explicit opt-in
  - BTC disabled until OOS validation passes
"""
import time, json, sys, os, joblib, numpy as np, pandas as pd
from typing import Optional, Dict, List

from . import config
from .exchange import fetch_balance, place_order
from .notifier import notify_trade, notify_result
from .realtime_feed import RealtimeFeed, RealtimePrice, get_realtime_feed
from .multi_timeframe_features import (
    compute_fast_entry_features, FAST_FEATURES, build_fast_feature_vector)

# New modules
from .run_mode import (
    RunMode, RunModeManager, LiveGate, get_run_mode_manager, is_killed)
from .strategy.decision_engine import DecisionEngine, Decision, DecisionType
from .risk.risk_manager import RiskManager, RiskCheckResult, get_risk_manager
from .data.trade_ledger import TradeLedger, TradeRecord, get_trade_ledger
from .settlement.reconciler import (
    SettlementReconciler, SettlementEvent, get_settlement_reconciler)
from .models.model_registry import ModelRegistry, get_model_registry
from .probability_calibrator import WalkForwardCalibrator
from .regime_detector import MarketRegimeDetector, get_regime_detector
from .experts import ExpertManager, get_expert_manager


def emit(event_type: str, payload: dict):
    event = {"type": event_type, "ts": int(time.time() * 1000), "payload": payload}
    line = json.dumps(event, ensure_ascii=False) + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    try:
        _debug_fd = getattr(emit, "_fd", None)
        if _debug_fd is None:
            import tempfile as _tmp
            emit._fd = open(os.path.join(_tmp.gettempdir(), "atradebot_emit.log"), "a", encoding="utf-8")
        emit._fd.write(line)
        emit._fd.flush()
    except Exception:
        pass


class TradingEngine:
    def __init__(self, run_mode: str = "shadow", smoke_test: bool = False):
        # ── New core modules ──
        self.run_mode_mgr = get_run_mode_manager()
        self.decision_engine = DecisionEngine()
        self.risk_manager = get_risk_manager()
        self.trade_ledger = get_trade_ledger()
        self.reconciler = get_settlement_reconciler()
        self.model_registry = get_model_registry()

        # ── Legacy modules (kept for model inference) ──
        self.regime_detector = get_regime_detector()
        self.expert_manager = get_expert_manager()
        self.calibrator = WalkForwardCalibrator(method="isotonic", min_samples=50)

        # ── State ──
        self.running = False
        self.paused = False
        self.balance = 0.0
        self.start_balance = 0.0
        self.total_pnl = 0.0
        self.active_hibt_order_ids: List[str] = []
        self.last_data_update: Dict[str, float] = {}
        self._cooldown_seconds = config.SIGNAL_COOLDOWN_SECONDS
        self._last_trade_time: Dict[str, float] = {}
        self._hourly_trade_count: Dict[str, int] = {}
        self._hourly_window_start = time.time()
        self._consecutive_same_symbol: Dict[str, int] = {}

        # ── Models ──
        self._fast_models: Dict[str, object] = {}
        self._fast_scalers: Dict[str, object] = {}
        self._fast_model_loaded = False
        self._slow_context: Dict[str, dict] = {}
        self._last_slow_update: Dict[str, float] = {}

        # ── Timing ──
        self._last_fast_scan = 0.0
        self._fast_scan_count = 0
        self._warmup_until = time.time() + 180  # 3 min warmup
        self._funnel_last_report = time.time()
        self._strat_funnel: Dict[str, Dict[str, int]] = {}
        self._realtime_feed: Optional[RealtimeFeed] = None

        # ── Smoke test ──
        self._smoke_test = smoke_test
        self._smoke_order_count = 0
        self._smoke_max_orders = 1

    # ═══════════════════════════════════════════════════════════
    # Start / Stop
    # ═══════════════════════════════════════════════════════════

    def start(self):
        self.running = True
        self._warmup_until = time.time() + 180

        symbols = list(config.SHADOW_SYMBOL_MODE.keys())
        self._realtime_feed = RealtimeFeed(symbols, scan_interval=config.FAST_SCAN_INTERVAL_SECONDS)
        self._realtime_feed.start()
        self._load_fast_models()
        self._register_models()

        # ── Seed from ledger ──
        settled = self.trade_ledger.get_settled_count()
        if settled["total"] > 0:
            emit("log", {"msg": f"Ledger: {settled['total']} settled trades loaded (W{settled['wins']}/L{settled['losses']})"})

        # ── Balance ──
        self.balance = fetch_balance()
        if self.balance < 0:
            self.balance = 0.0
        run_mode = self.run_mode_mgr.mode
        if run_mode == RunMode.SHADOW and self.balance < config.MIN_ORDER_USD:
            self.balance = max(500.0, config.MIN_ORDER_USD * 10)

        self.start_balance = self.balance

        # ── Payout + Settlement checks ──
        payout_call, payout_put = self.reconciler.get_available_payout("BTCUSDT")
        payout_available = payout_call is not None and payout_put is not None
        settlement_available = self.reconciler.can_settle_via_hibt()

        # ── LIVE Gate ──
        calibrator_ready = self.calibrator.is_ready()
        live_result = self.run_mode_mgr.get_live_gate().check(
            calibrator_ready=calibrator_ready,
            data_fresh=True,
            model_valid=self._fast_model_loaded,
            has_pending_orders=len(self.trade_ledger.get_pending_settlements()) > 0,
            risk_paused=False,
            settlement_available=settlement_available,
            payout_available=payout_available,
        )

        sym_modes = {s: self.run_mode_mgr.get_symbol_mode(s) for s in symbols}
        active_symbols = [s for s, m in sym_modes.items() if m == "SHADOW_ACTIVE"]

        emit("status", {
            "state": "running",
            "run_mode": run_mode.value,
            "calibration": "READY" if calibrator_ready else "NOT_READY",
            "live_gate": {
                "passed": live_result.passed,
                "hard_blocks": live_result.hard_blocks,
                "soft_warnings": live_result.soft_warnings,
                "reasons": live_result.reasons,
                "checks": live_result.all_checks,
            },
            "payout_available": payout_available,
            "settlement_available": settlement_available,
            "symbol_modes": sym_modes,
            "fast_scan_interval": config.FAST_SCAN_INTERVAL_SECONDS,
            "fast_model_loaded": self._fast_model_loaded,
        })

        emit("log", {"msg": (
            f"Engine v6 EV-Based | RunMode={run_mode.value} | "
            f"Balance={self.balance:.0f}U | Scan={config.FAST_SCAN_INTERVAL_SECONDS}s | "
            f"Active={','.join(active_symbols) if active_symbols else 'none'} | "
            f"Calibrator={'OK' if calibrator_ready else 'NOT_READY'} | "
            f"Payout={'API' if payout_available else 'HARDCODED'} | "
            f"Settlement={'HIBT' if settlement_available else 'UNAVAILABLE'} | "
            f"LIVE={'OK' if live_result.passed else 'BLOCKED: '+','.join(live_result.hard_blocks[:3])}"
        )})

    def stop(self):
        self.running = False
        self.paused = False
        if self._realtime_feed:
            self._realtime_feed.stop()
        emit("status", {"state": "stopped"})

    def pause(self):
        self.paused = True
        emit("status", {"state": "paused"})

    def resume(self):
        self.paused = False
        self.risk_manager.reset_pause()
        emit("status", {"state": "running"})

    def _register_models(self):
        """Register loaded models in the registry."""
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            fast_path = os.path.join(config.MODEL_DIR, f"{sym.lower()}_fast_entry.pkl")
            if os.path.exists(fast_path):
                self.model_registry.register(
                    model_name=f"{sym.lower()}_fast_entry_v1",
                    model_type="fast_entry",
                    symbol=sym,
                    file_path=fast_path,
                    feature_count=len(FAST_FEATURES),
                    is_active=1 if sym != "BTCUSDT" else 0,  # BTC disabled
                )

    def _load_fast_models(self):
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        if not os.path.isdir(model_dir):
            model_dir = os.path.join(os.getcwd(), "models")
        loaded = 0
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            path = os.path.join(model_dir, f"{sym.lower()}_fast_entry.pkl")
            if os.path.exists(path):
                try:
                    bundle = joblib.load(path)
                    self._fast_models[sym] = bundle["model"]
                    self._fast_scalers[sym] = bundle["scaler"]
                    loaded += 1
                except Exception:
                    pass
        self._fast_model_loaded = loaded > 0

    # ═══════════════════════════════════════════════════════════
    # Fast Scan (every 5s)
    # ═══════════════════════════════════════════════════════════

    def _run_fast_entry_scan(self):
        now = time.time()

        if now < self._warmup_until:
            return
        if now - self._last_fast_scan < config.FAST_SCAN_INTERVAL_SECONDS:
            return
        self._last_fast_scan = now
        self._fast_scan_count += 1

        if self._realtime_feed is None:
            return

        # Reset hourly counter
        if now - self._hourly_window_start > 3600:
            self._hourly_trade_count = {}
            self._hourly_window_start = now

        # ── Check kill switch ──
        if is_killed():
            return

        # ── Check risk manager pause ──
        risk_status = self.risk_manager.get_status()
        if risk_status["paused"]:
            if self._fast_scan_count % 60 == 0:  # Log every 5 min
                emit("log", {"msg": f"Risk paused: {risk_status['paused_reason']}, remaining={risk_status['pause_remaining_seconds']}s"})
            return

        decisions: List[Decision] = []

        for sym in self._realtime_feed.symbols:
            # Skip disabled symbols
            if not self.run_mode_mgr.is_symbol_active(sym):
                continue
            if not self.risk_manager.is_symbol_enabled(sym):
                continue

            # Get data
            df_1m = self._realtime_feed.get_klines(sym, "1m")
            df_5m = self._realtime_feed.get_klines(sym, "5m")
            rt = self._realtime_feed.get_realtime_price(sym)
            if df_1m is None or len(df_1m) < 50 or rt is None:
                continue

            self.last_data_update[sym] = now

            # Update slow context
            self._update_slow_context(sym, df_1m)

            # Compute features
            fast_features = compute_fast_entry_features(
                sym, rt, df_1m, df_5m,
                slow_context=self._slow_context.get(sym))

            # Fast model prediction
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

            # Slow model probability
            slow_ctx = self._slow_context.get(sym, {})
            slow_prob = slow_ctx.get("probability", 0.50)

            # Calibrate
            calibrated_prob = self._calibrate_probability(fast_prob)

            # ── Get payout (try API first, fall back to hardcoded) ──
            payout_call_net, payout_put_net = self.reconciler.get_available_payout(sym)
            payout_source = "api"
            if payout_call_net is None or payout_put_net is None:
                payout_source = "hardcoded"
                payout_call_net = config.PAYOUT_RATES.get(sym, 0.80)
                payout_put_net = config.PAYOUT_RATES.get(sym, 0.80)

            # ── EV-based decision ──
            decision = self.decision_engine.evaluate(
                symbol=sym,
                fast_probability=fast_prob,
                slow_probability=slow_prob,
                calibrated_probability=calibrated_prob,
                payout_call_net=payout_call_net,
                payout_put_net=payout_put_net,
                payout_source=payout_source,
                regime=slow_ctx.get("regime", "RANGE"),
                model_version="fast_entry_v1",
                feature_version="v1",
                gate_price=rt.price,
                signal_time_ms=int(now * 1000),
            )

            # ── Emit scan result ──
            emit("fast_scan", {
                "symbol": sym,
                "direction": decision.decision.value,
                "fast_prob": round(fast_prob, 4),
                "slow_prob": round(slow_prob, 4),
                "calibrated_prob": round(calibrated_prob, 4),
                "ev_call": round(decision.ev_call, 6),
                "ev_put": round(decision.ev_put, 6),
                "selected_ev": round(decision.selected_ev, 6),
                "payout_source": payout_source,
                "status": "PASSED" if decision.passed else decision.reject_reason,
                "active_contracts": len(self.active_hibt_order_ids),
            })

            # ── Record abstentions ──
            if not decision.passed:
                self._strat_funnel_count(sym, f"reject_{decision.reject_reason.lower()}")
                self.trade_ledger.mark_abstain(
                    trade_id="", symbol=sym, direction=decision.decision.value,
                    reject_reason=decision.reject_reason,
                    signal_time_ms=int(now * 1000), gate_price=rt.price,
                    fast_prob=fast_prob, slow_prob=slow_prob, calibrated=calibrated_prob,
                    ev_call=decision.ev_call, ev_put=decision.ev_put,
                    run_mode=self.run_mode_mgr.mode.value,
                )
                continue

            # ── Risk checks ──
            risk_check = self.risk_manager.check_all(
                active_order_count=len(self.active_hibt_order_ids),
                data_last_update=self.last_data_update.get(sym, 0),
                symbol=sym,
            )
            if not risk_check.allowed:
                decision.passed = False
                decision.reject_reason = risk_check.reason
                self._strat_funnel_count(sym, f"risk_{risk_check.reason.lower()}")
                continue

            # ── Cooldown ──
            key = f"{sym}_{decision.decision.value}"
            if key in self._last_trade_time:
                elapsed = now - self._last_trade_time[key]
                if elapsed < self._cooldown_seconds:
                    self._strat_funnel_count(sym, "cooldown_rejected")
                    continue

            # ── Concentration check ──
            same_sym_count = self._consecutive_same_symbol.get(sym, 0)
            if same_sym_count >= 3:
                self._strat_funnel_count(sym, "concentration_rejected")
                continue

            # ── Hourly limit ──
            hourly = self._hourly_trade_count.get(sym, 0)
            if hourly >= config.MAX_NEW_TRADES_PER_HOUR:
                self._strat_funnel_count(sym, "max_hourly_rejected")
                continue

            self._strat_funnel_count(sym, "decision_passed")
            decisions.append(decision)

        # ── Execute best decision ──
        if decisions:
            # Select highest EV decision
            best = max(decisions, key=lambda d: d.selected_ev)
            self._execute_decision(best)

        # ── Funnel report ──
        self._emit_funnel_report()

    def _calibrate_probability(self, raw_prob: float) -> float:
        """Calibrate raw model probability."""
        if self.calibrator.is_ready():
            cal = self.calibrator._calibrate_single(raw_prob)
            return cal
        return raw_prob

    # ═══════════════════════════════════════════════════════════
    # Execution
    # ═══════════════════════════════════════════════════════════

    def _execute_decision(self, decision: Decision):
        """Execute a trade decision."""
        now = int(time.time() * 1000)
        run_mode = self.run_mode_mgr.mode

        # ── Compute stake ──
        stake, bet_fraction = self.risk_manager.compute_stake(self.balance)
        stake = max(config.MIN_ORDER_USD, min(stake, self.balance * 0.05))

        # ── Create trade record ──
        rec = self.trade_ledger.create_record(
            symbol=decision.symbol,
            direction=decision.decision.value,
            direction_int=decision.direction_int,
            signal_time_ms=decision.signal_time_ms,
            gate_price_at_signal=decision.gate_price,
            fast_probability=decision.fast_probability,
            slow_probability=decision.slow_probability,
            calibrated_probability=decision.calibrated_probability,
            payout_call=decision.payout_call_net,
            payout_put=decision.payout_put_net,
            expected_value=decision.selected_ev,
            ev_call=decision.ev_call,
            ev_put=decision.ev_put,
            stake=stake,
            bet_fraction=bet_fraction,
            model_version=decision.model_version,
            feature_version=decision.feature_version,
            run_mode=run_mode.value,
            reject_reason=decision.reject_reason,
        )

        # ── SHADOW mode: record only, no real order ──
        if run_mode != RunMode.LIVE:
            emit("shadow_order_created", {
                "symbol": decision.symbol,
                "direction": decision.decision.value,
                "trade_id": rec.trade_id,
                "entry_price": decision.gate_price,
                "amount": stake,
                "ev": round(decision.selected_ev, 6),
                "ev_call": round(decision.ev_call, 6),
                "ev_put": round(decision.ev_put, 6),
                "lifecycle": "SHADOW_RECORDED",
            })
            self._strat_funnel_count(decision.symbol, "shadow_trade")
            self._after_trade(decision.symbol, decision.decision.value)
            return

        # ── LIVE mode: submit real order ──
        # Idempotency check
        idemp_key = f"{decision.symbol}_{decision.decision.value}_{int(time.time()/60)}"
        id_check = self.risk_manager.check_order_idempotency(idemp_key)
        if not id_check.allowed:
            self.trade_ledger.mark_rejected(rec.trade_id, id_check.reason, id_check.detail)
            return

        self.trade_ledger.mark_submitted(rec.trade_id, int(time.time() * 1000))

        result = place_order(
            decision.symbol,
            decision.direction_int,
            stake,
            config.HOLD_MINUTES,
        )

        if result.ok and result.order_id:
            self.risk_manager.record_order_idempotency(idemp_key)
            self.risk_manager.record_trade()
            self.trade_ledger.mark_accepted(
                rec.trade_id,
                hibt_order_id=result.order_id,
                accepted_time_ms=int(time.time() * 1000),
                hibt_open_price=result.open_price or decision.gate_price,
            )
            self.active_hibt_order_ids.append(result.order_id)
            self.balance = fetch_balance()
            if self.balance < 0:
                self.balance -= stake

            emit("trade_executed", {
                "symbol": decision.symbol,
                "direction": decision.decision.value,
                "trade_id": rec.trade_id,
                "hibt_order_id": result.order_id,
                "entry_price": decision.gate_price,
                "amount": stake,
                "ev": round(decision.selected_ev, 6),
                "ev_call": round(decision.ev_call, 6),
                "ev_put": round(decision.ev_put, 6),
                "balance": self.balance,
                "lifecycle": "ORDER_ACCEPTED",
                "payout_verified": result.payout_ratio is not None,
                "hibt_open_price": result.open_price,
            })
            self._after_trade(decision.symbol, decision.decision.value)
        else:
            self.trade_ledger.mark_rejected(rec.trade_id, "ORDER_FAILED", result.msg or "unknown")
            emit("trade_rejected", {
                "symbol": decision.symbol,
                "reason": "ORDER_FAILED",
                "detail": result.msg,
            })

    def _after_trade(self, symbol: str, direction: str):
        """Post-trade bookkeeping."""
        now = time.time()
        key = f"{symbol}_{direction}"
        self._last_trade_time[key] = now
        self._hourly_trade_count[symbol] = self._hourly_trade_count.get(symbol, 0) + 1
        self._consecutive_same_symbol[symbol] = self._consecutive_same_symbol.get(symbol, 0) + 1
        for s in list(self._consecutive_same_symbol.keys()):
            if s != symbol:
                self._consecutive_same_symbol[s] = 0

    # ═══════════════════════════════════════════════════════════
    # Settlement
    # ═══════════════════════════════════════════════════════════

    def _check_settlement(self):
        """Check and reconcile pending orders via HIBT API."""
        if not self.active_hibt_order_ids:
            return

        # Try HIBT settlement reconciliation
        events = self.reconciler.reconcile_all_pending(self.active_hibt_order_ids)

        for hibt_id, event in events.items():
            # Find matching trade record
            records = self.trade_ledger.query()
            for rec in records:
                if rec.hibt_order_id == hibt_id and rec.settlement_status == "PENDING":
                    self.trade_ledger.mark_settled(
                        rec.trade_id,
                        result=event.result,
                        pnl=event.pnl,
                        settlement_source="hibt_closed_orders",
                        hibt_close_price=event.close_price,
                    )
                    self.active_hibt_order_ids.remove(hibt_id)
                    self.balance = fetch_balance()
                    if self.balance < 0:
                        self.balance += event.pnl

                    is_win = event.result == "WIN"
                    self.risk_manager.record_result(is_win)
                    self.total_pnl += event.pnl
                    self.calibrator.update(rec.calibrated_probability, 1 if is_win else 0)

                    emit("trade_result", {
                        "symbol": rec.symbol,
                        "result": event.result,
                        "pnl": round(event.pnl, 4),
                        "trade_id": rec.trade_id,
                        "hibt_order_id": hibt_id,
                        "settlement": "HIBT_VERIFIED",
                        "close_price": event.close_price,
                        "open_price": event.open_price,
                    })

                    # Reset concentration counter after settlement
                    if rec.symbol and self._consecutive_same_symbol.get(rec.symbol, 0) >= 3:
                        self._consecutive_same_symbol[rec.symbol] = 0

                    emit("balance_update", {"balance": self.balance})
                    break

        # If HIBT settlement unavailable, use time-based expiry
        # (only in SHADOW mode — LIVE requires HIBT settlement)
        if self.run_mode_mgr.mode != RunMode.LIVE:
            self._settle_shadow_orders()

    def _settle_shadow_orders(self):
        """Time-based settlement for SHADOW mode (no real orders)."""
        now = time.time()
        # Get pending SHADOW records from ledger
        pending = [r for r in self.trade_ledger.get_pending_settlements()
                   if r.run_mode == "SHADOW" and r.signal_time_ms > 0]
        for rec in pending:
            elapsed_ms = now * 1000 - rec.signal_time_ms
            settle_ms = rec.expiry_minutes * 60000 + 30000
            if elapsed_ms < settle_ms:
                continue

            rt = self._realtime_feed.get_realtime_price(rec.symbol) if self._realtime_feed else None
            settle_price = rt.price if rt else rec.gate_price_at_signal

            if rec.direction == "CALL":
                is_win = settle_price > rec.gate_price_at_signal
            else:
                is_win = settle_price < rec.gate_price_at_signal
            is_tie = abs(settle_price - rec.gate_price_at_signal) < 0.0001 * rec.gate_price_at_signal

            payout = rec.payout_call if rec.direction == "CALL" else rec.payout_put
            if is_tie:
                pnl = 0.0
                result = "TIE"
            elif is_win:
                pnl = rec.stake * payout
                result = "WIN"
            else:
                pnl = -rec.stake
                result = "LOSS"

            self.trade_ledger.mark_settled(
                rec.trade_id, result=result, pnl=pnl,
                settlement_source="gate_price_proxy",
                hibt_close_price=settle_price,
            )

            if not is_tie:
                self.risk_manager.record_result(is_win)
                self.calibrator.update(rec.calibrated_probability, 1 if is_win else 0)

            emit("shadow_order_settled", {
                "symbol": rec.symbol,
                "result": result,
                "pnl": round(pnl, 4),
                "trade_id": rec.trade_id,
                "settlement": "SHADOW_SETTLED",
                "settle_price": settle_price,
                "entry_price": rec.gate_price_at_signal,
            })

    # ═══════════════════════════════════════════════════════════
    # Slow Context (15m)
    # ═══════════════════════════════════════════════════════════

    def _update_slow_context(self, sym: str, df_1m: pd.DataFrame):
        """Update slow context every 15 minutes."""
        if df_1m is None or len(df_1m) < 15:
            return
        last_bar = df_1m.index[-1]
        last_update = self._last_slow_update.get(sym, 0)
        if hasattr(last_bar, 'timestamp'):
            bar_ts = last_bar.timestamp()
            if bar_ts - last_update < 840:
                return
            self._last_slow_update[sym] = bar_ts

        # Use 15m candles
        df_15m = None
        if self._realtime_feed:
            df_15m = self._realtime_feed.get_klines(sym, "15m")

        if df_15m is None or len(df_15m) < 50:
            # Aggregate from 1m
            df_15m_raw = df_1m.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        else:
            df_15m_raw = df_15m

        indicators = {"ADX": 20.0, "RSI": 50.0, "BB_Pos": 0.5, "bb_width": 0.02,
                       "volatility_ratio": 1.0, "ATR_pct": 0.003, "price_vs_MA20": 0.0,
                       "MACD": 0.0, "MA_trend": 0.0, "VWAP_dist": 0.0, "vol_ratio": 1.0, "CCI": 0.0}

        if df_15m_raw is not None and len(df_15m_raw) >= 50:
            closes = df_15m_raw["close"].values.astype(float)
            eps = 1e-10
            mid = float(np.mean(closes[-20:]))
            std = float(np.std(closes[-20:]))
            indicators["ATR_pct"] = float(std / max(mid, eps))
            indicators["ADX"] = min(40.0, 20.0 + abs(float(np.mean(np.diff(closes[-20:])) / max(np.mean(closes[-20:]), eps) * 10000)))
            indicators["price_vs_MA20"] = float((closes[-1] - mid) / max(mid, eps))
            indicators["BB_Pos"] = max(0.0, min(1.0, float((closes[-1] - (mid - 2*std)) / max(4*std, eps))))
            indicators["volatility_ratio"] = max(0.5, min(3.0, float(std / max(mid, eps) * 100)))
            indicators["bb_width"] = float(std / max(mid, eps))

        row = {"ret_1": 0.0, "ret_3": 0.0, "ret_6": 0.0, "body_pct": 0.3}

        try:
            predictions = self.expert_manager.predict_all(sym, indicators, row)
            regime = self.regime_detector.detect(indicators)
            ensemble = self.expert_manager.ensemble(predictions, regime)
            self._slow_context[sym] = {
                "probability": ensemble.ensemble_probability,
                "regime": regime.regime,
                "trend_strength": float(indicators.get("ADX", 20)) / 40.0,
                "volatility": float(indicators.get("volatility_ratio", 1.0)),
            }
        except Exception:
            self._slow_context[sym] = {
                "probability": 0.50, "regime": "RANGE",
                "trend_strength": 0.0, "volatility": 0.5,
            }

    # ═══════════════════════════════════════════════════════════
    # Main Tick + Helpers
    # ═══════════════════════════════════════════════════════════

    def tick(self):
        try:
            self._check_settlement()
            self._run_fast_entry_scan()
            self._emit_funnel_report()
        except Exception as e:
            emit("error", {"msg": f"Error: {str(e)[:200]}"})

    def get_status(self) -> dict:
        cal_ready = self.calibrator.is_ready()
        settled = self.trade_ledger.get_settled_count()
        total = settled["wins"] + settled["losses"]
        wr = f"{(settled['wins'] / total * 100):.1f}%" if total > 0 else "0.0%"
        profit = self.balance - self.start_balance

        run_mode = self.run_mode_mgr.mode
        symbol_modes = {}
        for sym in list(config.SHADOW_SYMBOL_MODE.keys()):
            symbol_modes[sym] = self.run_mode_mgr.get_symbol_mode(sym)

        return {
            "state": "running" if self.running else "stopped",
            "balance": self.balance,
            "wins": settled["wins"], "losses": settled["losses"],
            "winRate": wr,
            "activeTrades": len(self.active_hibt_order_ids),
            "maxConcurrentTrades": self.risk_manager.max_concurrent_orders,
            "consecutiveLosses": self.risk_manager.get_status()["consecutive_losses"],
            "currentBet": config.MIN_ORDER_USD,
            "betMode": "ev_based",
            "profit": round(profit, 2),
            "runMode": run_mode.value,
            "calibrationReady": cal_ready,
            "healthTradeCount": settled["total"],
            "liveGate": self.run_mode_mgr.get_live_gate().check(
                calibrator_ready=cal_ready,
                data_fresh=True,
                model_valid=self._fast_model_loaded,
                has_pending_orders=len(self.trade_ledger.get_pending_settlements()) > 0,
                risk_paused=self.risk_manager.get_status()["paused"],
                settlement_available=self.reconciler.can_settle_via_hibt(),
                payout_available=(self.reconciler.get_available_payout("BTCUSDT")[0] is not None),
            ).__dict__ if hasattr(self.run_mode_mgr.get_live_gate().check, '__dict__') else {},
            "symbolModes": symbol_modes,
            "fastScanCount": self._fast_scan_count,
            "fastScanInterval": config.FAST_SCAN_INTERVAL_SECONDS,
            "fastModelLoaded": self._fast_model_loaded,
            "risk": self.risk_manager.get_status(),
        }

    def set_run_mode(self, mode: str):
        """Set run mode from frontend command."""
        if mode == "paper":
            self.run_mode_mgr.set_paper()
        elif mode == "live":
            self.run_mode_mgr.attempt_live()
        else:
            self.run_mode_mgr.set_shadow()

    def _strat_funnel_count(self, symbol: str, stage: str):
        if symbol not in self._strat_funnel:
            self._strat_funnel[symbol] = {}
        self._strat_funnel[symbol][stage] = self._strat_funnel[symbol].get(stage, 0) + 1

    def _emit_funnel_report(self):
        now = time.time()
        if now - self._funnel_last_report < 60:
            return
        self._funnel_last_report = now
        emit("funnel", {
            "type": "system",
            "fast_scans": self._fast_scan_count,
        })
        for sym in sorted(self._strat_funnel.keys()):
            f = self._strat_funnel[sym]
            emit("funnel", {"type": "strategy", "symbol": sym,
                "reject_insufficient_ev": f.get("reject_insufficient_ev", 0),
                "reject_payout_not_verified": f.get("reject_payout_not_verified", 0),
                "decision_passed": f.get("decision_passed", 0),
                "shadow_trade": f.get("shadow_trade", 0),
                "risk_symbol_disabled": f.get("risk_symbol_disabled", 0),
                "risk_consecutive_loss_pause": f.get("risk_consecutive_loss_pause", 0),
                "risk_daily_loss_limit": f.get("risk_daily_loss_limit", 0),
                "risk_daily_max_trades": f.get("risk_daily_max_trades", 0),
                "risk_max_concurrent_orders": f.get("risk_max_concurrent_orders", 0),
                "cooldown_rejected": f.get("cooldown_rejected", 0),
                "concentration_rejected": f.get("concentration_rejected", 0),
                "max_hourly_rejected": f.get("max_hourly_rejected", 0),
            })
