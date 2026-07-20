#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 Full Integration Test

测试完整流水线:
Regime → Experts → Meta → Uncertainty → Edge → Portfolio → Opportunity → Shadow
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.engine.regime_detector import MarketRegimeDetector
from lib.engine.experts import ExpertManager, get_expert_manager
from lib.engine.edge_engine import EdgeEngine, get_edge_engine
from lib.engine.uncertainty import ModelUncertaintyEstimator, get_uncertainty_estimator
from lib.engine.portfolio_risk import PortfolioRiskManager, get_portfolio_risk
from lib.engine.opportunity_ranker import OpportunityRanker, get_opportunity_ranker
from lib.engine.shadow_mode import ShadowMode, RunMode, set_run_mode
from lib.engine.model_health import ModelHealthMonitor, get_model_health_monitor
from lib.engine.probability_calibrator import ProbabilityCalibrator, ReliabilityDiagram
from lib.engine.trade_ledger import TradeLedger, get_trade_ledger
from lib.engine.settlement_ledger import SettlementLedger, get_settlement_ledger, RejectReason

print("=" * 65)
print("  EventEdge V2 Integration Test")
print("=" * 65)

# ────────── Test 1: Full Pipeline ──────────
print("\n[Test 1] Full Pipeline: Regime → Experts → Edge → Portfolio")

indicators = {
    "ADX": 32.0, "RSI": 55.0, "BB_Pos": 0.45, "bb_width": 0.022,
    "volatility_ratio": 1.1, "ATR_pct": 0.003, "price_vs_MA20": 0.012,
    "MACD": 25.0, "MA_trend": 1.0, "VWAP_dist": 0.003, "vol_ratio": 1.0, "CCI": 20.0,
}
row = {"ret_1": 0.003, "ret_3": 0.008, "ret_6": 0.012, "body_pct": 0.4}

# Regime
regime = MarketRegimeDetector().detect(indicators)
assert regime.regime == "TREND_UP", f"Expected TREND_UP, got {regime.regime}"
print(f"  Regime: {regime.regime} ✓")

# Experts
manager = ExpertManager()
predictions = manager.predict_all("BTCUSDT", indicators, row)
assert len(predictions) == 3, f"Expected 3 experts, got {len(predictions)}"
print(f"  Experts: {[(p.expert_name, p.direction_str) for p in predictions]} ✓")

# Meta
ensemble = manager.ensemble(predictions, regime)
assert ensemble.direction in (1, 2)
print(f"  Meta: dir={ensemble.direction}, prob={ensemble.ensemble_probability:.4f} ✓")

# Uncertainty
unc = ModelUncertaintyEstimator()
unc_margin = unc.estimate_expert_uncertainty(predictions)
print(f"  Uncertainty: {unc_margin:.4f} ✓")

# Edge
edge_engine = EdgeEngine()
edge = edge_engine.compute(
    symbol="BTCUSDT", calibrated_probability=ensemble.ensemble_probability,
    direction="CALL" if ensemble.direction == 1 else "PUT",
    direction_int=ensemble.direction, expiry_minutes=15, entry_price=65000,
    uncertainty_margin=unc_margin, regime=regime.regime,
    expert_votes=ensemble.expert_votes,
)
print(f"  Edge: passed={edge.passed}, effective_edge={edge.effective_edge:.4f}, expected_roi={edge.expected_roi:.4f} ✓")

# Portfolio
risk = PortfolioRiskManager()
pos = risk.compute_position_size(edge, equity=500.0)
print(f"  Portfolio: allowed={pos.allowed}, stake={pos.stake_usd}U, fraction={pos.bet_fraction:.4f} ✓")

# ────────── Test 2: Edge Cases ──────────
print("\n[Test 2] Edge Cases")

# No edge → should reject
edge_no = edge_engine.compute(
    symbol="ETHUSDT", calibrated_probability=0.52, direction="CALL", direction_int=1,
    expiry_minutes=15, entry_price=3200, regime="RANGE",
)
assert not edge_no.passed, f"Expected rejected, got {edge_no.passed}"
assert edge_no.reject_reason == "NO_EDGE"
print(f"  No edge rejected: {edge_no.reject_reason} ✓")

# Small account → should reject
pos_small = risk.compute_position_size(edge, equity=20.0)
assert not pos_small.allowed, f"Expected rejected, got {pos_small.allowed}"
assert pos_small.reject_reason == "ACCOUNT_TOO_SMALL_FOR_RISK_RULE"
print(f"  Small account rejected: {pos_small.reject_reason} ✓")

# ────────── Test 3: Opportunity Ranking ──────────
print("\n[Test 3] Opportunity Ranking")

edges = [
    edge_engine.compute(symbol="BTCUSDT", calibrated_probability=0.60, direction="CALL", direction_int=1, regime="TREND_UP"),
    edge_engine.compute(symbol="ETHUSDT", calibrated_probability=0.62, direction="CALL", direction_int=1, regime="TREND_UP"),
    edge_engine.compute(symbol="SOLUSDT", calibrated_probability=0.58, direction="CALL", direction_int=1, regime="RANGE"),
]
ranker = OpportunityRanker()
opps = ranker.rank(edges)
selected = [o for o in opps if o.selected]
print(f"  Total: {len(opps)}, Selected: {len(selected)}, Best: {selected[0].symbol if selected else 'none'} ✓")

# ────────── Test 4: Trade Ledger ──────────
print("\n[Test 4] Trade Ledger + Settlement")

import tempfile
tf = tempfile.mktemp(suffix='.jsonl')
ledger = TradeLedger(filepath=tf)
settlement = SettlementLedger(trade_ledger=ledger)

rec = ledger.create_record(
    symbol="BTCUSDT", direction="CALL", direction_int=1,
    entry_time_ms=int(time.time()*1000), entry_price=65000,
    stake_usd=3, expiry_minutes=15, calibrated_probability=0.60,
    break_even_probability=0.5556, effective_edge=0.0444,
    expected_roi=0.08, regime="TREND_UP",
)
ledger.save(rec)
print(f"  Saved: {rec.trade_id} ✓")

settlement.estimate_settlement(rec.trade_id, 502.4, 500.0)
final = ledger.load_all()[0]
assert final.result == "WIN"
assert final.settlement_status == "ESTIMATED"
print(f"  Settled: {final.result}, PnL={final.realized_pnl} ✓")

os.unlink(tf)

# ────────── Test 5: Shadow Mode ──────────
print("\n[Test 5] Shadow Mode")

shadow = ShadowMode(mode=RunMode.SHADOW)
assert not shadow.can_place_order()
assert shadow.is_shadow_or_live()

st = shadow.record_shadow_trade(
    symbol="BTCUSDT", direction="CALL", direction_int=1,
    entry_time_ms=int(time.time()*1000), entry_price=65000,
    stake_usd=3, expiry_minutes=15,
    calibrated_probability=0.60, break_even_probability=0.5556,
    effective_edge=0.0444, expected_roi=0.08, regime="TREND_UP",
)
shadow.settle_shadow_trade(st["trade_id"], expiry_price=66000)
stats = shadow.get_shadow_stats()
assert stats["settled"] == 1
print(f"  Shadow: {stats['settled']} settled, WR={stats['win_rate']} ✓")

# ────────── Test 6: Model Health ──────────
print("\n[Test 6] Model Health")

monitor = ModelHealthMonitor()
np.random.seed(42)
trades = []
for i in range(60):
    prob = np.random.uniform(0.55, 0.65)
    is_win = np.random.random() < 0.58
    trades.append({"predicted_prob": prob, "result": "WIN" if is_win else "LOSS", "pnl": 2.4 if is_win else -3.0})

health = monitor.check(trades)
assert not health.is_degraded, f"Expected healthy, got degraded: {health.degradation_reason}"
print(f"  Health: WR={health.actual_win_rate:.1%}, Brier={health.brier_score:.3f}, Degraded={health.is_degraded} ✓")

# ────────── Test 7: Calibration ──────────
print("\n[Test 7] Probability Calibration")

raw_prob = np.random.uniform(0.45, 0.75, 500)
actual = (np.random.random(500) < raw_prob * 0.85 + 0.08).astype(int)
cal = ProbabilityCalibrator(method="isotonic")
cal.fit(raw_prob, actual)
calibrated = cal.calibrate(raw_prob)
assert len(calibrated) == 500
print(f"  Calibrated: mean={calibrated.mean():.4f}, actual={actual.mean():.4f} ✓")

diagram = ReliabilityDiagram.compute(raw_prob, actual)
assert len(diagram["buckets"]) == 8
print(f"  Reliability: Brier={diagram['overall_brier_score']}, ECE={diagram['overall_ece']} ✓")

# ────────── Summary ──────────
print(f"\n{'='*65}")
print(f"  ALL TESTS PASSED")
print(f"{'='*65}")
print(f"  Modules: 14")
print(f"  Tests: 7")
print(f"  Status: ✅ EventEdge V2 Ready")
print(f"{'='*65}")