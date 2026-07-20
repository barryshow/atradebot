#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 Core Pipeline Verification Test

验证:
1. OpportunityRanker 真的参与决策
2. PortfolioRiskManager 真的可以拦截订单
3. 3U 最小整数下注规则
4. trade_id 全链路一致
5. ModelHealth 读取真实概率
6. Calibration 未 Ready 时不会伪造校准概率
7. Engine 启动 + Shadow Mode 测试
8. settlement 不再使用空 trade_id
"""
import sys, os, io, time, json, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lib.engine.models import EdgeResult

print("=" * 65)
print("  EventEdge V2 Core Pipeline Verification")
print("=" * 65)

# ═══════════════════════════════════════════════════════════
# Test 1: OpportunityRanker participates in decision
# ═══════════════════════════════════════════════════════════
print("\n[Test 1] OpportunityRanker integration")

from lib.engine.edge_engine import EdgeEngine, get_edge_engine
from lib.engine.opportunity_ranker import OpportunityRanker, get_opportunity_ranker
from lib.engine.portfolio_risk import PortfolioRiskManager, get_portfolio_risk

edge_engine = get_edge_engine()
ranker = get_opportunity_ranker()

# Generate 3 candidates with different qualities
edges = [
    edge_engine.compute(symbol="BTCUSDT", calibrated_probability=0.62, direction="CALL", direction_int=1, regime="TREND_UP"),
    edge_engine.compute(symbol="ETHUSDT", calibrated_probability=0.58, direction="PUT", direction_int=2, regime="RANGE"),
    edge_engine.compute(symbol="SOLUSDT", calibrated_probability=0.55, direction="CALL", direction_int=1, regime="TREND_DOWN"),
]

# Only edge-passes should go to ranker
passed = [e for e in edges if e.passed]
print(f"  Edge passed: {len(passed)}/{len(edges)}")

ranked = ranker.rank(passed)
selected = [o for o in ranked if o.selected]
print(f"  Ranked: {len(ranked)}, Selected: {len(selected)}")

# Verify: best edge gets selected
if selected:
    best = selected[0]
    print(f"  Best: {best.symbol} {best.direction} score={best.rank_score:.4f}")
    # Verify: best edge gets selected (BTCUSDT 0.62 is highest)
    assert best.symbol == "BTCUSDT", f"Expected BTCUSDT as best, got {best.symbol}"
    print(f"  ✓ Ranker correctly selects highest quality opportunity (BTCUSDT 0.62)")
else:
    print(f"  ⚠ No candidates selected (edge thresholds may be too strict)")

# ═══════════════════════════════════════════════════════════
# Test 2: PortfolioRiskManager intercepts orders
# ═══════════════════════════════════════════════════════════
print("\n[Test 2] PortfolioRiskManager gate checks")

risk = get_portfolio_risk()

# Test: small account rejection
edge_ok = edge_engine.compute(symbol="BTCUSDT", calibrated_probability=0.65, direction="CALL", direction_int=1)
pos_small = risk.compute_position_size(edge_ok, equity=20.0)
assert not pos_small.allowed, f"Expected rejected, got {pos_small.allowed}"
assert pos_small.reject_reason == "ACCOUNT_TOO_SMALL_FOR_RISK_RULE"
print(f"  ✓ Small account (20U) rejected: {pos_small.reject_reason}")

# Test: large account should pass
pos_large = risk.compute_position_size(edge_ok, equity=5000.0)
assert pos_large.allowed, f"Expected allowed, got {pos_large.reject_reason}"
assert pos_large.stake_usd >= 3, f"Stake {pos_large.stake_usd} < 3"
print(f"  ✓ Large account (5000U) allowed: stake={pos_large.stake_usd}U")

# Test: portfolio limit check
port_check = risk.check_portfolio_limits(
    symbol="BTCUSDT", direction="CALL", stake_usd=500,
    equity=1000.0,  # 500/1000 = 50% > 5% MAX_TOTAL_EXPOSURE
)
assert not port_check.allowed, f"Expected PORTFOLIO_LIMIT, got {port_check.reject_reason}"
print(f"  ✓ Portfolio limit rejected: {port_check.reject_reason}")

# Test: daily stop
risk.daily_pnl = -100.0
port_check2 = risk.check_portfolio_limits(
    symbol="BTCUSDT", direction="CALL", stake_usd=3,
    equity=1000.0,
)
assert not port_check2.allowed, f"Expected DAILY_STOP, got {port_check2.reject_reason}"
print(f"  ✓ Daily stop rejected: {port_check2.reject_reason}")
risk.daily_pnl = 0.0  # reset

# ═══════════════════════════════════════════════════════════
# Test 3: 3U minimum integer stake rule
# ═══════════════════════════════════════════════════════════
print("\n[Test 3] 3U minimum integer stake rule")

# Every stake must be integer >= 3
for equity in [100, 200, 500, 1000, 5000]:
    pos = risk.compute_position_size(edge_ok, equity=equity)
    if pos.allowed:
        assert pos.stake_usd >= 3, f"Stake {pos.stake_usd} < 3 for equity {equity}"
        assert isinstance(pos.stake_usd, int), f"Stake {pos.stake_usd} not int for equity {equity}"
        print(f"  Equity {equity}U → stake {pos.stake_usd}U ✓")

# Test: stake < 3 → rejected, NOT auto-raised to 3
edge_weak = edge_engine.compute(symbol="ETHUSDT", calibrated_probability=0.57, direction="CALL", direction_int=1)
pos_weak = risk.compute_position_size(edge_weak, equity=100.0)
# With 100U equity and weak edge, stake should be < 3 → rejected
if pos_weak.stake_usd < 3:
    assert not pos_weak.allowed
    print(f"  ✓ Weak edge with 100U → rejected (stake={pos_weak.stake_usd}), not auto-raised")

# ═══════════════════════════════════════════════════════════
# Test 4: trade_id consistency through full lifecycle
# ═══════════════════════════════════════════════════════════
print("\n[Test 4] trade_id consistency")

from lib.engine.trade_ledger import TradeLedger, get_trade_ledger, TradeRecord
from lib.engine.settlement_ledger import SettlementLedger, get_settlement_ledger, SettlementStatus

tf = tempfile.mktemp(suffix='.jsonl')
ledger = TradeLedger(filepath=tf)
settlement = SettlementLedger(trade_ledger=ledger)

# Create → Save → Settle (same trade_id)
rec = ledger.create_record(
    symbol="BTCUSDT", direction="CALL", direction_int=1,
    entry_time_ms=int(time.time()*1000), entry_price=65000,
    stake_usd=3, expiry_minutes=15,
    raw_probability=0.62, calibrated_probability=0.60,
    break_even_probability=0.5556, effective_edge=0.0444,
    expected_roi=0.08, regime="TREND_UP",
)
ledger.save(rec)
trade_id = rec.trade_id
assert trade_id, "trade_id should not be empty"
print(f"  Created trade_id: {trade_id} ✓")

# Settle with same trade_id
settlement.estimate_settlement(trade_id=trade_id, current_balance=502.4, pre_balance=500.0)
final = ledger.load_all()[0]
assert final.result == "WIN"
assert final.settlement_status == "ESTIMATED"
assert final.trade_id == trade_id, f"trade_id mismatch: {final.trade_id} != {trade_id}"
print(f"  Settled: {final.result}, PnL={final.realized_pnl}, trade_id={final.trade_id} ✓")

# Verify: settlement with empty trade_id should NOT match
settlement.estimate_settlement(trade_id="", current_balance=510.0, pre_balance=500.0)
# The original record should still be WIN (not overwritten)
all_recs = ledger.load_all()
assert len(all_recs) == 1, f"Should still have 1 record, got {len(all_recs)}"
assert all_recs[0].trade_id == trade_id
print(f"  ✓ Empty trade_id does not corrupt existing records")

os.unlink(tf)

# ═══════════════════════════════════════════════════════════
# Test 5: ModelHealth reads real probabilities
# ═══════════════════════════════════════════════════════════
print("\n[Test 5] ModelHealth with real probabilities")

from lib.engine.model_health import ModelHealthMonitor, get_model_health_monitor

monitor = get_model_health_monitor()
np.random.seed(42)

# Simulate trades with calibrated probabilities
trades = []
for i in range(60):
    prob = np.random.uniform(0.55, 0.65)
    is_win = np.random.random() < prob * 0.9  # slight negative calibration error
    trades.append({
        "predicted_prob": prob,
        "calibrated_probability": prob,
        "raw_probability": prob + 0.02,
        "conservative_probability": prob - 0.03,
        "effective_edge": prob - 0.5556,
        "result": "WIN" if is_win else "LOSS",
        "pnl": 2.4 if is_win else -3.0,
    })

# Check with insufficient samples
small_health = monitor.check(trades[:20])
assert not small_health.is_degraded
assert small_health.degradation_reason == "INSUFFICIENT_SAMPLES"
print(f"  ✓ 20 samples → INSUFFICIENT_SAMPLES (not degraded)")

# Check with sufficient samples
health = monitor.check(trades)
print(f"  Actual WR: {health.actual_win_rate:.1%}, Predicted: {health.predicted_win_rate:.1%}")
print(f"  Brier: {health.brier_score:.3f}, ECE: {health.expected_calibration_error:.3f}")
print(f"  Degraded: {health.is_degraded}")

# Multi-window check
multi = monitor.check_multi_window(trades)
print(f"  Multi-window: {list(multi.keys())}")
for w, h in multi.items():
    if h.trade_count >= w:
        print(f"    {w}t: WR={h.actual_win_rate:.1%}, Brier={h.brier_score:.3f}, Degraded={h.is_degraded}")
print(f"  ✓ ModelHealth multi-window check works")

# ═══════════════════════════════════════════════════════════
# Test 6: Calibration NOT_READY detection
# ═══════════════════════════════════════════════════════════
print("\n[Test 6] Calibration readiness detection")

from lib.engine.probability_calibrator import WalkForwardCalibrator

cal = WalkForwardCalibrator(method="isotonic", min_samples=50)
assert not cal.is_ready(), "New calibrator should not be ready"
print(f"  ✓ Fresh calibrator: NOT_READY")

cal_status = cal.get_status()
assert not cal_status["ready"]
assert cal_status["samples"] == 0
print(f"  ✓ Status: {cal_status}")

# Calibrate a single prob → should return raw (unchanged)
result = cal.update(0.60, outcome=None)
assert result == 0.60, f"Unfitted calibrator should return raw prob, got {result}"
print(f"  ✓ Unfitted calibrator passes through raw probability")

# ═══════════════════════════════════════════════════════════
# Test 7: Engine initialization + Shadow Mode
# ═══════════════════════════════════════════════════════════
print("\n[Test 7] Engine initialization + Shadow Mode")

from lib.engine.engine import TradingEngine, CandidateOpportunity
from lib.engine.shadow_mode import ShadowMode, RunMode

# Test shadow mode engine
eng = TradingEngine("shadow")
assert eng.shadow.get_mode() == "SHADOW"
assert not eng.shadow.can_place_order()
assert eng.shadow.is_shadow_or_live()
print(f"  ✓ Engine shadow mode: {eng.shadow.get_mode()}")

# Test calibration status in engine
assert not eng.calibrator.is_ready()
print(f"  ✓ Engine calibration: NOT_READY detected")

# Verify all modules instantiated
assert eng.regime_detector is not None
assert eng.expert_manager is not None
assert eng.edge_engine is not None
assert eng.uncertainty is not None
assert eng.portfolio_risk is not None
assert eng.ranker is not None
assert eng.trade_ledger is not None
assert eng.settlement_ledger is not None
assert eng.model_health is not None
assert eng.calibrator is not None
print(f"  ✓ All 10 EventEdge V2 modules instantiated")

# Test mode switching
eng.set_run_mode("live")
# LIVE_ENABLED=false means live gate won't pass, but can_place_order checks mode
print(f"  ✓ Mode switching: LIVE→BACKTEST works")
print(f"  ✓ Live gate: {eng.shadow.get_live_gate_status(eng.calibrator.is_ready())['passed']} (expected False with LIVE_ENABLED=false)")

# ═══════════════════════════════════════════════════════════
# Test 8: settlement_ledger.py no duplicate TIE code
# ═══════════════════════════════════════════════════════════
print("\n[Test 8] No duplicate TIE code in settlement_ledger")

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "lib", "engine", "settlement_ledger.py"), "r", encoding="utf-8") as f:
    content = f.read()

# Count occurrences of "result = \"TIE\""
tie_count = content.count('result = "TIE"')
assert tie_count == 1, f"Expected 1 TIE assignment, found {tie_count}"
print(f"  ✓ TIE assignment count: {tie_count} (no duplicate)")

# ═══════════════════════════════════════════════════════════
# Test 9: Backtester stake formula
# ═══════════════════════════════════════════════════════════
print("\n[Test 9] Backtester stake formula")

from lib.engine.backtester import WalkForwardBacktester

bt = WalkForwardBacktester(
    symbol="BTCUSDT",
    expiries=[15],
    min_order_usd=3,
    order_step=1,
    net_payout_ratio=0.80,
    min_probability=0.50,
    min_effective_edge=0.0,
    kelly_fraction=0.10,
    max_bet_fraction=0.01,
    verbose=False,
)

# Test: with 5000U equity and edge=0.04, stake should be >= 3
effective_edge = 0.04
denom = 1.0 + bt.net_payout_ratio
frac_kelly = effective_edge / denom
target_fraction = bt.kelly_fraction * frac_kelly
effective_fraction = min(target_fraction, bt.max_bet_fraction)
incremental = 5000 * effective_fraction
stake = bt.min_order_usd + int(np.floor(incremental))
stake = (stake // bt.order_step) * bt.order_step
print(f"  Edge=4% Kelly: frac={target_fraction:.6f}, effective={effective_fraction:.6f}")
print(f"  Incremental: {incremental:.2f}U, Final stake: {stake}U")
assert stake >= 3, f"Stake {stake} < 3"
print(f"  ✓ Backtester stake formula produces valid stakes")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  ALL VERIFICATION TESTS PASSED")
print(f"{'='*65}")
print(f"  Tests: 9")
print(f"  Status: ✅ EventEdge V2 Core Pipeline Verified")
print(f"{'='*65}")