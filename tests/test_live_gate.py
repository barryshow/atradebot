#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Prove that LIVE mode is blocked when preconditions are not met.

1. No HIBT settlement data → LIVE blocked
2. No real-time payout → LIVE blocked
3. No probability calibrator → LIVE blocked
4. ENABLE_LIVE_TRADING=false → LIVE blocked
5. Admin not confirmed → LIVE blocked
"""
import os
import sys
import io
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["ENABLE_LIVE_TRADING"] = "false"
os.environ["BACKTEST_MODE"] = "false"

from lib.engine.run_mode import RunMode, RunModeManager, LiveGate, get_run_mode_manager


def test_live_blocked_by_env():
    """Test: LIVE blocked when ENABLE_LIVE_TRADING is not 'true'."""
    mgr = RunModeManager()
    result = mgr.get_live_gate().check(
        calibrator_ready=True,
        data_fresh=True,
        model_valid=True,
        has_pending_orders=False,
        risk_paused=False,
        settlement_available=True,
        payout_available=True,
    )
    assert not result.passed, "LIVE should be blocked when ENABLE_LIVE_TRADING != true"
    assert "env_enabled" in result.all_checks
    assert not result.all_checks["env_enabled"]
    print("✓ test_live_blocked_by_env PASSED")


def test_live_blocked_by_admin():
    """Test: LIVE blocked when admin not confirmed."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr = RunModeManager()
    # Don't confirm admin
    result = mgr.get_live_gate().check(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True, payout_available=True,
    )
    assert not result.passed, "LIVE should be blocked without admin confirmation"
    print("✓ test_live_blocked_by_admin PASSED")


def test_live_blocked_by_calibrator():
    """Test: LIVE blocked when calibrator not ready."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr = RunModeManager()
    mgr.get_live_gate().confirm_admin()
    result = mgr.get_live_gate().check(
        calibrator_ready=False,  # NOT ready
        data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True, payout_available=True,
    )
    assert not result.passed, "LIVE should be blocked when calibrator is not ready"
    print("✓ test_live_blocked_by_calibrator PASSED")


def test_live_blocked_by_settlement():
    """Test: LIVE blocked when HIBT settlement unavailable."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr = RunModeManager()
    mgr.get_live_gate().confirm_admin()
    result = mgr.get_live_gate().check(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=False,  # HIBT settlement NOT available
        payout_available=True,
    )
    assert not result.passed, "LIVE should be blocked when settlement is unavailable"
    assert not result.all_checks.get("settlement_available", True)
    print("✓ test_live_blocked_by_settlement PASSED")


def test_live_blocked_by_payout():
    """Test: LIVE blocked when real-time payout unavailable."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr = RunModeManager()
    mgr.get_live_gate().confirm_admin()
    result = mgr.get_live_gate().check(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True,
        payout_available=False,  # Payout NOT available
    )
    assert not result.passed, "LIVE should be blocked when payout is unavailable"
    assert not result.all_checks.get("payout_available", True)
    print("✓ test_live_blocked_by_payout PASSED")


def test_live_blocked_by_kill_switch():
    """Test: LIVE blocked when kill switch is active."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    from lib.engine.run_mode import kill, reset_kill
    mgr = RunModeManager()
    mgr.get_live_gate().confirm_admin()
    kill()
    result = mgr.get_live_gate().check(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True, payout_available=True,
    )
    assert not result.passed, "LIVE should be blocked when kill switch is active"
    reset_kill()
    print("✓ test_live_blocked_by_kill_switch PASSED")


def test_all_passed_live():
    """Test: LIVE passes when ALL conditions met."""
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr = RunModeManager()
    mgr.get_live_gate().confirm_admin()
    result = mgr.get_live_gate().check(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True, payout_available=True,
    )
    assert result.passed, f"LIVE should pass when all conditions met, got: {result.hard_blocks}"
    assert len(result.hard_blocks) == 0
    assert all(v for v in result.all_checks.values()), f"All checks should be true: {result.all_checks}"
    print("✓ test_all_passed_live PASSED")


def test_default_mode_is_shadow():
    """Test: Default run mode is SHADOW."""
    # Reset singleton
    from lib.engine import run_mode
    run_mode._run_mode_manager = None
    mgr = get_run_mode_manager()
    assert mgr.mode == RunMode.SHADOW, f"Default mode should be SHADOW, got {mgr.mode}"
    assert not mgr.can_place_orders, "Should not be able to place orders in SHADOW"
    print("✓ test_default_mode_is_shadow PASSED")


def test_restart_resets_to_shadow():
    """Test: Creating new RunModeManager defaults to SHADOW."""
    mgr = RunModeManager()
    # Simulate: set to LIVE, then "restart" by creating new instance
    os.environ["ENABLE_LIVE_TRADING"] = "true"
    mgr.get_live_gate().confirm_admin()
    mgr.attempt_live(
        calibrator_ready=True, data_fresh=True, model_valid=True,
        has_pending_orders=False, risk_paused=False,
        settlement_available=True, payout_available=True,
    )
    assert mgr.mode == RunMode.LIVE, "Should be LIVE now"

    # "Restart" — new instance
    from lib.engine import run_mode
    run_mode._run_mode_manager = None
    new_mgr = get_run_mode_manager()
    assert new_mgr.mode == RunMode.SHADOW, f"After restart, mode should be SHADOW, got {new_mgr.mode}"
    print("✓ test_restart_resets_to_shadow PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  ATradeBot LIVE Gate Tests")
    print("=" * 60)
    tests = [
        test_live_blocked_by_env,
        test_live_blocked_by_admin,
        test_live_blocked_by_calibrator,
        test_live_blocked_by_settlement,
        test_live_blocked_by_payout,
        test_live_blocked_by_kill_switch,
        test_all_passed_live,
        test_default_mode_is_shadow,
        test_restart_resets_to_shadow,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
    print(f"\n{'='*60}")
    print(f"  {passed}/{len(tests)} tests PASSED")
    print(f"  LIVE gate {'IS FUNCTIONING' if passed == len(tests) else 'HAS ISSUES'}")
    print(f"{'='*60}")
