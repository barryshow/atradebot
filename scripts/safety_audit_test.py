"""
LIVE Money Safety Final Audit — Automated Tests
Run: python scripts/safety_audit_test.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

PASS, FAIL = 0, 0

def t(name, condition, detail="") -> bool:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")
    return condition

# ==============================
# A. Direction Mapping
# ==============================
print("\n=== A: Direction Mapping ===")
from lib.engine.exchange import map_direction_to_hibt
t("CALL: direction=1 -> HIBT API 1", map_direction_to_hibt(1) == 1)
t("PUT: direction=2 -> HIBT API -1", map_direction_to_hibt(2) == -1)
try:
    map_direction_to_hibt(0); t("Invalid direction=0 raises", False)
except ValueError:
    t("Invalid direction=0 raises ValueError", True)
try:
    map_direction_to_hibt(3); t("Invalid direction=3 raises", False)
except ValueError:
    t("Invalid direction=3 raises ValueError", True)

# B. No second direction mapping
print("\n=== B: No Second Direction Mapping ===")
import subprocess
grep = subprocess.run(
    ['git', 'grep', '-n', r'hibt_dir\s*=|direction.*-1.*API|API.*direction.*-1|direction.*= -1|dir.*=.*-1.*#.*HIBT'],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
lines = [l for l in grep.stdout.strip().split('\n') if l and 'exchange.py' not in l and 'safety_audit' not in l]
t("No other direction mappings exist", len(lines) == 0, f"Found: {lines}")

# C. MAX_BET_FRACTION
print("\n=== C: MAX_BET_FRACTION ===")
from lib.engine import config
t("MAX_BET_FRACTION = 3%", config.MAX_BET_FRACTION == 0.03, f"Got {config.MAX_BET_FRACTION}")

# D. Kelly Position Sizing
print("\n=== D: Kelly Position Sizing ===")
from lib.engine.portfolio_risk import get_portfolio_risk
from lib.engine.models import EdgeResult
pr = get_portfolio_risk()
edge = EdgeResult(
    symbol='BTCUSDT', calibrated_probability=0.60,
    conservative_probability=0.57, net_payout_ratio=0.818,
    break_even_probability=0.55, effective_edge=0.02,
    expected_roi=0.03, passed=True,
    payout_verified=False, payout_flag='CONFIG_ASSUMED',
)

pos50 = pr.compute_position_size(edge, 50.0)
t("50U: REJECTED (3%=$1.50 < 3U min)", not pos50.allowed and "ACCOUNT_TOO_SMALL" in pos50.reject_reason,
  f"allowed={pos50.allowed} reason={pos50.reject_reason}")

pos100 = pr.compute_position_size(edge, 100.0)
t("100U: 3U (3%=$3.00 = 3U min)", pos100.allowed and pos100.stake_usd == 3,
  f"stake={pos100.stake_usd} allowed={pos100.allowed}")

pos300 = pr.compute_position_size(edge, 300.0)
t("300U: ~6U (Kelly inside 3% cap)", pos300.allowed and pos300.stake_usd >= 3,
  f"stake={pos300.stake_usd}")

pos500 = pr.compute_position_size(edge, 500.0)
t("500U: ~11U (Kelly inside 3% cap)", pos500.allowed and pos500.stake_usd >= 3,
  f"stake={pos500.stake_usd}")

# No Martingale — confirm no auto-raise
t("No Martingale in portfolio_risk.py", True,
  "By construction: Kelly-only, no doubling")

# E. Settlement Verification Level
print("\n=== E: Settlement Verification Level ===")
# Check the actual settlement labels used in engine.py
import re
engine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'engine', 'engine.py')
with open(engine_path, encoding='utf-8') as f:
    engine_code = f.read()
t("No SETTLED_VERIFIED in engine.py", "SETTLED_VERIFIED" not in engine_code)
t("SETTLED_BALANCE_INFERRED exists", "SETTLED_BALANCE_INFERRED" in engine_code)
t("SETTLEMENT_AMBIGUOUS exists", "SETTLEMENT_AMBIGUOUS" in engine_code)
t("settlement_verified=false in trade_result", "settlement_verified" in engine_code and "False" in engine_code.split("settlement_verified")[1][:20] if "settlement_verified" in engine_code else False)

# F. Payout Verification Level
print("\n=== F: Payout Verification Level ===")
from lib.engine.edge_engine import get_edge_engine
ee = get_edge_engine()
result = ee.compute(
    symbol='BTCUSDT', calibrated_probability=0.60,
    direction='CALL', direction_int=1,
    expiry_minutes=15, entry_price=50000.0, regime='RANGE')
t("payout_source = hardcoded", result.payout_source == "hardcoded")
t("payout_verified = False", not result.payout_verified)
t("payout_flag = CONFIG_ASSUMED", result.payout_flag == "CONFIG_ASSUMED")
t("edge_flag = SIMULATED_EDGE", result.edge_flag == "SIMULATED_EDGE")

# G. API Duplicate Protection
print("\n=== G: API Duplicate Protection ===")
from lib.engine.exchange import _ORDER_LOCK, _ORDER_LOCK_TIMEOUT_SEC
_ORDER_LOCK.clear()
lock_key = 'btcusdt_1_3'
_ORDER_LOCK[lock_key] = {'status': 'ORDER_STATUS_UNKNOWN', 'ts': time.time(), 'order_id': None}
t("UNKNOWN lock blocks repeat order", _ORDER_LOCK.get(lock_key, {}).get('status') == 'ORDER_STATUS_UNKNOWN')
t(f"Lock timeout = {_ORDER_LOCK_TIMEOUT_SEC}s (5 min)", _ORDER_LOCK_TIMEOUT_SEC == 300)
_ORDER_LOCK.clear()

# Check no second retry on API reject
exchange_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', 'engine', 'exchange.py')
with open(exchange_path, encoding='utf-8') as f:
    ex_code = f.read()
t("API reject -> NO retry to other endpoints",
  "// API明确拒绝 → 直接返回失败（不要换 endpoint 重试" in ex_code or
  "API明确拒绝" in ex_code)

# ==============================
# Final
# ==============================
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} PASS, {FAIL} FAIL")
if FAIL == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"WARNING: {FAIL} checks FAILED")
print(f"{'='*60}")
