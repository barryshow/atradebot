# -*- coding: utf-8 -*-
import os
from pathlib import Path

# --- Load .env.local if present (so env vars work on VPS without pm2 config) ---
_env_local = Path(__file__).resolve().parent.parent.parent / ".env.local"
if _env_local.exists():
    for line in _env_local.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("\"'")
        if key and not os.environ.get(key):  # Don't override actual env vars
            os.environ[key] = val

# --- AI ---
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_URL = os.getenv("AI_URL", "https://api.siliconflow.cn/v1/chat/completions")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-ai/DeepSeek-V3")

# --- Exchange ---
HIBT_TOKEN = os.getenv("HIBT_TOKEN", "")
HIBT_AUTHORIZATION = os.getenv("HIBT_AUTHORIZATION", "")     # Authorization 头
HIBT_X_AUTH_TOKEN = os.getenv("HIBT_X_AUTH_TOKEN", "")       # x-auth-token 头
HIBT_BGET_KEY = os.getenv("HIBT_BGET_KEY", "")               # bget_key / vKey
HIBT_BGET_ID = os.getenv("HIBT_BGET_ID", "")                 # bget_id / memberId
HIBT_V = os.getenv("HIBT_V", "")                              # v 参数（静态兜底）
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")

# --- Trading (激进) ---
SYMBOLS = os.getenv("TRADE_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
PAYOUT_RATES = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}

# 持仓时间（分钟） — 建议15或30，避免短期无序波动
HOLD_MINUTES = int(os.getenv("HOLD_MINUTES", "15"))

# K线检测粒度（分钟） — 每分钟检查一次是否有新信号
CANDLE_INTERVAL_MIN = int(os.getenv("CANDLE_INTERVAL_MIN", "1"))

# 特征计算聚合粒度（分钟） — 用更长时间窗口算特征，保持模型兼容
FEATURE_INTERVAL_MIN = int(os.getenv("FEATURE_INTERVAL_MIN", "15"))

MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "999"))  # 不限制
TRADE_COOLDOWN_SEC = int(os.getenv("TRADE_COOLDOWN_SEC", "120"))

# --- 本金管理: 凯利滚仓 ---
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "14"))

# --- 凯利公式参数 ---
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.50"))   # 半凯利(保守), 可调
BET_MIN = int(os.getenv("BET_MIN", "3"))                       # 最低3U
BET_MAX = int(os.getenv("BET_MAX", "50"))                      # 上限

# --- Data ---
CSV_FILE = os.getenv("RADAR_CSV_PATH", "./hibt_ticks.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# --- Risk Gate Thresholds ---
BB_DEAD_ZONE_LOW = float(os.getenv("BB_DEAD_ZONE_LOW", "0.4"))
BB_DEAD_ZONE_HIGH = float(os.getenv("BB_DEAD_ZONE_HIGH", "0.6"))
ADX_OSCILLATING = float(os.getenv("ADX_OSCILLATING_THRESHOLD", "35"))
ADX_EXTREME = float(os.getenv("ADX_EXTREME_THRESHOLD", "44"))
BB_EXTREME_HIGH = float(os.getenv("BB_EXTREME_HIGH", "0.72"))
BB_EXTREME_LOW = float(os.getenv("BB_EXTREME_LOW", "0.28"))
BB_OSCILLATE_LONG = float(os.getenv("BB_OSCILLATE_LONG", "0.25"))
BB_OSCILLATE_SHORT = float(os.getenv("BB_OSCILLATE_SHORT", "0.75"))
MIN_PROBABILITY = float(os.getenv("MIN_PROBABILITY", "0.30"))

# --- SignalValidator 配置 ---
# L0: 防接刀（Anti-Knife Filter）
ANTI_KNIFE_BARS = int(os.getenv("ANTI_KNIFE_BARS", "5"))           # 检测K线数
ANTI_KNIFE_BODY_RATIO = float(os.getenv("ANTI_KNIFE_BODY_RATIO", "0.6"))  # 实体占比阈值
ANTI_KNIFE_CCI = float(os.getenv("ANTI_KNIFE_CCI", "100"))         # CCI极端阈值

# L1: 硬性概率门槛（被 predictor.py 的 0.62 覆盖，此处保留为冗余）
HARD_PROB_THRESHOLD = float(os.getenv("HARD_PROB_THRESHOLD", "0.62"))

# L2: 极值翻转概率重置
REVERSAL_PROB = float(os.getenv("REVERSAL_PROB", "0.55"))          # 翻转后固定胜率

# --- RiskManager 配置 ---
# L3: 共振分门槛
CONFLUENCE_MIN = float(os.getenv("CONFLUENCE_MIN", "0.65"))        # 从0.30提升到0.65

# L4: 冷却
REJECT_COOLDOWN_SEC = int(os.getenv("REJECT_COOLDOWN_SEC", "60"))  # 被拒后冷却
SETTLEMENT_COOLDOWN_SEC = int(os.getenv("SETTLEMENT_COOLDOWN_SEC", "60"))  # 结算后额外冷却

# L5: 加仓（已移除 — 二元期权每单独立，不做加仓）

# --- Circuit Breaker ---
CONSECUTIVE_LOSS_PAUSE_SEC = int(os.getenv("CONSECUTIVE_LOSS_PAUSE_SEC", "300"))
CONSECUTIVE_LOSS_HALT = int(os.getenv("CONSECUTIVE_LOSS_HALT", "8"))
RECENT_WINDOW = int(os.getenv("RECENT_WINDOW", "10"))

# --- Streak Adjustments (fixed模式下保留) ---
WIN_STREAK_BOOST = float(os.getenv("WIN_STREAK_BOOST", "1.5"))
LOSE_STREAK_CUT = float(os.getenv("LOSE_STREAK_CUT", "0.7"))
WIN_STREAK_TRIGGER = int(os.getenv("WIN_STREAK_TRIGGER", "2"))
LOSE_STREAK_TRIGGER = int(os.getenv("LOSE_STREAK_TRIGGER", "2"))

# --- Session ---
ACTIVE_HOURS_START = int(os.getenv("ACTIVE_HOURS_START", "0"))
ACTIVE_HOURS_END = int(os.getenv("ACTIVE_HOURS_END", "24"))

# ── Phase 2: Trade/Settlement Ledger ──
TRADE_LEDGER_PATH = os.getenv("TRADE_LEDGER_PATH", "./data/trade_ledger.jsonl")

# ── Shadow Mode: Per-symbol activation ──
# "SHADOW_ACTIVE" = simulate trades, record results
# "OBSERVE_ONLY" = generate predictions, record, but NO trades (even simulated)
# "DISABLED" = skip entirely
SHADOW_SYMBOL_MODE = {
    "BTCUSDT": os.getenv("SHADOW_BTC_MODE", "OBSERVE_ONLY"),
    "ETHUSDT": os.getenv("SHADOW_ETH_MODE", "SHADOW_ACTIVE"),
    "SOLUSDT": os.getenv("SHADOW_SOL_MODE", "SHADOW_ACTIVE"),
}

# ── LIVE Gate ──
LIVE_ENABLED = os.getenv("LIVE_ENABLED", "false").lower() == "true"
# LIVE requires: calibration READY + shadow validation PASSED + LIVE_ENABLED=true
LIVE_REQUIRE_CALIBRATION = os.getenv("LIVE_REQUIRE_CALIBRATION", "false").lower() == "true"  # SOFT WARNING
LIVE_REQUIRE_SHADOW_VALIDATION = os.getenv("LIVE_REQUIRE_SHADOW_VALIDATION", "false").lower() == "true"  # SOFT WARNING

# ── HARD BLOCKS (always enforced) ──
LIVE_HARD_REQUIRE_DATA = os.getenv("LIVE_HARD_REQUIRE_DATA", "true").lower() == "true"
LIVE_HARD_REQUIRE_BALANCE = os.getenv("LIVE_HARD_REQUIRE_BALANCE", "true").lower() == "true"

# ── Shadow Graduation Criteria ──
SHADOW_MIN_DAYS = int(os.getenv("SHADOW_MIN_DAYS", "7"))
SHADOW_MIN_TRADES = int(os.getenv("SHADOW_MIN_TRADES", "200"))
SHADOW_MIN_WIN_RATE = float(os.getenv("SHADOW_MIN_WIN_RATE", "0.50"))

# ── Shadow Record Path ──
SHADOW_RECORD_PATH = os.getenv("SHADOW_RECORD_PATH", "./data/shadow_candidates.jsonl")

# ── Integer Order Constraints ──
MIN_ORDER_USD = int(os.getenv("MIN_ORDER_USD", "3"))           # HIBT 最低下注
ORDER_AMOUNT_STEP = int(os.getenv("ORDER_AMOUNT_STEP", "1"))   # 必须是整数 USDT

# ── Portfolio Risk ──
MAX_BET_FRACTION = float(os.getenv("MAX_BET_FRACTION", "0.01"))   # 单笔最大净值比例 (1%)
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.05"))  # 最大总敞口 (5%)
MAX_CORRELATED_EXPOSURE = float(os.getenv("MAX_CORRELATED_EXPOSURE", "0.03"))  # 最大关联敞口 (3%)
DAILY_STOP = float(os.getenv("DAILY_STOP", "0.05"))              # 日亏损限制 (5%)
WEEKLY_DRAWDOWN_STOP = float(os.getenv("WEEKLY_DRAWDOWN_STOP", "0.10"))  # 周回撤限制 (10%)

# ── Small Account ──
SMALL_ACCOUNT_MAX_BET_FRACTION = float(os.getenv("SMALL_ACCOUNT_MAX_BET_FRACTION", "0.0"))  # 0=禁止自动放宽

# ── Edge Engine ──
MIN_EFFECTIVE_EDGE = float(os.getenv("MIN_EFFECTIVE_EDGE", "0.02"))       # 最小有效优势 (2%)
MIN_EXPECTED_ROI = float(os.getenv("MIN_EXPECTED_ROI", "0.005"))           # 最小期望ROI (0.5%)

# ── Uncertainty ──
DEFAULT_UNCERTAINTY_MARGIN = float(os.getenv("DEFAULT_UNCERTAINTY_MARGIN", "0.02"))    # 默认不确定性折扣
DEFAULT_CALIBRATION_MARGIN = float(os.getenv("DEFAULT_CALIBRATION_MARGIN", "0.01"))    # 默认校准折扣
DEFAULT_DEGRADATION_MARGIN = float(os.getenv("DEFAULT_DEGRADATION_MARGIN", "0.00"))    # 默认模型退化折扣

# ── Model Health ──
MIN_HEALTH_SAMPLE = int(os.getenv("MIN_HEALTH_SAMPLE", "50"))            # 最小健康检查样本
MODEL_DEGRADED_WIN_RATE_DELTA = float(os.getenv("MODEL_DEGRADED_WIN_RATE_DELTA", "0.10"))  # 实际胜率低于预测10%触发
MODEL_DEGRADED_BRIER_THRESHOLD = float(os.getenv("MODEL_DEGRADED_BRIER_THRESHOLD", "0.30"))  # Brier Score > 0.30 触发

# ── Available Expiries (已验证) ──
AVAILABLE_EXPIRIES = [int(x) for x in os.getenv("AVAILABLE_EXPIRIES", "15").split(",")]  # 15m 已验证
EXPIRY_DISCOVERY_UNAVAILABLE = os.getenv("EXPIRY_DISCOVERY_UNAVAILABLE", "true").lower() == "true"  # 启动时警告

# ── Payout Source ──
PAYOUT_SOURCE = os.getenv("PAYOUT_SOURCE", "hardcoded")  # "api" / "hardcoded" / "estimated"