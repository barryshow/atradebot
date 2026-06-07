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
FIXED_BET = float(os.getenv("FIXED_BET", "3"))
HOLD_MINUTES = int(os.getenv("HOLD_MINUTES", "5"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "999"))  # 不限制
TRADE_COOLDOWN_SEC = int(os.getenv("TRADE_COOLDOWN_SEC", "120"))
REJECT_COOLDOWN_SEC = int(os.getenv("REJECT_COOLDOWN_SEC", "30"))

# --- 本金管理: 两阶段 + Turbo加速 ---
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "14"))   # 14U启动金
BOOTSTRAP_TARGET = float(os.getenv("BOOTSTRAP_TARGET", "26"))
BOOTSTRAP_MODE = os.getenv("BOOTSTRAP_MODE", "turbo")         # "normal" 或 "turbo"
BET_MODE = os.getenv("BET_MODE", "fixed")
BET_BASE = int(FIXED_BET)
FIXED_BET_MIN = int(os.getenv("FIXED_BET_MIN", "3"))
FIXED_BET_MAX = int(os.getenv("FIXED_BET_MAX", "15"))

# --- 凯利公式参数 (启动成功后启用) ---
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.50"))   # 半凯利(保守), 可调
BET_MIN = int(os.getenv("BET_MIN", "3"))                       # 最低3U(凯利)
BET_MAX = int(os.getenv("BET_MAX", "50"))                      # 上限(凯利)

# --- 启动成功条件 ---
BOOTSTRAP_TARGET = float(os.getenv("BOOTSTRAP_TARGET", "26"))

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
CONFLUENCE_MIN = float(os.getenv("CONFLUENCE_MIN", "0.30"))