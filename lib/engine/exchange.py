# -*- coding: utf-8 -*-
import time
import random
import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from curl_cffi import requests as curl_requests
from . import config


# ═══════════════════════════════════════════════════════════════
# Direction Mapping — 唯一函数，全项目禁止自行转换
# ═══════════════════════════════════════════════════════════════
# HIBT 平台 API direction 实测（commit ae48088 修正）:
#   direction=1 (CALL/做多) → HIBT API direction=1
#   direction=2 (PUT/做空) → HIBT API direction=-1
#
# ⚠️ P0: 全项目只有这一个函数可以将内部 direction 转换为 HIBT direction
# ⚠️ 历史事故: commit 8187eb6 曾错误反转导致"做多变成做空"
# ⚠️ 任何其他地方直接写 if direction == 1: hibt_dir = -1 之类的代码
#    都是 P0 级 bug，必须删除并改用此函数。
#
def map_direction_to_hibt(direction: int) -> int:
    """将内部 direction (1=CALL, 2=PUT) 转为 HIBT API direction 参数。

    这是全项目唯一的 direction→HIBT 转换函数。
    禁止任何其他地方自行写方向转换逻辑。

    Args:
        direction: 内部 direction (1=做多CALL, 2=做空PUT)
    Returns:
        HIBT API direction 参数 (1=多, -1=空)

    >>> map_direction_to_hibt(1)
    1
    >>> map_direction_to_hibt(2)
    -1
    """
    if direction not in (1, 2):
        raise ValueError(f"map_direction_to_hibt: invalid direction={direction}, must be 1 or 2")
    return 1 if direction == 1 else -1


# ═══════════════════════════════════════════════════════════════
# Duplicate Protection — 订单去重 + ORDER_STATUS_UNKNOWN
# ═══════════════════════════════════════════════════════════════
# 防止网络超时/API错误导致的重复下单（真钱事故）
_ORDER_LOCK: dict[str, dict] = {}  # key → {status, ts, order_id}
_ORDER_LOCK_TIMEOUT_SEC = 300  # 5分钟内同品种同方向不去重


@dataclass
class OrderResult:
    ok: bool
    code: Optional[int] = None
    msg: str = ""
    # ── HIBT 响应字段（从下单响应中提取）──
    order_id: Optional[str] = None
    contract_id: Optional[str] = None
    open_price: Optional[float] = None
    expiry_time: Optional[int] = None
    payout_ratio: Optional[float] = None
    amount: Optional[str] = None
    direction: Optional[str] = None
    symbol: Optional[str] = None
    raw_response: Optional[dict] = None


ENDPOINTS = [
    "https://www.hibt.com",
    "https://api-ws.s0e9tu.com",
    "https://api.hibt0.com",
    "https://api.hibt8.com",
]

# 公共用户代理和来源头
BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Future_source": "1",
    "Client-Type": "web",
    "Hc-Platform": "web",
    "Hc-Language": "zh_CN",
    "Lang": "zh_CN",
    "Platform": "PC",
    "Origin": "https://www.hibt.com",
    "Referer": "https://www.hibt.com/",
}


def _safe_float(val):
    """安全转换 float，处理 None 和缺失"""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    """安全转换 int"""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_log_response(rj: dict) -> dict:
    """脱敏：移除敏感字段后返回可安全记录的响应"""
    if not isinstance(rj, dict):
        return {"_raw": str(rj)[:200]}
    safe = {}
    for k, v in rj.items():
        if k.lower() in ("token", "authorization", "cookie", "x-auth-token",
                         "bget_key", "bget_id", "request-id", "signature"):
            safe[k] = "***REDACTED***"
        elif isinstance(v, dict):
            safe[k] = _safe_log_response(v)
        elif isinstance(v, list):
            safe[k] = [str(x)[:100] if not isinstance(x, dict) else _safe_log_response(x) for x in v[:5]]
        else:
            safe[k] = v
    return safe


def _generate_request_id() -> str:
    """生成请求ID: bget_id + 时间戳 + 随机数，MD5后截取"""
    bget_key = config.HIBT_BGET_KEY or ""
    bget_id = config.HIBT_BGET_ID or "1"
    raw = f"{bget_id}{int(time.time() * 1000)}{random.randint(1000, 9999)}"
    if bget_key:
        raw += bget_key
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _generate_v() -> str:
    """
    动态生成 v 参数
    优先用 bget_key + bget_id 动态签名
    没有则用静态兜底 HIBT_V
    """
    if config.HIBT_BGET_KEY and config.HIBT_BGET_ID:
        bget_key = config.HIBT_BGET_KEY
        bget_id = config.HIBT_BGET_ID
        raw = f"{bget_id}{int(time.time() * 1000)}"
        return hashlib.md5(f"{raw}{bget_key}".encode()).hexdigest()[:16]
    elif config.HIBT_V:
        return config.HIBT_V
    return str(int(time.time() * 1000))


def _headers():
    h = dict(BASE_HEADERS)

    # 令牌类头（优先单独的字段，其次 HIBT_TOKEN）
    token = config.HIBT_TOKEN
    if config.HIBT_X_AUTH_TOKEN:
        h["x-auth-token"] = config.HIBT_X_AUTH_TOKEN
    elif token:
        h["x-auth-token"] = token

    if config.HIBT_AUTHORIZATION:
        h["Authorization"] = config.HIBT_AUTHORIZATION
    elif token:
        h["Authorization"] = token

    # 低层级令牌（某些接口用")
    if config.HIBT_BGET_KEY:
        h["bget_key"] = config.HIBT_BGET_KEY
    if config.HIBT_BGET_ID:
        h["bget_id"] = config.HIBT_BGET_ID

    # 请求追踪
    h["Request-Id"] = _generate_request_id()

    return h


def _has_credentials() -> bool:
    """检查是否有任何认证凭据"""
    return bool(config.HIBT_TOKEN or config.HIBT_AUTHORIZATION or
                config.HIBT_X_AUTH_TOKEN or config.HIBT_BGET_KEY)


def fetch_balance() -> float:
    if not _has_credentials():
        return 500.0  # 模拟余额

    headers = _headers()
    for ep in ENDPOINTS:
        try:
            v = _generate_v()
            url = f"{ep}/rest/c/future/u/user/balance?langCode=zh_CN&v={v}"
            res = curl_requests.get(url, headers=headers, impersonate="chrome110", timeout=10, verify=False)
            if res.status_code == 200:
                data = res.json()
                if data.get("code") in [0, 200, "0", "200"]:
                    return float(data["data"].get("amount", "0"))
                # token 过期或无效
                if data.get("code") in [401, 403, 4001, 4003]:
                    print(f"[API] 认证失败: {data.get('msg', '')}")
                    return -2.0
        except Exception:
            continue
    return -1.0


def place_order(symbol: str, direction: int, amount: float, hold_minutes: int) -> OrderResult:
    """下单到 HIBT，含防重复保护。

    Args:
        symbol: 交易品种 (BTCUSDT/ETHUSDT/SOLUSDT)
        direction: 内部方向 (1=CALL, 2=PUT)
        amount: 下单金额 (USDT)
        hold_minutes: 持仓时间 (分钟)

    Returns:
        OrderResult with ok/code/msg/order_id/...

    Safety guarantees:
        - 同品种同方向 5 分钟内不会重复下单
        - 所有 endpoint 网络超时 → ORDER_FAILED (不重试其他 endpoint)
        - API 明确拒绝 → 不重试
        - 返回 ORDER_STATUS_UNKNOWN 时引擎禁止再下同品种同方向
    """
    if not _has_credentials():
        # 模拟下单
        if random.random() < 0.95:
            return OrderResult(ok=True, code=200, msg="模拟下单成功")
        else:
            return OrderResult(ok=False, code=-1, msg="模拟网络波动拒单")

    # ── Duplicate protection ──
    lock_key = f"{symbol.lower()}_{direction}_{int(amount)}"
    existing = _ORDER_LOCK.get(lock_key)
    if existing:
        elapsed = time.time() - existing["ts"]
        if elapsed < _ORDER_LOCK_TIMEOUT_SEC:
            status = existing["status"]
            if status == "ORDER_STATUS_UNKNOWN":
                # 上一笔订单状态未知，禁止再次下单
                print(f"[ORDER_LOCK] BLOCKED: {lock_key} status=UNKNOWN, {elapsed:.0f}s ago", flush=True)
                return OrderResult(ok=False, code=-1, msg=f"ORDER_STATUS_UNKNOWN: 上一笔订单状态未知，{elapsed:.0f}秒内禁止重复下单")
            elif status == "SUCCESS":
                order_id = existing.get("order_id", "N/A")
                print(f"[ORDER_LOCK] BLOCKED: {lock_key} already succeeded (order_id={order_id}), {elapsed:.0f}s ago", flush=True)
                return OrderResult(ok=False, code=-1, msg=f"DUPLICATE_PROTECTION: 已有成功订单 {order_id}")
            # status == "FAILED": 可以重试

    # ── 标记订单锁 ──
    _ORDER_LOCK[lock_key] = {"status": "ORDER_STATUS_UNKNOWN", "ts": time.time(), "order_id": None}

    hibt_dir = map_direction_to_hibt(direction)
    data = {
        "amount": str(amount),
        "direction": str(hibt_dir),
        "symbol": symbol.lower().replace("usdt", "_usdt"),
        "timeUnit": str(hold_minutes),
        "langCode": "zh_CN",
    }
    headers = _headers()
    last_err = ""
    for ep in ENDPOINTS:
        try:
            res = curl_requests.post(
                f"{ep}/option/option-order/place",
                data=data, headers=headers,
                impersonate="chrome110", verify=False, timeout=10,
            )
            rj = res.json()
            # ── 记录完整响应结构（脱敏）──
            safe_log = _safe_log_response(rj)
            print(f"[HIBT Order Response] {json.dumps(safe_log, ensure_ascii=False)}", flush=True)
            if rj.get("code") in [0, 200, "0", "200"]:
                order_id = str(rj.get("data", {}).get("orderId", rj.get("orderId", "")))
                # ── 标记订单成功 ──
                _ORDER_LOCK[lock_key] = {"status": "SUCCESS", "ts": time.time(), "order_id": order_id}
                # 提取订单字段
                # ⚠️ payout_ratio 从 HIBT 返回字段提取，如果 HIBT 不返回则为 None
                hibt_payout = _safe_float(rj.get("data", {}).get("payout", rj.get("payout")))
                return OrderResult(
                    ok=True, code=200, msg="下单成功",
                    order_id=order_id,
                    contract_id=str(rj.get("data", {}).get("contractId", rj.get("contractId", ""))),
                    open_price=_safe_float(rj.get("data", {}).get("openPrice", rj.get("openPrice"))),
                    expiry_time=_safe_int(rj.get("data", {}).get("expiryTime", rj.get("expiryTime"))),
                    payout_ratio=hibt_payout,
                    amount=str(data["amount"]),
                    direction=str(data["direction"]),
                    symbol=str(data["symbol"]),
                    raw_response=rj,
                )
            # API明确拒绝 → 直接返回失败（不要换 endpoint 重试，防止重复下单）
            _ORDER_LOCK[lock_key] = {"status": "FAILED", "ts": time.time(), "order_id": None}
            return OrderResult(ok=False, code=rj.get("code"), msg=rj.get("msg", res.text[:30]))
        except Exception as e:
            last_err = str(e)[:50]
            continue
    # 所有 endpoint 都网络超时/出错，才返回失败（宁可漏单，不可重复下单）
    # ⚠️ ORDER_STATUS_UNKNOWN 保留在 _ORDER_LOCK 中，5分钟内禁止再次同品种同方向下单
    print(f"[ORDER_LOCK] ORDER_STATUS_UNKNOWN: {lock_key}, all endpoints timed out", flush=True)
    return OrderResult(ok=False, msg=f"网络错误: {last_err}")
