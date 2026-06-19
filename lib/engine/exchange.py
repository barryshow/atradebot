# -*- coding: utf-8 -*-
import time
import random
import hashlib
import json
from dataclasses import dataclass
from typing import Optional
from curl_cffi import requests as curl_requests
from . import config


@dataclass
class OrderResult:
    ok: bool
    code: Optional[int] = None
    msg: str = ""


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
    if not _has_credentials():
        # 模拟下单
        if random.random() < 0.95:
            return OrderResult(ok=True, code=200, msg="模拟下单成功")
        else:
            return OrderResult(ok=False, code=-1, msg="模拟网络波动拒单")

    # HIBT API direction（实测）:
    #   direction=1 (CALL/做多) → API传1 (HIBT 多)
    #   direction=2 (PUT/做空) → API传-1 (HIBT 空)
    hibt_dir = 1 if direction == 1 else -1
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
            if rj.get("code") in [0, 200, "0", "200"]:
                return OrderResult(ok=True, code=200, msg="下单成功")
            # API明确拒绝 → 直接返回失败（不要换 endpoint 重试，防止重复下单）
            return OrderResult(ok=False, code=rj.get("code"), msg=rj.get("msg", res.text[:30]))
        except Exception as e:
            last_err = str(e)[:50]
            continue
    # 所有 endpoint 都网络超时/出错，才返回失败（宁可漏单，不可重复下单）
    return OrderResult(ok=False, msg=f"网络错误: {last_err}")
