# -*- coding: utf-8 -*-
"""
SettlementReconciler — 逐单对账，禁止余额推断。

HIBT API 结算状态:
- /option/option-order/historyOrderList — 历史订单列表（含结算状态）
- 返回字段: orderId, symbol, direction, amount, openPrice, closePrice, payout, result, pnl

如果 HIBT 不返回逐单结算数据 → 禁止 LIVE 模式。
"""
import time
import json
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from curl_cffi import requests as curl_requests

from ..exchange import ENDPOINTS, _headers, _safe_float


@dataclass
class SettlementEvent:
    """单个结算事件（来自 HIBT API 响应）"""
    hibt_order_id: str = ""
    symbol: str = ""
    direction: str = ""             # "CALL" / "PUT"
    amount: float = 0.0
    open_price: float = 0.0
    close_price: float = 0.0
    payout: float = 0.0
    result: str = ""                # "WIN" / "LOSS" / "TIE"
    pnl: float = 0.0
    expiry_time_ms: int = 0
    settle_time_ms: int = 0
    raw_response: dict = field(default_factory=dict)


class SettlementReconciler:
    """
    结算对账器。

    策略:
    1. 优先从 HIBT 已关闭订单列表获取结算数据
    2. 如果 HIBT 不返回逐单数据 → settlement_available = False → 禁止 LIVE
    3. 绝不使用余额变化推断单笔结果
    """

    def __init__(self):
        self._settlement_available: Optional[bool] = None  # None = 未探测
        self._last_check_time: float = 0
        self._check_interval = 300  # 5 分钟检查一次

    def can_settle_via_hibt(self) -> bool:
        """检查 HIBT 是否返回逐单结算数据。"""
        if self._settlement_available is not None:
            return self._settlement_available

        # 探测：尝试获取最近订单列表
        try:
            orders = self._fetch_closed_orders(limit=3)
            if orders and len(orders) > 0:
                # 检查响应中是否包含 pnl/closePrice 等结算字段
                first = orders[0]
                has_settlement = (
                    first.get("pnl") is not None
                    or first.get("closePrice") is not None
                    or first.get("result") is not None
                )
                self._settlement_available = has_settlement
                return has_settlement
        except Exception:
            pass

        self._settlement_available = False
        return False

    def reconcile(self, hibt_order_id: str) -> Optional[SettlementEvent]:
        """
        根据 HIBT order_id 查询结算结果。

        Returns:
            SettlementEvent if found, None if not settled or not found.
        """
        try:
            orders = self._fetch_closed_orders(limit=50)
            for order in orders:
                oid = str(order.get("orderId", ""))
                if oid == hibt_order_id:
                    return self._parse_settlement(order)
        except Exception:
            pass
        return None

    def reconcile_all_pending(self, pending_hibt_ids: List[str]) -> Dict[str, SettlementEvent]:
        """
        批量对账所有 pending 订单。

        Returns:
            {hibt_order_id: SettlementEvent} 只返回已结算的
        """
        results = {}
        if not pending_hibt_ids:
            return results

        try:
            orders = self._fetch_closed_orders(limit=100)
            id_set = set(pending_hibt_ids)
            for order in orders:
                oid = str(order.get("orderId", ""))
                if oid in id_set:
                    event = self._parse_settlement(order)
                    if event and event.result in ("WIN", "LOSS", "TIE"):
                        results[oid] = event
        except Exception:
            pass

        return results

    def _fetch_closed_orders(self, limit: int = 50) -> List[dict]:
        """从 HIBT 获取已关闭/历史订单列表。"""
        from .. import config
        from ..exchange import _generate_v

        headers = _headers()
        v = _generate_v()

        for ep in ENDPOINTS:
            try:
                url = f"{ep}/option/option-order/historyOrderList"
                params = {
                    "pageNum": 1,
                    "pageSize": limit,
                    "langCode": "zh_CN",
                    "v": v,
                }
                res = curl_requests.get(
                    url, params=params, headers=headers,
                    impersonate="chrome110", timeout=10, verify=False,
                )
                if res.status_code != 200:
                    continue
                data = res.json()
                if data.get("code") in [0, 200, "0", "200"]:
                    orders = data.get("data", {}).get("list", data.get("data", []))
                    if isinstance(orders, list):
                        return orders
                # API 不存在或返回错误
                if data.get("code") in [404, 500, -1]:
                    continue
            except Exception:
                continue

        return []

    def _parse_settlement(self, order: dict) -> Optional[SettlementEvent]:
        """解析 HIBT 订单响应的结算字段。"""
        try:
            result_map = {0: "TIE", 1: "WIN", -1: "LOSS", "0": "TIE", "1": "WIN", "-1": "LOSS"}
            raw_result = order.get("result", order.get("settleResult", 0))
            result = result_map.get(raw_result, "TIE")

            return SettlementEvent(
                hibt_order_id=str(order.get("orderId", "")),
                symbol=str(order.get("symbol", "")).upper(),
                direction="CALL" if str(order.get("direction", "1")) in ("1", "1.0") else "PUT",
                amount=_safe_float(order.get("amount", 0)) or 0.0,
                open_price=_safe_float(order.get("openPrice", 0)) or 0.0,
                close_price=_safe_float(order.get("closePrice", 0)) or 0.0,
                payout=_safe_float(order.get("payout", 0)) or 0.0,
                result=result,
                pnl=_safe_float(order.get("pnl", order.get("realizedPnl", 0))) or 0.0,
                expiry_time_ms=int(order.get("expiryTime", 0)),
                settle_time_ms=int(order.get("settleTime", int(time.time() * 1000))),
                raw_response=order,
            )
        except Exception:
            return None

    def get_available_payout(self, symbol: str, expiry_minutes: int = 15) -> Tuple[float, float]:
        """
        尝试从 HIBT 获取实时赔付率。

        Returns:
            (payout_call_net, payout_put_net) — 净赔付率
            如果获取失败返回 (None, None)
        """
        from ..exchange import _headers, _generate_v

        # 尝试从 HIBT 合约列表获取
        try:
            headers = _headers()
            v = _generate_v()
            for ep in ENDPOINTS:
                try:
                    url = f"{ep}/option/option-order/contractList"
                    params = {
                        "symbol": symbol.lower().replace("usdt", "_usdt"),
                        "langCode": "zh_CN",
                        "v": v,
                    }
                    res = curl_requests.get(
                        url, params=params, headers=headers,
                        impersonate="chrome110", timeout=10, verify=False,
                    )
                    if res.status_code != 200:
                        continue
                    data = res.json()
                    if data.get("code") in [0, 200, "0", "200"]:
                        contracts = data.get("data", [])
                        if isinstance(contracts, list):
                            for c in contracts:
                                if c.get("timeUnit") == expiry_minutes:
                                    call = _safe_float(c.get("payoutCall", c.get("callPayout", 0.80))) or 0.80
                                    put = _safe_float(c.get("payoutPut", c.get("putPayout", 0.80))) or 0.80
                                    return (call, put)
                except Exception:
                    continue
        except Exception:
            pass

        return (None, None)


# ── Global singleton ──
_reconciler: Optional[SettlementReconciler] = None


def get_settlement_reconciler() -> SettlementReconciler:
    global _reconciler
    if _reconciler is None:
        _reconciler = SettlementReconciler()
    return _reconciler
