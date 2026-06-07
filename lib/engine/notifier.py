# -*- coding: utf-8 -*-
from curl_cffi import requests as curl_requests
from . import config


def send_feishu(text: str):
    if not config.FEISHU_WEBHOOK:
        return
    try:
        curl_requests.post(
            config.FEISHU_WEBHOOK,
            json={"msg_type": "text", "content": {"text": text}},
            timeout=5, verify=False,
        )
    except Exception:
        pass


def notify_trade(symbol: str, direction: str, entry: float, amount: float,
                 ml_prob: float, indicators: dict, reason: str,
                 balance: float, active: int, flipped: bool):
    flip_tag = "🔄 [信号翻转] " if flipped else ""
    msg = (
        f"{flip_tag}💥 {symbol} {direction}\n"
        f"💰 价: ${entry:.4f} | 投入: {amount}U\n"
        f"🤖 ML胜率: {ml_prob:.3f}\n"
        f"📊 ADX: {indicators['ADX']:.1f} | BOLL: {indicators['BB_Pos']:.2f}\n"
        f"🎯 AI裁决: {reason}\n"
        f"📈 持仓: {active}/{config.MAX_CONCURRENT_TRADES} | 余额: {balance:.2f}U"
    )
    send_feishu(msg)


def notify_result(symbol: str, is_win: bool, pnl: float):
    tag = "✅盈利" if is_win else "❌亏损"
    sign = "+" if is_win else ""
    send_feishu(f"🔔 结算: {symbol} [{tag}] {sign}{pnl:.2f}U")
