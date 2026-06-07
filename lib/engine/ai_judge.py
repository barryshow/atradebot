# -*- coding: utf-8 -*-
import json
import re
from curl_cffi import requests as curl_requests
from . import config


def build_prompt(
    symbol: str, direction: str, ml_prob: float, indicators: dict,
    flipped: bool, confluence: float = 0.0,
    price_trend: str = "", vol_trend: str = "", streak_info: str = "",
) -> str:
    flip_note = "\n⚠️ 注意：底层信号已被极值反转系统翻转！" if flipped else ""
    return f"""你是一个理智且果断的加密短线交易决策官。

【审查对象】 品种: {symbol} | 申请方向: {direction} | ML胜率: {ml_prob:.3f}{flip_note}
【多指标共振分】 {confluence:.2f} (满分1.0, 越高越好)

【盘面指标快照】
ADX(趋势强度)={indicators['ADX']:.2f} | MACD={indicators['MACD']:.5f} | RSI={indicators['RSI']:.2f}
BB_Pos(布林带位置)={indicators['BB_Pos']:.3f} | 资金流压(BSP_5)={indicators['BSP_5']:+.3f}

【近期走势】 {price_trend}
【成交量】 {vol_trend}
【系统战绩】 {streak_info}

【审批铁律】(必须严格执行以下逻辑)
1. 震荡死区(红灯)：若 BB_Pos 在 0.4 到 0.6 之间，说明毫无方向，坚决否决 (给0分)。
2. 做多超卖(绿灯)：若申请【做多(CALL)】，且 BB_Pos < 0.32 (处于下轨超卖区位置极佳)，不要嫌位置低，必须果断批准！(给0.8分以上)
3. 做空超买(绿灯)：若申请【做空(PUT)】，且 BB_Pos > 0.68 (处于上轨超买区位置极佳)，不要嫌位置高，必须果断批准！(给0.8分以上)
4. 顺势爆发(绿灯)：若 ADX > 30 且方向与资金流一致，属于顺势行情，立刻批准！
5. 共振加分：共振分≥0.7说明多指标高度一致，应提高信心；共振分<0.3应谨慎。
6. 连亏警示：若系统近期连亏较多，说明市场环境恶劣，应更加谨慎审批。

请根据铁律，输出严格JSON: {{"Approval_Probability": 0.0-1.0, "Reason": "15字内理由"}}"""


def call_ai(prompt: str) -> tuple[float, str]:
    try:
        res = curl_requests.post(
            config.AI_URL,
            json={
                "model": config.AI_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            headers={"Authorization": f"Bearer {config.AI_API_KEY}"},
            impersonate="chrome110",
            timeout=15,
            verify=False,
        )
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content.replace("\n", ""), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return float(data.get("Approval_Probability", 0.0)), data.get("Reason", "无理由")[:15]
    except Exception:
        pass
    return 0.0, "风控解析超时"


def check_anti_fool(direction: int, reason: str) -> bool:
    """Reject if AI reason contradicts the requested direction."""
    if direction == 2 and ("多" in reason or "涨" in reason):
        return False
    if direction == 1 and ("空" in reason or "跌" in reason):
        return False
    return True


def judge(
    symbol: str, direction: int, ml_prob: float, indicators: dict,
    flipped: bool, confluence: float = 0.0,
    price_trend: str = "", vol_trend: str = "", streak_info: str = "",
) -> tuple[float, str, bool]:
    """
    Returns (approval, reason, anti_fool_triggered).
    Dynamic threshold is applied by the caller based on confluence score.
    """
    dir_str = "做多(CALL)" if direction == 1 else "做空(PUT)"
    prompt = build_prompt(
        symbol, dir_str, ml_prob, indicators, flipped,
        confluence=confluence, price_trend=price_trend,
        vol_trend=vol_trend, streak_info=streak_info,
    )
    approval, reason = call_ai(prompt)

    if approval >= 0.50 and not check_anti_fool(direction, reason):
        return 0.0, "防呆:方向矛盾", True

    return approval, reason, False
