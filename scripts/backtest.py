#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回测 v3：直接在15分钟K线上逐根检测，速度快30倍
比较 5 / 15 / 30 分钟持仓胜率

用法: python3 scripts/backtest.py
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from curl_cffi import requests

from lib.engine import predictor, config
from lib.engine.engine import _calc_30m_features

MIN_PROB = 0.62
PAYOUT = 0.80
SYMBOLS_MAP = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}

def fetch_klines(pair, interval="15m", days=60):
    api = "https://api.gateio.ws/api/v4/spot/candlesticks"
    limit = 1000
    all_rows, last_ts = [], int(time.time())
    needed = days * 96
    while len(all_rows) < needed:
        try:
            r = requests.get(api, params={"currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts}, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200: break
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1800
            if len(data) < limit: break
        except: break
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=["ts","qv","close","high","low","open","volume","final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close"])


def backtest_symbol_15m(df_15m, hold_min):
    """
    直接在15分钟K线上回测。
    每根新15分钟K线 → 算特征 → 预测 → 记录。
    hold_min: 持仓分钟数 (5/15/30)
    """
    if len(df_15m) < 250:
        return []

    trades = []
    last_bar_ts = None
    warmup = True

    # 预计算全局特征（一次性算完所有行的特征）
    full_feat = _calc_30m_features(df_15m)

    # 逐根K线滑动检测
    for i in range(len(df_15m)):
        bar = df_15m.iloc[i]
        bar_ts = bar.name

        # 预热：跳过前200根（特征需要200根历史）
        if warmup:
            if i < 200:
                continue
            warmup = False
            last_bar_ts = bar_ts
            continue

        # 同一根不重复
        if last_bar_ts == bar_ts:
            continue

        # 取到当前K线为止的特征数据
        feat_slice = df_15m.iloc[:i+1]
        if len(feat_slice) < 200:
            last_bar_ts = bar_ts
            continue

        row = _calc_30m_features(feat_slice)
        if row is None:
            last_bar_ts = bar_ts
            continue

        # 预测
        pred = predictor.predict("BTCUSDT", row)
        if pred is None or pred.prob_win < MIN_PROB:
            last_bar_ts = bar_ts
            continue

        # 风控：ADX震荡过滤
        adx = float(row.get("ADX", 20))
        rsi = float(row.get("RSI", 50))
        if adx < 18 and 35 < rsi < 65:
            last_bar_ts = bar_ts
            continue

        entry_price = float(bar["close"])

        # 找持仓结束时的价格（按15分钟K线找）
        exit_time = bar_ts + pd.Timedelta(minutes=hold_min)
        future = df_15m[df_15m.index >= exit_time]
        if future.empty:
            last_bar_ts = bar_ts
            continue
        exit_price = float(future.iloc[0]["close"])

        is_win = (exit_price > entry_price) if pred.direction == 1 else (exit_price < entry_price)

        trades.append({
            "ts": bar_ts.strftime("%m-%d %H:%M"),
            "dir": "CALL" if pred.direction == 1 else "PUT",
            "entry": round(entry_price, 1),
            "exit": round(exit_price, 1),
            "prob": round(pred.prob_win, 3),
            "result": "WIN" if is_win else "LOSS",
            "adx": round(adx, 1),
            "rsi": round(rsi, 1),
        })
        last_bar_ts = bar_ts

    return trades


def main():
    print("=" * 60)
    print("  ATradeBot 离线回测 v3")
    print("  数据: Gate.io 15m K线 | 逐根检测")
    print(f"  阈值: {MIN_PROB} | 赔付率: {PAYOUT*100:.0f}%")
    print("=" * 60)

    n = predictor.load_models()
    print(f"  模型: {n}/{len(config.SYMBOLS)} 个\n")

    all_data = {}
    for sym, pair in SYMBOLS_MAP.items():
        print(f"  [{sym}] 拉取数据...", end=" ", flush=True)
        df = fetch_klines(pair, "15m", 60)
        if df is None or len(df) < 300:
            print(f"失败或数据不足")
            continue
        print(f"{len(df)} 根 ({df.index[0].date()} ~ {df.index[-1].date()})")
        all_data[sym] = df

    if not all_data:
        print("  没有可用数据，退出")
        return

    results = {}
    for hold in [5, 15, 30]:
        print(f"\n  {'='*60}")
        print(f"  持仓 {hold} 分钟")
        print(f"  {'='*60}")
        results[hold] = {}
        for sym, df in all_data.items():
            trades = backtest_symbol_15m(df, hold)
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = len(trades) - wins
            profit = wins * PAYOUT - losses * 1.0
            wr = wins / len(trades) * 100 if trades else 0
            results[hold][sym] = (len(trades), wins, losses, wr, profit, trades)

            if trades:
                print(f"  {sym:>10}: {len(trades):>4}次 {wins:>3}胜 {losses:>3}负 {wr:>5.1f}% 利润{profit:>+7.2f}U/单")

    # 汇总
    print(f"\n\n  {'='*60}")
    print(f"  汇总对比")
    print(f"  {'='*60}")
    print(f"  {'持仓':>8} {'总交易':>8} {'胜':>6} {'负':>6} {'胜率':>8} {'利润/单':>10}")
    print(f"  {'-'*45}")

    best_hold, best_profit = 15, -999
    for hold in [5, 15, 30]:
        total_t = sum(results[hold][s][0] for s in results[hold])
        total_w = sum(results[hold][s][1] for s in results[hold])
        total_l = sum(results[hold][s][2] for s in results[hold])
        total_p = sum(results[hold][s][3] for s in results[hold])
        wr = total_w / total_t * 100 if total_t > 0 else 0
        profit = total_w * PAYOUT - total_l * 1.0
        print(f"  {hold:>5}分钟 {total_t:>8} {total_w:>6} {total_l:>6} {wr:>7.1f}% {profit:>+9.2f}U")
        if profit > best_profit:
            best_hold, best_profit = hold, profit

    print(f"\n  最优: {best_hold}分钟持仓 (利润 {best_profit:+.2f}U/单)")

    # 显示最优持仓的明细
    print(f"\n  {best_hold}分钟持仓 — 最近交易:")
    for sym in all_data:
        trades = results[best_hold][sym][5]
        if trades:
            print(f"\n  {sym} (最近{min(8,len(trades))}笔):")
            print(f"  {'时间':>12} {'方向':>5} {'入场':>10} {'出场':>10} {'概率':>6} {'结果':>5}")
            print(f"  {'-'*48}")
            for t in trades[-8:]:
                r = "WIN " if t["result"] == "WIN" else "LOSS"
                print(f"  {t['ts']:>12} {t['dir']:>5} {t['entry']:>10} {t['exit']:>10} {t['prob']:.3f} {r:>5}")


if __name__ == "__main__":
    main()