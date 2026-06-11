#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回测 v5：先诊断模型输出分布，再智能回测
用法: python3 scripts/backtest.py
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from curl_cffi import requests
from lib.engine import predictor, config

PAYOUT = 0.80


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


def calc_all_features(df):
    """在_existing engine feature计算上封装"""
    from lib.engine.engine import _calc_30m_features
    # 逐行算特征（给模型预测用）
    rows = []
    for i in range(len(df)):
        slice_df = df.iloc[:i+1]
        if len(slice_df) < 200:
            rows.append(None)
            continue
        row = _calc_30m_features(slice_df)
        rows.append(row)
    return rows


def simulate_hold(feat_rows, price_series, hold_min, threshold):
    """快速模拟：给定预测信号，判断持仓hold_min后的盈亏"""
    trades = []
    last_bar_ts = None
    warmup = True

    for i in range(len(feat_rows)):
        if feat_rows[i] is None:
            continue
        if warmup:
            warmup = False
            last_bar_ts = price_series.index[i]
            continue

        bar_ts = price_series.index[i]
        if last_bar_ts == bar_ts:
            continue

        row = feat_rows[i]
        # 调模型
        pred = predictor.predict("BTCUSDT", row)
        if pred is None or pred.prob_win < threshold:
            last_bar_ts = bar_ts
            continue

        # ADX过滤
        adx = float(row.get("ADX", 20))
        rsi = float(row.get("RSI", 50))
        if adx < 18 and 35 < rsi < 65:
            last_bar_ts = bar_ts
            continue

        entry_price = price_series.iloc[i]

        exit_time = bar_ts + pd.Timedelta(minutes=hold_min)
        future = price_series[price_series.index >= exit_time]
        if future.empty:
            last_bar_ts = bar_ts
            continue
        exit_price = future.iloc[0]

        is_win = (exit_price > entry_price) if pred.direction == 1 else (exit_price < entry_price)
        trades.append({
            "ts": bar_ts.strftime("%m-%d %H:%M"),
            "dir": "CALL" if pred.direction == 1 else "PUT",
            "entry": entry_price, "exit": exit_price,
            "prob": pred.prob_win,
            "result": "WIN" if is_win else "LOSS",
        })
        last_bar_ts = bar_ts
    return trades


def main():
    print("=" * 65)
    print("  离线回测 v5 — 先诊断，后回测")
    print("=" * 65)

    n = predictor.load_models()
    print(f"  模型: {n} 个\n")

    # 拉数据 (仅BTC)
    print("  拉取 BTC 数据...", end=" ", flush=True)
    df = fetch_klines("BTC_USDT", "15m", 60)
    if df is None or len(df) < 300:
        print("失败")
        return
    print(f"{len(df)} 根 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 预计算特征
    print("  计算特征...", end=" ", flush=True)
    feat_rows = calc_all_features(df)
    print(f"完成 (非空: {sum(1 for r in feat_rows if r is not None)} 行)")

    # === 诊断1: 模型概率分布 ===
    print(f"\n  {'='*65}")
    print("  [诊断] 模型概率分布 (所有K线)")
    print(f"  {'='*65}")
    probs = {"call": [], "put": [], "max": []}
    for row in feat_rows:
        if row is None: continue
        pred = predictor.predict("BTCUSDT", row)
        if pred is None: continue
        probs["max"].append(pred.prob_win)
        if pred.direction == 1:
            probs["call"].append(pred.prob_long)
        else:
            probs["put"].append(1-pred.prob_long)

    if probs["max"]:
        arr = np.array(probs["max"])
        print(f"  总预测次数: {len(arr)}")
        print(f"  概率范围: {arr.min():.4f} ~ {arr.max():.4f}")
        print(f"  平均概率: {arr.mean():.4f}")
        print(f"  中位数: {np.median(arr):.4f}")
        for pct in [50, 55, 60, 62, 65, 70]:
            cnt = int((arr >= pct/100).sum())
            print(f"  prob >= {pct}%: {cnt}次 ({cnt/len(arr)*100:.1f}%)")
        print(f"  做多信号: {len(probs['call'])} | 做空信号: {len(probs['put'])}")
    else:
        print("  没有有效的预测")

    # === 诊断2: 不同阈值下的交易数量 ===
    print(f"\n  {'='*65}")
    print("  [诊断] 不同阈值 vs 交易数量")
    print(f"  {'='*65}")
    for thresh in [0.50, 0.55, 0.58, 0.60, 0.62, 0.65]:
        count = 0
        for row in feat_rows:
            if row is None: continue
            pred = predictor.predict("BTCUSDT", row)
            if pred is not None and pred.prob_win >= thresh:
                count += 1
        print(f"  阈值 {thresh:.2f}: {count} 次触发")

    # === 回测: 循环所有阈值组合 ===
    print(f"\n  {'='*65}")
    print("  回测 — 多阈值 × 多持仓")
    print(f"  {'='*65}")

    thresholds = [0.50, 0.55, 0.58, 0.60, 0.62]
    hold_times = [5, 15, 30]

    print(f"\n  BTCUSDT 回测结果:")
    header = f"  {'持仓':>5} {'阈值':>6} {'交易':>6} {'胜':>5} {'负':>5} {'胜率':>7} {'利润/单':>9}"
    print(header)
    print(f"  {'-'*len(header)}")

    for hold in hold_times:
        for thresh in thresholds:
            trades = simulate_hold(feat_rows, df["close"], hold, thresh)
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = len(trades) - wins
            profit = wins * PAYOUT - losses * 1.0
            wr = wins / len(trades) * 100 if trades else 0
            print(f"  {hold:>5}分钟 {thresh:>.2f} {len(trades):>6} {wins:>5} {losses:>5} {wr:>6.1f}% {profit:>+8.2f}U")
        print()

    # 最优组合明细
    print(f"\n  {'='*65}")
    print("  最优组合交易明细")
    print(f"  {'='*65}")
    best_profit = -999
    best_params = None
    best_trades = []
    for hold in hold_times:
        for thresh in thresholds:
            trades = simulate_hold(feat_rows, df["close"], hold, thresh)
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = len(trades) - wins
            profit = wins * PAYOUT - losses * 1.0
            if profit > best_profit:
                best_profit = profit
                best_params = (hold, thresh)
                best_trades = trades

    if best_params:
        hold, thresh = best_params
        print(f"\n  最优: 持仓{hold}分钟 | 阈值{thresh:.2f} | 利润{best_profit:+.2f}U/单")
        print(f"  共 {len(best_trades)} 笔交易")
        print(f"\n  最近{min(15,len(best_trades))}笔:")
        print(f"  {'时间':>12} {'方向':>5} {'入场':>10} {'出场':>10} {'概率':>6} {'结果':>5}")
        print(f"  {'-'*48}")
        for t in best_trades[-15:]:
            r = "WIN " if t["result"] == "WIN" else "LOSS"
            print(f"  {t['ts']:>12} {t['dir']:>5} {t['entry']:>10.1f} {t['exit']:>10.1f} {t['prob']:.3f} {r:>5}")


if __name__ == "__main__":
    main()
