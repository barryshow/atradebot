#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回测 v4：预计算特征，超高速
直接在15分钟K线上逐根检测
用法: python3 scripts/backtest.py
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from curl_cffi import requests

from lib.engine import predictor, config

MIN_PROB = 0.62
PAYOUT = 0.80
FEATURES_CACHE = {}  # 预计算的特征缓存


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
    """一次性计算所有行的全部特征，返回特征矩阵(行索引同df)"""
    d = df.copy()
    eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)
    d["ret_1"] = d["close"].pct_change(1).fillna(0)
    d["ret_3"] = d["close"].pct_change(3).fillna(0)
    d["ret_6"] = d["close"].pct_change(6).fillna(0)
    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    d["MACD"] = (2 * (macd - macd.ewm(span=9, adjust=False).mean())).fillna(0)
    d["MACD_hist"] = d["MACD"].fillna(0)
    delta = d["close"].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = (100 - (100 / (1 + gain / loss))).fillna(50)
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std().fillna(0)
    d["BB_Pos"] = ((d["close"] - (mid - 2 * std)) / (4 * std + eps)).clip(0, 1)
    d["BB_width"] = (((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)).fillna(0)
    tr = pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift(1)).abs(),(d["low"]-d["close"].shift(1)).abs()], axis=1).max(axis=1)
    d["ATR_pct"] = (tr.rolling(14).mean() / (d["close"] + eps)).fillna(0)
    up = d["high"] - d["high"].shift(1)
    dn = d["low"].shift(1) - d["low"]
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=d.index)
    tr14 = tr.rolling(14).sum().replace(0, eps)
    pdi = 100 * pdm.rolling(14).sum() / tr14
    ndi = 100 * ndm.rolling(14).sum() / tr14
    d["ADX"] = (100 * abs(pdi - ndi) / (pdi + ndi + eps)).rolling(14).mean().fillna(20)
    d["MA10"] = d["close"].rolling(10).mean().bfill()
    d["MA20"] = d["close"].rolling(20).mean().bfill()
    d["MA50"] = d["close"].rolling(50).mean().bfill()
    d["price_vs_MA20"] = ((d["close"] - d["MA20"]) / (d["MA20"] + eps)).fillna(0)
    d["price_vs_MA50"] = ((d["close"] - d["MA50"]) / (d["MA50"] + eps)).fillna(0)
    d["MA_trend"] = np.sign(d["MA10"] - d["MA20"]).fillna(0)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vwap = (d["volume"] * tp).cumsum() / (d["volume"].cumsum() + eps)
    d["VWAP_dist"] = ((d["close"] - vwap) / (vwap + eps)).fillna(0)
    d["vol_ratio"] = (d["volume"] / (d["volume"].rolling(5).mean() + eps)).fillna(1)
    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"] * obv_dir).cumsum()
    d["OBV_trend"] = np.sign(obv - obv.shift(5)).fillna(0)
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["CCI"] = ((tp - tp_sma) / (0.015 * tp_mad + eps)).fillna(0)
    atr14 = tr.rolling(14).sum()
    d["CHOP"] = (100 * np.log10(atr14 / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + eps)) / np.log10(14)).fillna(50)
    body = (d["close"] - d["open"]).abs()
    d["body_pct"] = (body / (d["high"] - d["low"] + eps)).fillna(0.5)
    d["is_green"] = (d["close"] > d["open"]).astype(int)
    return d.replace([np.inf, -np.inf], np.nan)


def backtest_symbol(feat_df, hold_min):
    """在预计算的特征上快速回测"""
    if len(feat_df) < 250:
        return []

    trades = []
    last_bar_ts = None
    warmup = True

    for i in range(200, len(feat_df)):
        bar = feat_df.iloc[i]
        bar_ts = bar.name

        if warmup:
            warmup = False
            last_bar_ts = bar_ts
            continue
        if last_bar_ts == bar_ts:
            continue

        row = feat_df.iloc[i]  # 特征已经算好了
        # NaN检查
        adx_val = row.get("ADX")
        if pd.isna(adx_val):
            last_bar_ts = bar_ts
            continue

        # 预测
        pred = predictor.predict("BTCUSDT", row)
        if pred is None or pred.prob_win < MIN_PROB:
            last_bar_ts = bar_ts
            continue

        # ADX震荡过滤
        rsi_val = row.get("RSI", 50)
        if adx_val < 18 and 35 < rsi_val < 65:
            last_bar_ts = bar_ts
            continue

        entry_price = float(bar["close"])

        # 找持仓结束
        exit_time = bar_ts + pd.Timedelta(minutes=hold_min)
        future = feat_df[feat_df.index >= exit_time]
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
            "adx": round(adx_val, 1),
            "rsi": round(rsi_val, 1),
        })
        last_bar_ts = bar_ts

    return trades


def main():
    print("=" * 60)
    print("  ATradeBot 离线回测 v4")
    print("  数据: Gate.io 15m K线 | 预计算特征")
    print(f"  阈值: {MIN_PROB} | 赔付率: {PAYOUT*100:.0f}%")
    print("=" * 60)

    n = predictor.load_models()
    print(f"  模型: {n}/{len(config.SYMBOLS)} 个\n")

    all_data = {}
    for sym, pair in [("BTCUSDT", "BTC_USDT"), ("ETHUSDT", "ETH_USDT"), ("SOLUSDT", "SOL_USDT")]:
        print(f"  [{sym}] 拉取数据...", end=" ", flush=True)
        df = fetch_klines(pair, "15m", 60)
        if df is None or len(df) < 300:
            print("失败或数据不足")
            continue
        print(f"{len(df)} 根", end=" ", flush=True)
        # 预计算特征
        sys.stdout.flush()
        feat_df = calc_all_features(df)
        feat_df = feat_df.dropna()
        print(f"(特征 {len(feat_df)} 行)")
        all_data[sym] = feat_df

    if not all_data:
        print("  没有数据，退出")
        return

    results = {}
    for hold in [5, 15, 30]:
        print(f"\n  --- 持仓 {hold} 分钟 ---")
        results[hold] = {}
        for sym, df in all_data.items():
            trades = backtest_symbol(df, hold)
            wins = sum(1 for t in trades if t["result"] == "WIN")
            losses = len(trades) - wins
            profit = wins * PAYOUT - losses * 1.0
            wr = wins / len(trades) * 100 if trades else 0
            results[hold][sym] = (len(trades), wins, losses, wr, profit, trades)
            if trades:
                print(f"  {sym:>10}: {len(trades):>4}次 {wins:>3}胜 {losses:>3}负 {wr:>5.1f}% 利润{profit:>+7.2f}U/单", flush=True)

    # 汇总
    print(f"\n\n  {'='*60}")
    print(f"  汇总对比")
    print(f"  {'='*60}")
    print(f"  {'持仓':>8} {'总交易':>8} {'胜':>6} {'负':>6} {'胜率':>8} {'利润/单':>10}")
    print(f"  {'-'*45}")

    best_hold, best_profit = 15, -999
    for hold in [5, 15, 30]:
        if hold not in results: continue
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

    print(f"\n  {best_hold}分钟持仓 — 最近交易:")
    for sym in all_data:
        trades = results[best_hold][sym][5]
        if trades:
            print(f"\n  {sym} (最近{min(10,len(trades))}笔):")
            print(f"  {'时间':>12} {'方向':>5} {'入场':>10} {'出场':>10} {'概率':>6} {'结果':>5}")
            print(f"  {'-'*48}")
            for t in trades[-10:]:
                r = "WIN " if t["result"] == "WIN" else "LOSS"
                print(f"  {t['ts']:>12} {t['dir']:>5} {t['entry']:>10} {t['exit']:>10} {t['prob']:.3f} {r:>5}")


if __name__ == "__main__":
    main()
