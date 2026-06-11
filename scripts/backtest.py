#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线回测 v2：比较 5/15/30 分钟持仓胜率
从 HIBT API 拉取实际 K 线数据回测，消除 CSV tick 数据重采样偏差

用法: python scripts/backtest.py
"""
import sys, os, warnings, time, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from curl_cffi import requests

from lib.engine import predictor, config
from lib.engine.engine import _calc_30m_features

# --- 配置 ---
SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API = "https://api.gateio.ws/api/v4/spot/candlesticks"
MIN_PROB = 0.62
PAYOUT = 0.80
DAYS = 60  # 拉60天数据


def fetch_klines(pair, interval="15m", days=60):
    """从 Gate.io 拉取K线"""
    limit = 1000
    all_rows = []
    last_ts = int(time.time())
    needed = days * 96
    retries = 0

    while len(all_rows) < needed:
        try:
            r = requests.get(API, params={
                "currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                retries += 1
                if retries > 3: break
                time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1800
            retries = 0
            if len(all_rows) >= needed: break
            if len(data) < limit: break
        except Exception:
            retries += 1
            if retries > 3: break
            time.sleep(3)

    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=["ts","qv","close","high","low","open","volume","final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open","close"])
    return df


def calc_features(df):
    """与 engine._calc_30m_features 一致"""
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
    result = d.replace([np.inf, -np.inf], np.nan).dropna()
    return result


def simulate_trades(feat_df, candle_df, hold_min):
    """
    feat_df: 15分钟K线(用于特征)
    candle_df: 1分钟K线(用于信号检测+出入场)
    hold_min: 持仓分钟数
    """
    candles = candle_df.copy()
    features = feat_df.copy()

    trades = []
    last_bar_ts = None
    warmup_done = False

    for idx in candles.index:
        # 当前这根1分钟K线
        bar_ts = idx

        # 预热: 跳过第一根
        if not warmup_done:
            last_bar_ts = bar_ts
            warmup_done = True
            continue

        # 同一根K线不重复检测
        if last_bar_ts == bar_ts:
            continue

        # 取到当前时间为止的15分钟特征数据
        feat_slice = features[features.index <= bar_ts]
        if len(feat_slice) < 200:
            last_bar_ts = bar_ts
            continue

        row = calc_features(feat_slice)
        if row.empty:
            last_bar_ts = bar_ts
            continue

        features_row = row.iloc[-1].to_dict()

        # 模型预测
        pred = predictor.predict("BTCUSDT", features_row)
        if pred is None or pred.prob_win < MIN_PROB:
            last_bar_ts = bar_ts
            continue

        # 风控: ADX震荡过滤
        adx = float(features_row.get("ADX", 20))
        rsi = float(features_row.get("RSI", 50))
        if adx < 18 and 35 < rsi < 65:
            last_bar_ts = bar_ts
            continue

        entry_price = float(candles.loc[bar_ts, "close"])

        # 找到持仓结束后的价格
        exit_time = bar_ts + pd.Timedelta(minutes=hold_min)
        future = candles[candles.index >= exit_time]
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


def print_trades(trades, max_show=15):
    if not trades:
        print("     无交易信号")
        return
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = len(trades) - wins
    wr = wins / len(trades) * 100
    profit = wins * PAYOUT - losses * 1.0

    print(f"     交易 {len(trades)} 次 | {wins}胜 {losses}负 | 胜率 {wr:.1f}% | 利润 {profit:+.2f}U/单")
    print()
    # header
    print(f"     {'时间':>12} {'方向':>5} {'入场':>10} {'出场':>10} {'概率':>6} {'结果':>5}")
    print(f"     {'-'*50}")
    for t in trades[-max_show:]:
        print(f"     {t['ts']:>12} {t['dir']:>5} {t['entry']:>10} {t['exit']:>10} {t['prob']:.3f} {'WIN' if t['result']=='WIN' else 'LOSS':>5}")
    return wr, profit


def main():
    print("=" * 65)
    print("  ATradeBot 离线回测 v2")
    print(f"  数据源: Gate.io 15m K线 (60天)")
    print(f"  阈值: {MIN_PROB} | 赔付率: {PAYOUT*100:.0f}%")
    print("=" * 65)

    n = predictor.load_models()
    if n == 0:
        print("  ERROR: 模型加载失败")
        sys.exit(1)

    all_results = {}
    for hold in [5, 15, 30]:
        all_results[hold] = {}
        for sym, pair in SYMBOLS.items():
            all_results[hold][sym] = []

    # 拉数据 + 回测
    for sym, pair in SYMBOLS.items():
        print(f"\n  [{sym}] 拉取数据...")
        df_15m = fetch_klines(pair, "15m", DAYS)
        if df_15m is None or len(df_15m) < 500:
            print(f"    数据不足，跳过")
            continue

        print(f"    15m K线: {len(df_15m)} 根 ({df_15m.index[0].date()} ~ {df_15m.index[-1].date()})")

        # 生成1分钟K线用于信号检测 (从15m插值)
        df_1m = df_15m.resample("1min").ffill().dropna()
        print(f"    1m K线: {len(df_1m)} 根")

        for hold in [5, 15, 30]:
            print(f"\n  --- 持仓 {hold} 分钟 ---")
            trades = simulate_trades(df_15m, df_1m, hold)
            wr, profit = print_trades(trades)
            all_results[hold][sym] = {"trades": len(trades), "wins": sum(1 for t in trades if t["result"]=="WIN"), "losses": len(trades)-sum(1 for t in trades if t["result"]=="WIN"), "wr": wr, "profit": profit}

    # 汇总
    print(f"\n\n  {'='*65}")
    print(f"  汇总对比")
    print(f"  {'='*65}")
    total_by_hold = {}
    for hold in [5, 15, 30]:
        total_trades = sum(all_results[hold][s]["trades"] for s in SYMBOLS)
        total_wins = sum(all_results[hold][s]["wins"] for s in SYMBOLS)
        total_losses = sum(all_results[hold][s]["losses"] for s in SYMBOLS)
        total_profit = sum(all_results[hold][s]["profit"] for s in SYMBOLS)
        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        total_by_hold[hold] = (total_trades, total_wins, total_losses, wr, total_profit)

        # 分品种明细
        print(f"\n  {hold}分钟持仓:")
        print(f"  {'品种':>10} {'交易':>6} {'胜':>4} {'负':>4} {'胜率':>7} {'利润/单':>9}")
        print(f"  {'-'*45}")
        for s in SYMBOLS:
            r = all_results[hold][s]
            if r["trades"] > 0:
                print(f"  {s:>10} {r['trades']:>6} {r['wins']:>4} {r['losses']:>4} {r['wr']:>6.1f}% {r['profit']:>+8.2f}U")
        print(f"  {'合计':>10} {total_trades:>6} {total_wins:>4} {total_losses:>4} {wr:>6.1f}% {total_profit:>+8.2f}U")

    # 结论
    print(f"\n\n  {'='*65}")
    print(f"  结论")
    print(f"  {'='*65}")
    best_hold = max(total_by_hold, key=lambda h: total_by_hold[h][4])
    best_data = total_by_hold[best_hold]
    print(f"  最优持仓: {best_hold}分钟")
    print(f"  总交易: {best_data[0]} 次 | 胜率: {best_data[3]:.1f}% | 利润: {best_data[4]:+.2f}U/单")
    print()
    print(f"  注意: 回测假设按收盘价成交(无滑点)")
    print(f"  赔付率固定80%, 数据时间范围短, 仅为参考")


if __name__ == "__main__":
    main()
