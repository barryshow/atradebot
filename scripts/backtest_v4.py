#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 回测 — 三品种 Walk-Forward

使用 hibt_ticks.csv 的 1m K 线数据，对 BTCUSDT/ETHUSDT/SOLUSDT
各跑 Walk-Forward 回测，验证 Edge Bucket 与 ROI 正相关。

数据范围: 2026-06-01 ~ 2026-06-08 (7天)
回测参数: Train 5天 / Test 2天 / Step 1天 → 2-3 窗口
"""
import sys, os, io, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from lib.engine.backtester import WalkForwardBacktester, BacktestReport

# ── 加载数据 ──
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hibt_ticks.csv")
df = pd.read_csv(CSV, engine="python", on_bad_lines="skip")
df = df.iloc[:, :7]
df.columns = ["ts", "symbol", "open", "high", "low", "close", "volume"]
if str(df.iloc[0, 1]).strip() == "symbol":
    df = df.iloc[1:].copy()
df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
df = df.dropna(subset=["ts", "open", "high", "low", "close"])
df["datetime"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
df = df.dropna(subset=["datetime"])

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtest_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  EventEdge V2 Walk-Forward Backtest")
print(f"  Data: {df['datetime'].min()} ~ {df['datetime'].max()} ({(df['datetime'].max()-df['datetime'].min()).days}d)")
print(f"  Symbols: {SYMBOLS}")
print("=" * 70)

all_reports = {}

for symbol in SYMBOLS:
    print(f"\n{'#'*70}")
    print(f"#  {symbol}")
    print(f"{'#'*70}")

    ds = df[df["symbol"] == symbol].copy()
    ds = ds.set_index("datetime").sort_index()
    # 保留 OHLCV
    ds = ds[["open", "high", "low", "close", "volume"]]
    ds = ds.astype(float)

    total_days = (ds.index[-1] - ds.index[0]).days
    if total_days < 3:
        print(f"  ⚠ 数据不足 ({total_days}d), 跳过")
        continue
    print(f"  Rows: {len(ds)}, Days: {total_days}d")

    # ── 回测 ──
    # 用 5d train / 2d test / 2d step (适合 7 天数据)
    train_days = min(5, max(2, total_days - 2))
    test_days = min(2, max(1, total_days - train_days))
    step_days = max(1, test_days)

    bt = WalkForwardBacktester(
        symbol=symbol,
        expiries=[15],
        train_window_days=train_days,
        test_window_days=test_days,
        step_days=step_days,
        min_history=100,
        min_samples_train=100,
        min_order_usd=3,
        order_step=1,
        net_payout_ratio=0.80,
        min_probability=0.50,
        min_effective_edge=0.0,
        kelly_fraction=0.10,
        max_bet_fraction=0.01,
        output_dir=OUTPUT_DIR,
        verbose=True,
    )

    report = bt.run(ds, equity=5000.0)
    all_reports[symbol] = report

    # ── 打印报告 ──
    bt.print_report(report)

    # ── 保存 JSON ──
    bt.save_report(report)

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print(f"  BACKTEST SUMMARY")
print(f"{'='*70}")

print(f"\n{'Symbol':<12} {'Windows':>7} {'Trades':>7} {'WR':>7} {'Breakeven':>9} {'PnL':>8} {'ROI':>7} {'AvgEdge':>8} {'Brier':>7} {'Sharpe':>7}")
print(f"{'─'*85}")

for sym, rpt in all_reports.items():
    print(f"  {sym:<12} {rpt.total_windows:>7} {rpt.total_trades:>7} "
          f"{rpt.overall_win_rate:>6.1%} {rpt.break_even_win_rate:>8.1%} "
          f"{rpt.total_pnl:>+8.2f} {rpt.overall_roi:>+6.1%} "
          f"{rpt.avg_effective_edge:>7.2%} {rpt.overall_brier_score:>7.4f} "
          f"{rpt.sharpe_ratio:>7.2f}")

# ── Edge Bucket 相关性验证 ──
print(f"\n{'='*70}")
print(f"  EDGE BUCKET vs ROI CORRELATION VERIFICATION")
print(f"{'='*70}")

for sym, rpt in all_reports.items():
    if not rpt.all_trades:
        continue

    # Recompute edge buckets from all_trades
    edge_buckets = bt._compute_edge_bucket_stats(rpt.all_trades)
    if not edge_buckets:
        print(f"\n  {sym}: No edge bucket data")
        continue

    print(f"\n  {sym} (N={rpt.total_trades}):")
    print(f"  {'Bucket':>10} {'Trades':>7} {'WR':>7} {'ROI':>7} {'AvgEdge':>8}")
    print(f"  {'─'*45}")

    prev_wr = None
    monotonic = True
    for bucket_name in ["negative", "0-1%", "1-2%", "2-3%", "3-5%", "5-7%", "7-10%", "10%+"]:
        if bucket_name in edge_buckets:
            s = edge_buckets[bucket_name]
            if s["total_trades"] > 0:
                wr = s["win_rate"]
                roi = s["roi"]
                avg_edge = s["avg_effective_edge"]
                marker = ""
                if prev_wr is not None and wr < prev_wr:
                    monotonic = False
                    marker = " ⚠"
                print(f"  {bucket_name:>10} {s['total_trades']:>7} {wr:>6.1%} {roi:>+6.1%} {avg_edge:>7.2%}{marker}")
                prev_wr = wr

    if monotonic:
        print(f"\n  ✅ Edge Bucket → ROI 正相关 (单调递增)")
    else:
        print(f"\n  ⚠ Edge Bucket → ROI 关系不单调 — 需要检查模型概率校准")

# ── 概率区间可靠性检查 ──
print(f"\n{'='*70}")
print(f"  PROBABILITY RELIABILITY CHECK")
print(f"{'='*70}")

for sym, rpt in all_reports.items():
    if not rpt.all_trades:
        continue
    prob_buckets = bt._compute_probability_bucket_stats(rpt.all_trades)
    if not prob_buckets:
        continue

    print(f"\n  {sym}:")
    print(f"  {'Bucket':>10} {'Trades':>7} {'Predicted':>9} {'Actual':>7} {'Bias':>7}")
    print(f"  {'─'*45}")
    for bucket_name in ["50-52%", "52-54%", "54-56%", "56-58%", "58-60%", "60-65%", "65-70%", "70%+"]:
        if bucket_name in prob_buckets:
            s = prob_buckets[bucket_name]
            if s["total_trades"] > 0:
                pred = s.get("avg_predicted_prob", 0)
                actual = s["win_rate"]
                bias = pred - actual
                bias_marker = " ⚠" if abs(bias) > 0.05 else ""
                print(f"  {bucket_name:>10} {s['total_trades']:>7} {pred:>8.1%} {actual:>6.1%} {bias:>+6.1%}{bias_marker}")

print(f"\n{'='*70}")
print(f"  BACKTEST COMPLETE")
print(f"  Results saved to: {OUTPUT_DIR}")
print(f"{'='*70}")