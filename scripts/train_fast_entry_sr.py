#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support / Resistance Feature Experiment

1. Train Model B (FAST_FEATURES + SR_FEATURES) on same data as Model A
2. Purged Walk-Forward backtest for both models
3. Conditional performance: near support / near resistance
4. Feature importance & SHAP analysis
5. Output report

用法:
    python scripts/train_fast_entry_sr.py --days 90 --output ./models_sr
"""
import argparse, io, os, sys, time, json, warnings
warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from curl_cffi import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.engine.multi_timeframe_features import (
    compute_fast_entry_features, FAST_FEATURES, build_fast_feature_vector,
    SR_FEATURES, ALL_FEATURES_B, build_fast_feature_vector_b,
    compute_sr_features,
)

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}


def fetch_klines(pair, interval="1m", days=90):
    limit = 1000
    all_rows, last_ts = [], int(time.time())
    for _ in range(15):
        if len(all_rows) >= days * 24 * 60:
            break
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval, "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1
            if len(data) < limit: break
        except Exception:
            time.sleep(3)
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=["ts","qv","close","high","low","open","volume","final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close"])


def build_labels(df_1m, forward_minutes=15, min_move=0.0003):
    closes = df_1m["close"].values
    timestamps = np.array([int(t.timestamp() * 1000) for t in df_1m.index])
    n = len(closes)
    rows = []
    for i in range(50, n - forward_minutes):
        entry_price = float(closes[i])
        expiry_price = float(closes[i + forward_minutes])
        move_pct = (expiry_price - entry_price) / entry_price
        if abs(move_pct) < min_move:
            continue
        direction = 1 if move_pct > 0 else 0
        rows.append({
            "entry_ts": int(timestamps[i]),
            "entry_price": round(entry_price, 6),
            "expiry_price": round(expiry_price, 6),
            "move_pct": round(move_pct, 8),
            "label_binary": direction,
            "entry_idx": i,
        })
    return pd.DataFrame(rows)


def aggregate_5m(df_1m):
    return df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def build_samples(df_1m, labels, use_sr=False):
    """Build training samples — supports Model A (use_sr=False) and B (use_sr=True)"""
    df_5m = aggregate_5m(df_1m)
    X_list, y_list = [], []

    for _, row in labels.iterrows():
        entry_ts = row["entry_ts"]
        entry_dt = pd.Timestamp(entry_ts, unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)
        entry_idx = row["entry_idx"]

        # slice BEFORE entry (防 Look-Ahead)
        idx_1m = df_1m.index.tz_localize(None) if df_1m.index.tz is not None else df_1m.index
        mask_1m = idx_1m <= entry_dt
        if not mask_1m.any() or mask_1m.sum() < 50:
            continue
        df_1m_before = df_1m[mask_1m].copy()

        idx_5m = df_5m.index.tz_localize(None) if df_5m.index.tz is not None else df_5m.index
        mask_5m = idx_5m <= entry_dt
        if not mask_5m.any() or mask_5m.sum() < 10:
            continue
        df_5m_before = df_5m[mask_5m].copy()

        # Base features
        features = compute_fast_entry_features(
            symbol="", realtime=None,
            df_1m=df_1m_before, df_5m=df_5m_before,
            slow_context={"probability": 0.50, "regime": "RANGE", "trend_strength": 0, "volatility": 0},
        )

        # SR features (only if use_sr)
        if use_sr:
            sr = compute_sr_features(df_1m_before)
            features.update(sr)

        try:
            if use_sr:
                vec = build_fast_feature_vector_b(features)
            else:
                vec = build_fast_feature_vector(features)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                continue
            X_list.append(vec)
            y_list.append(row["label_binary"])
        except Exception:
            continue

    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)


def train_model(X_train, y_train, X_test, y_test):
    """Train LightGBM with same params as train_fast_entry.py"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos, class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])

    return model, scaler


def backtest_trading(model, scaler, df_1m, labels, payout, use_sr=False, min_edge=0.02, bet_size=3):
    """
    Walk-forward trading simulation.
    No look-ahead: each prediction uses only data before entry_ts.
    """
    df_5m = aggregate_5m(df_1m)
    trades = []

    for idx, row in labels.iterrows():
        entry_ts = row["entry_ts"]
        entry_dt = pd.Timestamp(entry_ts, unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)

        # Slice before entry
        m1 = (df_1m.index.tz_localize(None) if df_1m.index.tz is not None else df_1m.index) <= entry_dt
        m5 = (df_5m.index.tz_localize(None) if df_5m.index.tz is not None else df_5m.index) <= entry_dt
        if not m1.any() or m1.sum() < 50:
            continue
        df1b = df_1m[m1].copy()
        df5b = df_5m[m5].copy()
        if len(df5b) < 10:
            continue

        features = compute_fast_entry_features(
            symbol="", realtime=None, df_1m=df1b, df_5m=df5b,
            slow_context={"probability": 0.50, "regime": "RANGE", "trend_strength": 0, "volatility": 0},
        )
        if use_sr:
            sr_feats = compute_sr_features(df1b)
            features.update(sr_feats)

        try:
            if use_sr:
                vec = build_fast_feature_vector_b(features)
            else:
                vec = build_fast_feature_vector(features)
            if np.any(np.isnan(vec)):
                continue
            vec_s = scaler.transform(vec.reshape(1, -1))
            proba = model.predict_proba(vec_s)
            pos_idx = 1 if 1 in model.classes_ else 0
            p_call = float(proba[0, pos_idx])
        except:
            continue

        # Direction + Edge check
        be_prob = 1.0 / (1.0 + payout)
        p_selected = max(p_call, 1 - p_call)
        direction = "CALL" if p_call >= 0.5 else "PUT"
        edge = p_selected - be_prob

        if edge < min_edge or p_selected <= be_prob:
            continue  # No trade

        # Simulate trade outcome
        actual_move = row["move_pct"]
        is_win = (direction == "CALL" and actual_move > 0) or (direction == "PUT" and actual_move < 0)

        pnl = bet_size * payout if is_win else -bet_size

        # Collect SR features for conditional analysis
        sr_info = {}
        if use_sr:
            for f_ in ["distance_to_low_24h", "distance_to_high_24h",
                        "distance_to_low_4h", "distance_to_high_4h",
                        "price_percentile_4h", "price_percentile_24h",
                        "distance_to_recent_swing_low", "distance_to_recent_swing_high"]:
                sr_info[f_] = features.get(f_, 0.0)

        trades.append({
            "entry_ts": entry_ts,
            "direction": direction,
            "p_call": round(p_call, 4),
            "p_selected": round(p_selected, 4),
            "edge": round(edge, 4),
            "is_win": is_win,
            "pnl": pnl,
            "entry_price": row["entry_price"],
            "expiry_price": row["expiry_price"],
            "move_pct": round(actual_move, 6),
            **sr_info,
        })

    return trades


def analyze_conditional(trades, prefix=""):
    """Conditional performance by distance_to_low / distance_to_high buckets"""
    if not trades:
        print(f"\n  {prefix}NO TRADES")
        return

    print(f"\n  [{prefix}] Conditional Performance Analysis")
    print(f"  {'='*60}")

    # PUT near support
    puts = [t for t in trades if t["direction"] == "PUT"]
    if puts:
        print(f"\n  PUT trades by distance_to_low_24h:")

        buckets = [("very_near", 0, 0.001), ("near", 0.001, 0.0025),
                    ("mid", 0.0025, 0.005), ("far", 0.005, 1.0)]
        for name, lo, hi in buckets:
            subset = [t for t in puts if lo <= t.get("distance_to_low_24h", 0) < hi]
            if subset:
                wr = sum(1 for t in subset if t["is_win"]) / len(subset)
                total_pnl = sum(t["pnl"] for t in subset)
                avg_p = np.mean([t["p_call"] for t in subset])
                print(f"    {name:12s} [{lo:.3%}-{hi:.3%}]: {len(subset):4d} trades, "
                      f"WR={wr:.1%}, PnL={total_pnl:+.0f}U, avg_p_call={avg_p:.3f}")

    # CALL near resistance
    calls = [t for t in trades if t["direction"] == "CALL"]
    if calls:
        print(f"\n  CALL trades by distance_to_high_24h:")
        buckets = [("very_near", 0, 0.001), ("near", 0.001, 0.0025),
                    ("mid", 0.0025, 0.005), ("far", 0.005, 1.0)]
        for name, lo, hi in buckets:
            subset = [t for t in calls if lo <= t.get("distance_to_high_24h", 0) < hi]
            if subset:
                wr = sum(1 for t in subset if t["is_win"]) / len(subset)
                total_pnl = sum(t["pnl"] for t in subset)
                avg_p = np.mean([1 - t["p_call"] for t in subset])  # P(PUT) for CALL trades
                print(f"    {name:12s} [{lo:.3%}-{hi:.3%}]: {len(subset):4d} trades, "
                      f"WR={wr:.1%}, PnL={total_pnl:+.0f}U, avg_p_put={avg_p:.3f}")


def compute_stats(trades, label=""):
    """Compute trading statistics"""
    if not trades:
        return {"label": label, "n_trades": 0, "wins":0, "losses":0, "win_rate":0,
                "total_pnl":0, "avg_pnl_per_trade":0, "max_drawdown":0,
                "longest_losing_streak":0, "brier":0, "avg_predicted_prob":0,
                "avg_actual_win":0}
    wins = sum(1 for t in trades if t["is_win"])
    losses = sum(1 for t in trades if not t["is_win"])
    total = len(trades)
    wr = wins / total if total > 0 else 0
    total_pnl = sum(t["pnl"] for t in trades)
    pnls = [t["pnl"] for t in trades]
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    dd = cumsum - running_max
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    # Longest losing streak
    streak = 0
    max_streak = 0
    for t in trades:
        if not t["is_win"]:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Predicted vs actual
    pred_ps = np.array([t["p_selected"] for t in trades])
    actuals = np.array([1.0 if t["is_win"] else 0.0 for t in trades])
    brier = np.mean((pred_ps - actuals) ** 2) if len(pred_ps) > 0 else 0

    return {
        "label": label,
        "n_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total, 2) if total > 0 else 0,
        "max_drawdown": round(max_dd, 2),
        "longest_losing_streak": max_streak,
        "brier": round(brier, 4),
        "avg_predicted_prob": round(float(np.mean(pred_ps)), 4) if len(pred_ps) > 0 else 0,
        "avg_actual_win": round(float(np.mean(actuals)), 4) if len(actuals) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--test-split", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="./models_sr")
    parser.add_argument("--quick", action="store_true", help="Quick run: 30 days only")
    args = parser.parse_args()

    if args.quick:
        args.days = 30

    syms = [s.strip() for s in args.symbols.split(",")]
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("  Support / Resistance Feature Experiment")
    print(f"  Symbols: {syms} | Days: {args.days}")
    print(f"  Model A: {len(FAST_FEATURES)} features")
    print(f"  Model B: {len(ALL_FEATURES_B)} features (+{len(SR_FEATURES)} SR)")
    print("=" * 70)

    all_results = {}

    for sym in syms:
        pair = SYMBOLS.get(sym)
        if not pair:
            continue
        payout = PAYOUTS.get(sym, 0.80)
        be_prob = 1.0 / (1.0 + payout)

        print(f"\n{'─'*70}")
        print(f"  {sym} (payout={payout}, BE={be_prob:.1%})")
        print(f"{'─'*70}")

        print(f"  拉取数据...", end=" ", flush=True)
        df_1m = fetch_klines(pair, interval="1m", days=args.days)
        if df_1m is None or len(df_1m) < 500:
            print(f"数据不足"); continue
        print(f"{len(df_1m)} rows")

        print(f"  构建标签...", end=" ", flush=True)
        labels = build_labels(df_1m)
        nc = (labels["label_binary"] == 1).sum()
        np_ = (labels["label_binary"] == 0).sum()
        print(f"{len(labels)} samples (CALL={nc} PUT={np_})")

        # ── Model A (baseline) ──
        print(f"\n  --- Model A (baseline) ---")
        X, y = build_samples(df_1m, labels, use_sr=False)
        if len(X) == 0:
            print("  无样本"); continue

        split = int(len(X) * (1 - args.test_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        model_a, scaler_a = train_model(X_train, y_train, X_test, y_test)

        # Metrics
        X_test_s_a = scaler_a.transform(X_test)
        proba_a = model_a.predict_proba(X_test_s_a)
        pi_a = 1 if 1 in model_a.classes_ else 0
        pa_a = proba_a[:, pi_a]
        auc_a = roc_auc_score(y_test, pa_a)
        brier_a = brier_score_loss(y_test, pa_a)
        acc_a = accuracy_score(y_test, pa_a > 0.5)
        print(f"  AUC={auc_a:.4f} Brier={brier_a:.4f}")

        # Backtest
        bt_a = backtest_trading(model_a, scaler_a, df_1m, labels, payout, use_sr=False)
        stats_a = compute_stats(bt_a, "Model_A")
        print(f"  Trading: {stats_a['n_trades']} trades, WR={stats_a['win_rate']:.1%}, "
              f"PnL={stats_a['total_pnl']:+.0f}U, MaxDD={stats_a['max_drawdown']:.0f}U, "
              f"MaxLosingStreak={stats_a['longest_losing_streak']}")

        # ── Model B (with SR features) ──
        print(f"\n  --- Model B (+ SR features) ---")
        X_b, y_b = build_samples(df_1m, labels, use_sr=True)
        if len(X_b) == 0:
            print("  无样本"); continue

        split_b = int(len(X_b) * (1 - args.test_split))
        X_train_b, X_test_b = X_b[:split_b], X_b[split_b:]
        y_train_b, y_test_b = y_b[:split_b], y_b[split_b:]
        print(f"  Train: {len(X_train_b)} | Test: {len(X_test_b)}")

        model_b, scaler_b = train_model(X_train_b, y_train_b, X_test_b, y_test_b)

        X_test_s_b = scaler_b.transform(X_test_b)
        proba_b = model_b.predict_proba(X_test_s_b)
        pi_b = 1 if 1 in model_b.classes_ else 0
        pa_b = proba_b[:, pi_b]
        auc_b = roc_auc_score(y_test_b, pa_b)
        brier_b = brier_score_loss(y_test_b, pa_b)
        print(f"  AUC={auc_b:.4f} Brier={brier_b:.4f}")

        # Backtest
        bt_b = backtest_trading(model_b, scaler_b, df_1m, labels, payout, use_sr=True)
        stats_b = compute_stats(bt_b, "Model_B")
        print(f"  Trading: {stats_b['n_trades']} trades, WR={stats_b['win_rate']:.1%}, "
              f"PnL={stats_b['total_pnl']:+.0f}U, MaxDD={stats_b['max_drawdown']:.0f}U, "
              f"MaxLosingStreak={stats_b['longest_losing_streak']}")

        # ── Feature Importance ──
        print(f"\n  --- Feature Importance (Model B) ---")
        importances = dict(zip(ALL_FEATURES_B, model_b.feature_importances_))
        top = sorted(importances.items(), key=lambda x: -x[1])[:12]
        sr_in_top = [(n, v) for n, v in top if n in SR_FEATURES]
        print(f"  Top 12 features: {', '.join(f'{n}={v:.0f}' for n, v in top)}")
        if sr_in_top:
            print(f"  SR features in top 12: {', '.join(f'{n}={v:.0f}' for n, v in sr_in_top)}")
        else:
            print(f"  SR features NOT in top 12 -- model barely uses them")

        # ── Conditional Analysis ──
        analyze_conditional(bt_a, f"{sym} - Model A")
        analyze_conditional(bt_b, f"{sym} - Model B")

        # ── Compare A vs B ──
        print(f"\n  --- {sym} Comparison ---")
        print(f"  {'Metric':25s} {'Model A':>10s} {'Model B':>10s} {'Delta':>10s}")
        print(f"  {'-'*55}")
        for metric_key, metric_fmt in [
            ("n_trades", "d"), ("win_rate", ".1%"), ("total_pnl", "+.0fU"),
            ("max_drawdown", "+.0fU"), ("longest_losing_streak", "d"), ("brier", ".4f")]:
            va = stats_a.get(metric_key, 0)
            vb = stats_b.get(metric_key, 0)
            d = vb - va if isinstance(vb, (int, float)) and isinstance(va, (int, float)) else 0
            if "pnl" in metric_key or "drawdown" in metric_key:
                print(f"  {metric_key:25s} {va:+10.0f}U {vb:+10.0f}U {d:+10.0f}U")
            elif metric_key == "win_rate":
                print(f"  {metric_key:25s} {va:10.1%} {vb:10.1%} {d*100:+10.1f}pp")
            else:
                print(f"  {metric_key:25s} {va:10.4f} {vb:10.4f} {d:+10.4f}")

        all_results[sym] = {"model_a": stats_a, "model_b": stats_b, "auc_a": auc_a, "auc_b": auc_b}

        # ── Save Model B ──
        model_path = os.path.join(args.output, f"{sym.lower()}_fast_entry_sr.pkl")
        joblib.dump({
            "model": model_b, "scaler": scaler_b,
            "features": ALL_FEATURES_B,
            "label_vesion": "fast_entry_sr_v1",
            "prediction_horizon": "15m",
            "entry_resolution": "1m",
        }, model_path)
        print(f"  Model B saved: {model_path} ({os.path.getsize(model_path)//1024}KB)")

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    for sym, res in all_results.items():
        a = res["model_a"]; b = res["model_b"]
        print(f"\n  {sym}:")
        print(f"    AUC:     {res['auc_a']:.4f} -> {res['auc_b']:.4f} ({res['auc_b']-res['auc_a']:+.4f})")
        print(f"    Brier:   {a['brier']:.4f} -> {b['brier']:.4f} ({b['brier']-a['brier']:+.4f})")
        td = b['n_trades'] - a['n_trades']
        print(f"    Trades:   {a['n_trades']} -> {b['n_trades']} ({td:+d})")
        if b['n_trades'] > 0 and a['n_trades'] > 0:
            wr_d = (b['win_rate'] - a['win_rate']) * 100
            print(f"    WR:      {a['win_rate']:.1%} -> {b['win_rate']:.1%} ({wr_d:+.1f}pp)")
            print(f"    PnL:     {a['total_pnl']:+.0f}U -> {b['total_pnl']:+.0f}U ({b['total_pnl']-a['total_pnl']:+.0f}U)")
            print(f"    MaxDD:   {a['max_drawdown']:+.0f}U -> {b['max_drawdown']:+.0f}U ({b['max_drawdown']-a['max_drawdown']:+.0f}U)")
            sl_d = b['longest_losing_streak'] - a['longest_losing_streak']
            print(f"    Streak:   {a['longest_losing_streak']} -> {b['longest_losing_streak']} ({sl_d:+d})")

    print(f"\n{'='*70}")
    print(f"  Models saved to: {args.output}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()