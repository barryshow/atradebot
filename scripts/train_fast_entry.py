#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Entry LightGBM 训练 — 1m 分辨率样本

每个 1m K 线时间点作为一个潜在 entry point，
预测 15 分钟后的事件合约结果。

与 Slow Model (15m) 的关键区别:
- 样本频率: 每 1m 一个，不是每 15m
- 特征: 1m/5m 多周期 + 实时微特征
- 目标: 仍是 15 分钟后的事件合约结果

用法:
    python scripts/train_fast_entry.py --days 90 --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
import argparse, io, os, sys, time, warnings
warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from curl_cffi import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.engine.multi_timeframe_features import (
    compute_fast_entry_features, FAST_FEATURES, build_fast_feature_vector,
    compute_1m_features, compute_5m_features,
)

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}


def fetch_klines(pair: str, interval: str = "1m", days: int = 90) -> Optional[pd.DataFrame]:
    """拉取 1m K 线"""
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


def build_labels_1m(df_1m: pd.DataFrame, forward_minutes: int = 15, min_move: float = 0.0003) -> pd.DataFrame:
    """
    每 1m 生成一个样本，预测 15 分钟后方向。
    过滤 |move| < min_move 的微小波动。
    """
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
        })
    return pd.DataFrame(rows)


def aggregate_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """从 1m 聚合 5m K 线"""
    return df_1m.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def build_training_samples(df_1m: pd.DataFrame, labels: pd.DataFrame) -> tuple:
    """为每个标签构建特征"""
    df_5m = aggregate_5m(df_1m)
    X_list, y_list = [], []

    for _, row in labels.iterrows():
        entry_ts = row["entry_ts"]
        entry_dt = pd.Timestamp(entry_ts, unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)

        # 1m K 线: 只用 entry_dt 之前的
        idx_1m = df_1m.index.tz_localize(None) if df_1m.index.tz is not None else df_1m.index
        mask_1m = idx_1m <= entry_dt
        if not mask_1m.any() or mask_1m.sum() < 50:
            continue
        df_1m_before = df_1m[mask_1m].copy()

        # 5m K 线: 只用 entry_dt 之前的
        idx_5m = df_5m.index.tz_localize(None) if df_5m.index.tz is not None else df_5m.index
        mask_5m = idx_5m <= entry_dt
        if not mask_5m.any() or mask_5m.sum() < 10:
            continue
        df_5m_before = df_5m[mask_5m].copy()

        # 计算特征
        features = compute_fast_entry_features(
            symbol="", realtime=None,
            df_1m=df_1m_before, df_5m=df_5m_before,
            slow_context={"probability": 0.50, "regime": "RANGE", "trend_strength": 0, "volatility": 0},
        )

        try:
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


def main():
    parser = argparse.ArgumentParser(description="Fast Entry LightGBM 训练")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--test-split", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="./models")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*65}")
    print(f"  Fast Entry LightGBM 训练 (1m resolution)")
    print(f"  Symbols: {symbols} | Days: {args.days} | Features: {len(FAST_FEATURES)}")
    print(f"{'='*65}")

    for sym in symbols:
        pair = SYMBOLS.get(sym)
        if not pair: continue
        payout = PAYOUTS.get(sym, 0.80)
        be_prob = 1.0 / (1.0 + payout)

        print(f"\n{'─'*65}")
        print(f"  {sym} (payout={payout}, BE={be_prob:.1%})")
        print(f"{'─'*65}")

        print(f"  拉取 1m K线...", end=" ", flush=True)
        df_1m = fetch_klines(pair, interval="1m", days=args.days)
        if df_1m is None or len(df_1m) < 500:
            print(f"数据不足"); continue
        print(f"{len(df_1m)} rows ({df_1m.index[0]} ~ {df_1m.index[-1]})")

        print(f"  构建标签...", end=" ", flush=True)
        labels = build_labels_1m(df_1m, forward_minutes=15)
        n_call = (labels["label_binary"] == 1).sum()
        n_put = (labels["label_binary"] == 0).sum()
        print(f"{len(labels)} samples (CALL={n_call} PUT={n_put})")

        print(f"  构建特征...", end=" ", flush=True)
        X, y = build_training_samples(df_1m, labels)
        if len(X) == 0:
            print("无样本"); continue
        print(f"{len(X)} samples")

        split = int(len(X) * (1 - args.test_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

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

        y_prob = model.predict_proba(X_test_s)
        pos_idx = 1 if 1 in model.classes_ else 0
        proba = y_prob[:, pos_idx]

        prob_mean = float(np.mean(proba))
        prob_std = float(np.std(proba))
        acc = accuracy_score(y_test, proba > 0.5)
        brier = brier_score_loss(y_test, proba)

        print(f"\n  AUC: {float(model.best_score_['valid_0']['auc']):.4f}")
        print(f"  Acc: {acc:.1%} | Prob: mean={prob_mean:.3f} std={prob_std:.4f}")
        print(f"  Brier: {brier:.4f}")

        # 概率校准
        buckets = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 1.0]
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i+1]
            mask = (proba >= lo) & (proba < hi)
            if mask.sum() >= 5:
                actual = y_test[mask].mean()
                print(f"    [{lo:.2f}-{hi:.2f}]: {mask.sum()} samples, pred={proba[mask].mean():.1%}, actual={actual:.1%}")

        # 保存
        model_path = os.path.join(args.output, f"{sym.lower()}_fast_entry.pkl")
        joblib.dump({
            "model": model, "scaler": scaler,
            "features": FAST_FEATURES,
            "label_version": "fast_entry_v1",
            "prediction_horizon": "15m",
            "entry_resolution": "1m",
        }, model_path)
        print(f"  Model: {model_path} ({os.path.getsize(model_path)//1024}KB)")

        # 重要特征
        importances = dict(zip(FAST_FEATURES, model.feature_importances_))
        top = sorted(importances.items(), key=lambda x: -x[1])[:8]
        print(f"  Top features: {', '.join(f'{f}={v:.0f}' for f, v in top)}")

    print(f"\n{'='*65}")
    print(f"  Fast Entry Training Complete")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()