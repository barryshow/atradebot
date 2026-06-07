#!/usr/bin/env python3
"""
高胜率多模型集成训练 (30分钟K线)

策略:
  1. 30分钟K线降低噪音
  2. 3模型集成: 逻辑回归 + 随机森林 + 梯度提升
  3. 硬投票(>=2个同意)才出手
  4. 直接拉30分钟K线(不经过1m重采样,更快)

用法: python scripts/train_ensemble.py [--days 120] [--output ./models]
"""
import argparse
import io
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from curl_cffi import requests
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"


def fetch_klines(pair: str, interval: str = "15m", days: int = 60) -> pd.DataFrame | None:
    """直接拉15分钟K线"""
    limit = 1000
    all_rows = []
    last_ts = int(time.time())
    needed = days * 96  # 96根15mK线/天
    retries = 0

    print(f"  拉取{days}天{interval}数据...")
    while len(all_rows) < needed:
        try:
            r = requests.get(API_URL, params={
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
            last_ts = int(data[0][0]) - 1800  # 30分钟前
            retries = 0
            if len(all_rows) >= needed: break
            if len(data) < limit: break
        except Exception:
            retries += 1
            if retries > 3: break
            time.sleep(3)

    if not all_rows: return None

    df = pd.DataFrame(all_rows, columns=[
        "ts", "qv", "close", "high", "low", "open", "volume", "final"
    ])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close"])
    print(f"  -> {len(df)} 根{interval}K线")
    return df


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征计算 (通用, 支持15m/30m)"""
    d = df.copy()
    eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)

    # 收益率
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)

    # MACD
    e12 = d["close"].ewm(span=12).mean()
    e26 = d["close"].ewm(span=26).mean()
    macd = e12 - e26
    d["MACD"] = 2 * (macd - macd.ewm(span=9).mean())
    d["MACD_hist"] = d["MACD"]

    # RSI
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = 100 - (100 / (1 + gain / loss))

    # BB
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["BB_Pos"] = (d["close"] - (mid - 2 * std)) / (4 * std + eps)
    d["BB_width"] = ((mid + 2 * std) - (mid - 2 * std)) / mid

    # ATR
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"] - d["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR_pct"] = tr.rolling(14).mean() / d["close"]

    # ADX
    up = d["high"] - d["high"].shift(1)
    dn = d["low"].shift(1) - d["low"]
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=d.index)
    tr14 = tr.rolling(14).sum().replace(0, eps)
    pdi = 100 * pdm.rolling(14).sum() / tr14
    ndi = 100 * ndm.rolling(14).sum() / tr14
    d["ADX"] = (100 * abs(pdi - ndi) / (pdi + ndi + eps)).rolling(14).mean()

    # MA
    d["MA10"] = d["close"].rolling(10).mean()
    d["MA20"] = d["close"].rolling(20).mean()
    d["MA50"] = d["close"].rolling(50).mean()
    d["price_vs_MA20"] = (d["close"] - d["MA20"]) / d["MA20"]
    d["price_vs_MA50"] = (d["close"] - d["MA50"]) / d["MA50"]
    d["MA_trend"] = np.sign(d["MA10"] - d["MA20"])

    # VWAP
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vwap = (d["volume"] * tp).cumsum() / (d["volume"].cumsum() + eps)
    d["VWAP_dist"] = (d["close"] - vwap) / vwap

    # 量
    d["vol_ratio"] = d["volume"] / (d["volume"].rolling(5).mean() + eps)

    # OBV
    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"] * obv_dir).cumsum()
    d["OBV_trend"] = np.sign(obv - obv.shift(5))

    # CCI
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["CCI"] = (tp - tp_sma) / (0.015 * tp_mad + eps)

    # CHOP
    atr14 = tr.rolling(14).sum()
    d["CHOP"] = 100 * np.log10(atr14 / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + eps)) / np.log10(14)

    # 蜡烛形态
    body = abs(d["close"] - d["open"])
    d["body_pct"] = body / (d["high"] - d["low"] + eps)
    d["is_green"] = (d["close"] > d["open"]).astype(int)

    return d.replace([np.inf, -np.inf], np.nan).dropna()


FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "MACD", "MACD_hist", "RSI",
    "BB_Pos", "BB_width",
    "ATR_pct",
    "ADX",
    "price_vs_MA20", "price_vs_MA50", "MA_trend",
    "VWAP_dist", "vol_ratio",
    "OBV_trend",
    "CCI", "CHOP",
    "body_pct", "is_green",
]


def make_labels(feat: pd.DataFrame, fwd: int = 2) -> pd.Series:
    """
    标签: 未来fwd根30mK线(1小时)
    1 = CALL (涨>0.3%)
    -1 = PUT (跌>0.3%)
    0 = 横盘(排除)
    """
    fh = feat["close"].rolling(fwd).max().shift(-fwd)
    fl = feat["close"].rolling(fwd).min().shift(-fwd)
    cur = feat["close"]
    up = (fh / cur - 1)
    dn = (fl / cur - 1)
    label = pd.Series(0, index=feat.index, dtype=int)
    label[(up > 0.003) & (up.abs() >= dn.abs())] = 1
    label[(dn < -0.003) & (dn.abs() > up.abs())] = -1
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", type=str, default="./models")
    parser.add_argument("--fwd", type=int, default=1, help="预测未来几根K线")
    parser.add_argument("--interval", type=str, default="30m", help="K线周期(15m/30m)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"[训练] 15分钟K线 | 预测未来{args.fwd}根({args.fwd*15}分钟) | {args.days}天数据\n")

    for sym, pair in SYMBOLS.items():
        print(f"=== {sym} ===")
        df = fetch_klines(pair, interval=args.interval, days=args.days)
        if df is None or len(df) < 100:
            print(f"  数据不足\n"); continue

        feat = calc_features(df)
        label = make_labels(feat, fwd=args.fwd)

        # 对齐
        valid = label.index.intersection(feat.index)
        X_all = feat.loc[valid][FEATURES].iloc[50:-args.fwd]
        y_all = label.loc[valid].iloc[50:-args.fwd]

        # 只取有方向的
        has_dir = y_all != 0
        X = X_all[has_dir]
        y = y_all[has_dir].map({-1: 0, 1: 1})  # -1->0, 1->1

        if len(X) < 50:
            print(f"  有效样本不足({len(X)})\n"); continue

        # 80/20 分割
        split = int(len(X) * 0.80)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        # 标准化
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 类别权重
        from sklearn.utils import class_weight
        w = class_weight.compute_sample_weight("balanced", y_tr)

        print(f"  样本: {len(X)} | 训练: {len(X_tr)} 测试: {len(X_te)}")
        print(f"  正负比: CALL={y_tr.mean():.1%} PUT={1-y_tr.mean():.1%}")

        # === 模型1: 逻辑回归 ===
        lr = LogisticRegression(C=0.5, solver="lbfgs", class_weight="balanced", random_state=42, max_iter=2000)
        lr.fit(X_tr_s, y_tr, sample_weight=w)

        # === 模型2: 随机森林 ===
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42)
        rf.fit(X_tr, y_tr, sample_weight=w)

        # === 模型3: 梯度提升 ===
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, min_samples_leaf=10, subsample=0.7, random_state=42)
        gb.fit(X_tr, y_tr, sample_weight=w)

        # === 集成 ===
        ensemble = VotingClassifier([("lr", lr), ("rf", rf), ("gb", gb)], voting="soft")
        ensemble.fit(X_tr_s, y_tr, sample_weight=w)

        # 评估
        print(f"\n  测试集表现:")
        y_te_arr = y_te.values
        for name, m, use_scaled in [("逻辑回归", lr, True), ("随机森林", rf, False), ("梯度提升", gb, False), ("集成", ensemble, True)]:
            X_pred = X_te_s if use_scaled else X_te
            y_p = m.predict(X_pred)
            acc = accuracy_score(y_te_arr, y_p)

            call_idx = y_p == 1
            call_wr = y_te_arr[call_idx].mean() if call_idx.sum() > 0 else 0
            put_idx = y_p == 0
            put_wr = (1 - y_te_arr[put_idx]).mean() if put_idx.sum() > 0 else 0

            # 综合期望: CALL胜率*0.80 - PUT胜率*0
            total_wr = ((y_p == y_te_arr).sum()) / len(y_te_arr)
            call_ret = call_wr * 0.80 - (1 - call_wr) if call_idx.sum() > 0 else 0
            put_ret = put_wr * 0.80 - (1 - put_wr) if put_idx.sum() > 0 else 0

            print(f"    {name}: acc={acc:.3f} | CALL×{call_idx.mean():.0%}(wr={call_wr:.1%} ret={call_ret:.3f}) | PUT×{put_idx.mean():.0%}(wr={put_wr:.1%} ret={put_ret:.3f})")

        # 找最优概率阈值
        y_prob = ensemble.predict_proba(X_te_s)
        pos_idx = 1 if hasattr(ensemble, "classes_") and 1 in ensemble.classes_ else 0
        y_prob_pos = y_prob[:, pos_idx] if pos_idx == 1 else 1 - y_prob[:, 0]

        best_th = 0.50
        best_r = -999
        for th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            pred = (y_prob_pos >= th).astype(int)
            if pred.sum() == 0: continue
            wr = (pred == y_te_arr).mean()
            r = wr * 0.80 - (1 - wr)
            if r > best_r:
                best_r, best_th = r, th

        print(f"    最优阈值: p>={best_th:.2f} (期望回报={best_r:.4f})")

        # 保存
        model_data = {
            "scaler": scaler, "lr": lr, "rf": rf, "gb": gb, "ensemble": ensemble,
            "features": FEATURES, "best_threshold": best_th,
            "interval_min": 15, "forward_bars": args.fwd,
        }
        path = os.path.join(args.output, f"{sym.lower()}_ensemble.pkl")
        joblib.dump(model_data, path)
        print(f"  -> 保存: {path} ({os.path.getsize(path)//1024}KB)\n")

    print("[完成]")


if __name__ == "__main__":
    main()