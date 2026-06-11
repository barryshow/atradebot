#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高胜率模型训练 — LightGBM + 自适应标签
预测未来1根15分钟K线的方向
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
from sklearn.metrics import accuracy_score
import joblib

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "MACD", "MACD_hist", "RSI",
    "BB_Pos", "BB_width", "ATR_pct", "ADX",
    "price_vs_MA20", "price_vs_MA50", "MA_trend",
    "VWAP_dist", "vol_ratio", "OBV_trend", "CCI", "CHOP",
    "body_pct", "is_green",
]


def fetch_klines(pair: str, interval: str = "15m", days: int = 60) -> pd.DataFrame | None:
    limit = 1000
    all_rows, last_ts = [], int(time.time())
    needed = days * 96
    while len(all_rows) < needed:
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
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


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)
    e12 = d["close"].ewm(span=12).mean(); e26 = d["close"].ewm(span=26).mean()
    macd = e12 - e26
    d["MACD"] = 2 * (macd - macd.ewm(span=9).mean()); d["MACD_hist"] = d["MACD"]
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = 100 - (100 / (1 + gain / loss))
    mid = d["close"].rolling(20).mean(); std = d["close"].rolling(20).std()
    d["BB_Pos"] = (d["close"] - (mid - 2 * std)) / (4 * std + eps)
    d["BB_width"] = ((mid + 2 * std) - (mid - 2 * std)) / mid
    tr = pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift(1)).abs(),(d["low"]-d["close"].shift(1)).abs()], axis=1).max(axis=1)
    d["ATR_pct"] = tr.rolling(14).mean() / d["close"]
    up = d["high"] - d["high"].shift(1); dn = d["low"].shift(1) - d["low"]
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=d.index)
    tr14 = tr.rolling(14).sum().replace(0, eps)
    pdi = 100 * pdm.rolling(14).sum() / tr14; ndi = 100 * ndm.rolling(14).sum() / tr14
    d["ADX"] = (100 * abs(pdi - ndi) / (pdi + ndi + eps)).rolling(14).mean()
    d["MA10"] = d["close"].rolling(10).mean().bfill()
    d["MA20"] = d["close"].rolling(20).mean().bfill()
    d["MA50"] = d["close"].rolling(50).mean().bfill()
    d["price_vs_MA20"] = (d["close"] - d["MA20"]) / d["MA20"]
    d["price_vs_MA50"] = (d["close"] - d["MA50"]) / d["MA50"]
    d["MA_trend"] = np.sign(d["MA10"] - d["MA20"])
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vwap = (d["volume"] * tp).cumsum() / d["volume"].cumsum()
    d["VWAP_dist"] = (d["close"] - vwap) / vwap
    d["vol_ratio"] = d["volume"] / (d["volume"].rolling(5).mean() + eps)
    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"] * obv_dir).cumsum()
    d["OBV_trend"] = np.sign(obv - obv.shift(5))
    tp_sma = tp.rolling(20).mean(); tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["CCI"] = (tp - tp_sma) / (0.015 * tp_mad + eps)
    atr14 = tr.rolling(14).sum()
    d["CHOP"] = 100 * np.log10(atr14 / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + eps)) / np.log10(14)
    body = abs(d["close"] - d["open"])
    d["body_pct"] = body / (d["high"] - d["low"] + eps)
    d["is_green"] = (d["close"] > d["open"]).astype(int)
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def make_labels(feat: pd.DataFrame) -> pd.Series:
    """
    标签: 预测未来1根K线方向
    用最近30天的ATR中位数作为动态阈值
    """
    # 未来1根K线涨跌幅
    ret_fwd = feat["close"].shift(-1) / feat["close"] - 1

    # 动态阈值: 最近30天ATR的20%
    atr = feat["ATR_pct"].rolling(96).median()  # 96根15m ≈ 1天
    dynamic_threshold = atr * 0.3  # ATR的30%
    dynamic_threshold = dynamic_threshold.fillna(0.001)  # 兜底0.1%

    label = pd.Series(0, index=feat.index, dtype=int)
    label[ret_fwd > dynamic_threshold] = 1    # CALL
    label[ret_fwd < -dynamic_threshold] = -1   # PUT
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--output", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"LightGBM训练 | 15分钟K线 | {args.days}天数据 | 自适应ATR阈值\n")

    for sym, pair in SYMBOLS.items():
        print(f"=== {sym} ===")
        df = fetch_klines(pair, interval="15m", days=args.days)
        if df is None or len(df) < 100:
            print(f"  数据不足\n"); continue

        feat = calc_features(df)
        label = make_labels(feat)

        # 对齐特征和标签
        valid = label.index.intersection(feat.index)
        X_all = feat.loc[valid][FEATURES].iloc[50:-1]
        y_all = label.loc[valid].iloc[50:-1]

        has_dir = y_all != 0
        X = X_all[has_dir]; y = y_all[has_dir].map({-1: 0, 1: 1})

        if len(X) < 100:
            print(f"  有效样本不足({len(X)}), 降低阈值重试...")
            # 兜底: 用固定0.1%阈值
            ret_fwd = feat["close"].shift(-1) / feat["close"] - 1
            label2 = pd.Series(0, index=feat.index, dtype=int)
            label2[ret_fwd > 0.001] = 1
            label2[ret_fwd < -0.001] = -1
            valid2 = label2.index.intersection(feat.index)
            X_all2 = feat.loc[valid2][FEATURES].iloc[50:-1]
            y_all2 = label2.loc[valid2].iloc[50:-1]
            has_dir2 = y_all2 != 0
            X = X_all2[has_dir2]; y = y_all2[has_dir2].map({-1: 0, 1: 1})
            if len(X) < 100:
                print(f"  兜底仍不足({len(X)}), 跳过\n"); continue

        split = int(len(X) * 0.80)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        call_pct = y_tr.mean() * 100
        put_pct = (1 - y_tr.mean()) * 100
        print(f"  样本: {len(X)} | 训练: {len(X_tr)} 测试: {len(X_te)}")
        print(f"  正负比: CALL={call_pct:.0f}% PUT={put_pct:.0f}%")

        # === LightGBM（处理类别不平衡） ===
        scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum() if (y_tr == 1).sum() > 0 else 1.0

        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            class_weight="balanced", random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)],
                      eval_metric="auc", callbacks=[lgb.log_evaluation(0)])

        # === 评估 ===
        print(f"\n  测试集表现:")
        for name, Xp in [("LightGBM", X_te_s)]:
            y_p = lgb_model.predict(Xp)
            acc = accuracy_score(y_te, y_p)

            call_idx = y_p == 1
            call_wr = y_te[call_idx].mean() if call_idx.sum() > 0 else 0
            put_idx = y_p == 0
            put_wr = (1 - y_te[put_idx]).mean() if put_idx.sum() > 0 else 0

            exp_val = call_wr * 0.80 - put_wr * 0.80
            print(f"  {name:>12}: 准确率{acc:.1%} | CALL胜率{call_wr:.1%} PUT胜率{put_wr:.1%} | 期望{exp_val:+.2%}")

        # === 保存 ===
        out_path = os.path.join(args.output, f"{sym.lower()}_ensemble.pkl")
        joblib.dump({
            "ensemble": lgb_model, "scaler": scaler,
            "features": FEATURES, "best_threshold": 0.50,
        }, out_path)
        print(f"  已保存: {out_path}")
        print()

    print("全部完成!")


if __name__ == "__main__":
    main()
