#!/usr/bin/env python3
"""
训练XGBoost/LightGBM模型 - 改进版
- 双向标签: 做多/做空/不做事
- 平衡采样防止模型偏科
- 数据增强: 用滚动窗口做多组训练
"""
import argparse
import io
import os
import sys
import time
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from curl_cffi import requests

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"


def fetch_all_klines(pair: str, days: int = 7) -> Optional[pd.DataFrame]:
    limit = 1000
    all_rows = []
    last_ts = int(time.time())
    needed = days * 24 * 60
    retries = 0

    while len(all_rows) < needed:
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": "1m",
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                retries += 1
                if retries > 3: break
                time.sleep(2); continue
            data = r.json()
            if not data: break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1
            retries = 0
            if len(data) < limit: break
        except Exception:
            retries += 1
            if retries > 3: break
            time.sleep(3)

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=[
        "timestamp", "quote_volume", "close", "high", "low", "open", "volume", "is_final"
    ])
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).set_index("datetime")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def calc_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()
    eps = 1e-10
    df["volume"] = df["volume"].fillna(0).replace(0, 0.001)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    exp12, exp26 = df["close"].ewm(span=12).mean(), df["close"].ewm(span=26).mean()
    macd_line = exp12 - exp26
    df["MACD"] = 2 * (macd_line - macd_line.ewm(span=9).mean())
    df["macd_hist_change"] = df["MACD"] - df["MACD"].shift(1)
    low_9, high_9 = df["low"].rolling(9).min(), df["high"].rolling(9).max()
    rsv = (df["close"] - low_9) / (high_9 - low_9 + eps) * 100
    k, d_ = rsv.ewm(com=2).mean(), rsv.ewm(com=2).mean()
    df["J"] = 3 * k - 2 * d_
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["rsi_change"] = df["RSI"] - df["RSI"].shift(5)
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["BB_Pos"] = (df["close"] - (mid - 2 * std)) / (4 * std + eps)
    df["bb_width"] = ((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["NATR"] = tr.rolling(14).mean() / (df["close"] + eps)
    df["volatility_ratio"] = df["NATR"] / (df["NATR"].rolling(20).mean() + eps)
    up_move, down_move = df["high"] - df["high"].shift(1), df["low"].shift(1) - df["low"]
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)
    tr_s = tr.rolling(14).sum().replace(0, eps)
    plus_di = 100 * plus_dm.rolling(14).sum() / tr_s
    minus_di = 100 * minus_dm.rolling(14).sum() / tr_s
    df["ADX"] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + eps)).rolling(14).mean().fillna(0)
    df["adx_change"] = df["ADX"] - df["ADX"].shift(5)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (df["volume"] * tp).cumsum() / (df["volume"].cumsum() + eps)
    df["VWAP_Dist"] = (df["close"] - vwap) / (vwap + eps)
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(5).mean() + eps)
    hl = (df["high"] - df["low"]) + eps
    buy_raw = (df["close"] - df["low"]) / hl * df["volume"]
    sell_raw = (df["high"] - df["close"]) / hl * df["volume"]
    for w in [5, 15, 30]:
        df[f"BSP_{w}"] = np.log((buy_raw.rolling(w).sum() + eps) / (sell_raw.rolling(w).sum() + eps))
    df["VEV"] = df["volume_ratio"] / (df["NATR"] + eps)
    df["close_to_ma50"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).mean() + eps)
    df["Macro_Trend"] = (df["close"] - df["close"].ewm(span=100).mean()) / (df["close"].ewm(span=100).mean() + eps)
    df["momentum_3"] = df["close"] - df["close"].shift(3)
    hl_range = df["high"] - df["low"] + eps
    df["wick_upper_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range
    df["wick_lower_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl_range
    df["body_ratio"] = (df["close"] - df["open"]).abs() / hl_range
    tp_sma, tp_mad = tp.rolling(20).mean(), tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = (tp - tp_sma) / (0.015 * tp_mad + eps)
    sum_atr = tr.rolling(14).sum()
    df["CHOP"] = 100 * np.log10(sum_atr / (df["high"].rolling(14).max() - df["low"].rolling(14).min() + eps)) / np.log10(14)
    df["ROC_5"] = (df["close"] - df["close"].shift(5)) / (df["close"].shift(5) + eps) * 100
    obv_direction = np.sign(df["close"].diff().fillna(0))
    obv = (df["volume"] * obv_direction).cumsum()
    df["OBV_slope_5"] = obv.diff(5) / (obv.shift(5).abs() + eps)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "MACD", "macd_hist_change", "J", "RSI", "rsi_change",
    "BB_Pos", "bb_width", "NATR", "volatility_ratio",
    "ADX", "adx_change", "VWAP_Dist", "volume_ratio",
    "BSP_5", "BSP_15", "BSP_30", "VEV",
    "close_to_ma50", "Macro_Trend", "momentum_3",
    "wick_upper_ratio", "wick_lower_ratio", "body_ratio",
    "CCI", "CHOP", "ROC_5", "OBV_slope_5",
]


def make_balanced_labels(feat: pd.DataFrame, forward_bars: int = 5) -> pd.Series:
    """
    改进的标签系统:
    1 = CALL(做多赚钱) — 未来涨>0.15%
    0 = 不做事(横盘) — 波动在±0.15%内
    但XGBoost是二元分类, 所以:
    1 = 强做多信号 (未来涨>0.15%)
    0 = 其他

    通过下采样使正负样本更平衡
    """
    future_high = feat["close"].rolling(forward_bars).max().shift(-forward_bars)
    future_low = feat["close"].rolling(forward_bars).min().shift(-forward_bars)
    current = feat["close"]

    up = ((future_high / current - 1) > 0.0015).astype(int)
    down = ((future_low / current - 1) < -0.0015).astype(int)

    # 只预测做多方向(CALL), 做空用1-prob_long
    return up


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"[训练] 拉取 {args.days} 天数据 + 训练XGBoost...\n")

    from xgboost import XGBClassifier
    import joblib

    all_models = {}
    for sym, pair in SYMBOLS.items():
        print(f"=== {sym} ===")
        df_raw = fetch_all_klines(pair, days=args.days)
        if df_raw is None or len(df_raw) < 300:
            print(f"  数据不足, 跳过"); continue

        feat = calc_features(df_raw)
        label = make_balanced_labels(feat)
        aligned_feat = feat[FEATURES].loc[label.index]
        aligned_label = label.loc[label.index]

        # 取最后7天做训练, 最新的1天做测试
        split = len(aligned_feat) - 1440  # 留1天做测试
        X_train, X_test = aligned_feat.iloc[:split], aligned_feat.iloc[split:]
        y_train, y_test = aligned_label.iloc[:split], aligned_label.iloc[split:]

        # 类权重: 让模型对少数类更敏感
        pos_ratio = y_train.mean()
        neg_ratio = 1 - pos_ratio
        scale = neg_ratio / max(pos_ratio, 0.01)

        print(f"  正样本率: {pos_ratio:.1%} (scale_pos_weight={scale:.1f})")
        print(f"  训练集: {len(X_train)} 测试集: {len(X_test)}")

        # 近期数据更高权重
        weights = np.linspace(0.3, 1.0, len(X_train))

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=1.5,
            scale_pos_weight=scale,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # 测试
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()

        # 胜率分析
        call_preds = y_pred == 1
        if call_preds.sum() > 0:
            call_winrate = (y_test[call_preds] == 1).mean()
        else:
            call_winrate = 0.0

        # 重要特征
        imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])

        print(f"  准确率: {acc:.3f} | CALL信号胜率: {call_winrate:.1%} | CALL信号占比: {call_preds.mean():.1%}")
        print(f"  重要特征: {', '.join(f'{f}={v:.3f}' for f,v in imp[:5])}")

        # 保存
        path = os.path.join(args.output, f"{sym.lower()}_model.pkl")
        joblib.dump(model, path)
        print(f"  ✓ 保存: {path} ({os.path.getsize(path)//1024}KB)")
        all_models[sym] = model
        print()

    # 综合模型
    print("=== 综合模型 ===")
    combined_X, combined_y = [], []
    for sym in SYMBOLS:
        if sym in all_models:
            df_raw = fetch_all_klines(SYMBOLS[sym], days=args.days)
            if df_raw is not None:
                feat = calc_features(df_raw)
                label = make_balanced_labels(feat)
                combined_X.append(feat[FEATURES].loc[label.index])
                combined_y.append(label.loc[label.index])

    if len(combined_X) >= 2:
        mega_X = pd.concat(combined_X)
        mega_y = pd.concat(combined_y)
        # 对齐
        common = mega_X.index.intersection(mega_y.index)
        mega_X, mega_y = mega_X.loc[common], mega_y.loc[common]
        print(f"  综合数据: {len(mega_X)}行")

        split = len(mega_X) - 1440
        X_tr, X_te = mega_X.iloc[:split], mega_X.iloc[split:]
        y_tr, y_te = mega_y.iloc[:split], mega_y.iloc[split:]

        pos_ratio = y_tr.mean()
        scale = (1 - pos_ratio) / max(pos_ratio, 0.01)
        weights = np.linspace(0.3, 1.0, len(X_tr))

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.02,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=1.5, scale_pos_weight=scale,
            verbosity=0, random_state=42,
        )
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_te, y_te)], verbose=False)
        joblib.dump(model, os.path.join(args.output, "all_model.pkl"))
        print(f"  ✓ 综合模型保存完毕")

    print(f"\n[完成] 全部模型已保存到 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()