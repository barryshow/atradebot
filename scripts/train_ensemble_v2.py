#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 模型训练 — LightGBM + Event Contract 真实标签

与旧版 train_ensemble.py 的区别:
1. 标签: Event Contract 真实结算结果 (WIN/LOSS/TIE)，不是 "next candle up/down"
2. 数据: 1 分钟 K 线 (用于构建精确的 entry/expiry 时间点)
3. 多期限: 支持 5m/15m/30m/60m，每个期限独立训练
4. Look-Ahead Bias 防护: 严格时间序列切分
5. 排除 TIE 样本

用法:
    python scripts/train_ensemble_v2.py --days 60 --expiries 15
    python scripts/train_ensemble_v2.py --days 90 --expiries 5,15,30
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

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.engine.label_builder import LabelBuilder, EXPIRY_HORIZONS
from lib.engine.models import TrainLabel

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"

# 特征列表（与旧版兼容，后续会扩展）
FEATURES = [
    "ret_1", "ret_3", "ret_6",
    "MACD", "MACD_hist", "RSI",
    "BB_Pos", "BB_width", "ATR_pct", "ADX",
    "price_vs_MA20", "price_vs_MA50", "MA_trend",
    "VWAP_dist", "vol_ratio", "OBV_trend", "CCI", "CHOP",
    "body_pct", "is_green",
]


def fetch_klines_1m(pair: str, days: int = 60) -> pd.DataFrame | None:
    """拉取 1 分钟 K 线数据（用于构建精确标签）"""
    limit = 1000
    all_rows = []
    last_ts = int(time.time())
    needed = days * 24 * 60  # 每天的分钟数

    for attempt in range(5):
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": "1m",
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                time.sleep(2)
                continue
            data = r.json()
            if not data or len(data) < 2:
                break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 60
            if len(data) < limit:
                break
        except Exception:
            time.sleep(3)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows, columns=["ts", "qv", "close", "high", "low", "open", "volume", "final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "close"])


def calc_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    """在 1m 数据上计算特征（与旧版 engine/_calc_30m_features 兼容）"""
    d = df_1m.copy()
    eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)

    d["ret_1"] = d["close"].pct_change(1).fillna(0)
    d["ret_3"] = d["close"].pct_change(3).fillna(0)
    d["ret_6"] = d["close"].pct_change(6).fillna(0)

    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    d["MACD"] = 2 * (macd - macd.ewm(span=9, adjust=False).mean()).fillna(0)
    d["MACD_hist"] = d["MACD"].fillna(0)

    delta = d["close"].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = (100 - (100 / (1 + gain / loss))).fillna(50)

    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std().fillna(0)
    d["BB_Pos"] = ((d["close"] - (mid - 2 * std)) / (4 * std + eps)).clip(0, 1).fillna(0.5)
    d["BB_width"] = (((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)).fillna(0)

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"] - d["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
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

    return d.replace([np.inf, -np.inf], np.nan).dropna()


def align_features_with_labels(
    feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    min_history: int = 200,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    对齐特征和标签，严格防止 Look-Ahead Bias。

    对每个标签样本:
    - 找到 entry_ts 对应的特征行
    - 只使用 entry_ts 之前的特征（特征已计算，不包含未来信息）
    - 特征计算中的 rolling 窗口天然保证了这一点

    返回:
        X: 特征矩阵
        y: 标签 (binary: 1=WIN, 0=LOSS)
        aligned_indices: 对齐后的索引列表
    """
    if feat_df.empty or labels_df.empty:
        return np.array([]), np.array([]), []

    X_list = []
    y_list = []
    aligned = []

    feat_index = feat_df.index
    feat_values = feat_df[FEATURES].values

    for _, label_row in labels_df.iterrows():
        entry_ts_ms = label_row["entry_ts"]
        # 统一转为 tz-naive datetime（特征索引通常是 tz-naive）
        entry_dt = pd.Timestamp(entry_ts_ms, unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)

        # 如果特征索引是 tz-aware，转换 entry_dt
        if feat_index.tz is not None:
            entry_dt = entry_dt.tz_localize("UTC")

        # 找到 entry_ts 对应的特征行索引
        # 使用 <= entry_dt 的最近一行特征
        mask = feat_index <= entry_dt
        if not mask.any():
            continue

        feat_idx = mask.sum() - 1  # 最后一个 <= entry_dt 的索引
        if feat_idx < min_history:
            continue

        try:
            feat_row = feat_values[feat_idx]
            # 检查特征是否有效（无 NaN/Inf）
            if np.any(np.isnan(feat_row)) or np.any(np.isinf(feat_row)):
                continue

            X_list.append(feat_row)
            y_list.append(label_row["label_binary"])  # 1=WIN, 0=LOSS
            aligned.append(label_row.to_dict())

        except (IndexError, KeyError):
            continue

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), aligned


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    symbol: str,
    expiry: int,
    output_dir: str,
) -> dict:
    """训练单个 LightGBM 模型"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 类别权重
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(0)],
    )

    # ── 评估 ──
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)

    # 分方向评估
    pos_idx = 1 if hasattr(model, "classes_") and 1 in model.classes_ else 0
    proba = y_prob[:, pos_idx]

    call_preds = y_pred == 1
    call_wr = y_test[call_preds].mean() if call_preds.sum() > 0 else 0.0
    put_preds = y_pred == 0
    put_wr = (1 - y_test[put_preds]).mean() if put_preds.sum() > 0 else 0.0

    brier = brier_score_loss(y_test, proba)
    ll = log_loss(y_test, proba)

    # 保存模型
    model_name = f"{symbol.lower()}_{expiry}m_ensemble_v2.pkl"
    model_path = os.path.join(output_dir, model_name)
    joblib.dump({
        "ensemble": model,
        "scaler": scaler,
        "features": FEATURES,
        "best_threshold": 0.50,
        "label_version": "v2_event_contract",
        "expiry_minutes": expiry,
        "price_source": "gate_io",
    }, model_path)

    # 特征重要性
    importances = dict(zip(FEATURES, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    return {
        "symbol": symbol,
        "expiry": expiry,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(acc, 4),
        "call_win_rate": round(call_wr, 4),
        "put_win_rate": round(put_wr, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "model_path": model_path,
        "model_size_kb": os.path.getsize(model_path) // 1024,
        "top_features": top_features,
        "pos_ratio_train": round(n_pos / len(y_train), 4),
        "pos_ratio_test": round((y_test == 1).sum() / len(y_test), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="EventEdge V2 模型训练")
    parser.add_argument("--days", type=int, default=60, help="训练数据天数")
    parser.add_argument("--expiries", type=str, default="15", help="到期期限，逗号分隔 (如 5,15,30)")
    parser.add_argument("--output", type=str, default="./models", help="模型输出目录")
    parser.add_argument("--symbols", type=str, default="", help="训练品种，逗号分隔 (默认全部)")
    parser.add_argument("--test-split", type=float, default=0.20, help="测试集比例")
    args = parser.parse_args()

    expiries = [int(x.strip()) for x in args.expiries.split(",")]
    symbols = args.symbols.split(",") if args.symbols else list(SYMBOLS.keys())
    symbols = [s.strip() for s in symbols if s.strip() and s.strip() in SYMBOLS]

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*65}")
    print(f"  EventEdge V2 模型训练")
    print(f"  Label: Event Contract Outcome (WIN/LOSS/TIE)")
    print(f"  Expiries: {[f'{e}m' for e in expiries]}")
    print(f"  Symbols: {symbols}")
    print(f"  Data: {args.days} 天 | Test split: {args.test_split:.0%}")
    print(f"{'='*65}\n")

    all_results = []

    for sym in symbols:
        pair = SYMBOLS[sym]
        print(f"\n{'─'*65}")
        print(f"  {sym} ({pair})")
        print(f"{'─'*65}")

        # 1. 拉取 1m 数据
        print(f"  拉取 1m K线数据...", end=" ", flush=True)
        df_1m = fetch_klines_1m(pair, days=args.days)
        if df_1m is None or len(df_1m) < 500:
            print(f"数据不足 ({len(df_1m) if df_1m is not None else 0} 行)")
            continue
        print(f"{len(df_1m)} 行 ({df_1m.index[0].date()} ~ {df_1m.index[-1].date()})")

        # 2. 计算特征
        print(f"  计算特征...", end=" ", flush=True)
        feat_df = calc_features(df_1m)
        print(f"{len(feat_df)} 行 (含特征)")

        # 3. 构建标签
        print(f"  构建 Event Contract 标签...")
        builder = LabelBuilder(price_source="gate_io", expiries=expiries)
        labels_df = builder.build_labels(df_1m, symbol=sym, min_samples=200)
        if labels_df.empty:
            print(f"  标签构建失败")
            continue
        binary_labels = builder.filter_binary_labels(labels_df)
        builder.print_stats(labels_df)

        # 4. 对齐特征和标签
        print(f"\n  对齐特征与标签...", end=" ", flush=True)
        X, y, aligned = align_features_with_labels(feat_df, binary_labels, min_history=200)
        if len(X) == 0:
            print("无有效样本")
            continue
        print(f"{len(X)} 个样本")

        # 5. 按 expiry 分组训练
        for expiry in expiries:
            # 筛选该 expiry 的样本
            expiry_mask = np.array([a["expiry_minutes"] == expiry for a in aligned])
            X_exp = X[expiry_mask]
            y_exp = y[expiry_mask]

            if len(X_exp) < 200:
                print(f"\n  [{expiry}m] 样本不足 ({len(X_exp)} < 200), 跳过")
                continue

            # 时间序列切分
            split = int(len(X_exp) * (1 - args.test_split))
            X_train, X_test = X_exp[:split], X_exp[split:]
            y_train, y_test = y_exp[:split], y_exp[split:]

            print(f"\n  [{expiry}m] 训练集: {len(X_train)} | 测试集: {len(X_test)}")
            print(f"    CALL: {(y_train == 1).sum()}/{len(y_train)} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

            result = train_single_model(
                X_train, y_train, X_test, y_test,
                symbol=sym, expiry=expiry, output_dir=args.output,
            )
            all_results.append(result)

            # 打印结果
            print(f"    准确率: {result['accuracy']:.1%} | "
                  f"CALL胜率: {result['call_win_rate']:.1%} | "
                  f"PUT胜率: {result['put_win_rate']:.1%}")
            print(f"    Brier: {result['brier_score']:.4f} | "
                  f"LogLoss: {result['log_loss']:.4f}")
            print(f"    重要特征: {', '.join(f'{f}={v:.3f}' for f, v in result['top_features'])}")
            print(f"    模型: {result['model_path']} ({result['model_size_kb']}KB)")

    # ── 汇总 ──
    print(f"\n\n{'='*65}")
    print(f"  训练汇总")
    print(f"{'='*65}")
    if all_results:
        print(f"\n  {'品种':<10} {'期限':>5} {'Train':>7} {'Test':>7} {'Acc':>7} {'CALL':>7} {'PUT':>7} {'Brier':>7}")
        print(f"  {'─'*60}")
        for r in sorted(all_results, key=lambda x: (x["symbol"], x["expiry"])):
            print(f"  {r['symbol']:<10} {str(r['expiry'])+'m':>5} {r['train_samples']:>7} {r['test_samples']:>7} "
                  f"{r['accuracy']:>7.1%} {r['call_win_rate']:>7.1%} {r['put_win_rate']:>7.1%} {r['brier_score']:>7.4f}")
    else:
        print("  无训练结果")
    print(f"\n  模型保存目录: {os.path.abspath(args.output)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()