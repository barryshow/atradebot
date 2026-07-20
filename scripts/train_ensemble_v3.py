#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V3 模型训练 — 方向预测 + 噪音过滤 + 15m K线

与 V2 的关键区别（修复）:
1. 用 15m K 线（不是 1m）— 每页 1000 行 ≈ 10 天，5 页 ≈ 50 天
2. 每个 entry point 只生成 1 条样本（预测涨跌），不是 CALL+PUT 两条矛盾样本
3. 噪音过滤: 排除 |move| < min_move 的小波动
4. 31 维丰富特征 + 时间加权 + early stopping

用法:
    python scripts/train_ensemble_v3.py --days 60 --expiries 15
    python scripts/train_ensemble_v3.py --days 90 --expiries 5,15,30 --min-move 0.001
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

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"

# ── 31 维特征 ──
FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "MACD", "macd_hist_change", "RSI", "rsi_change",
    "ROC_5", "momentum_3", "Macro_Trend",
    "BB_Pos", "bb_width", "NATR", "volatility_ratio",
    "ADX", "adx_change",
    "VWAP_Dist", "close_to_ma50", "MA_trend",
    "volume_ratio", "VEV",
    "BSP_5", "BSP_15", "BSP_30",
    "wick_upper_ratio", "wick_lower_ratio", "body_ratio",
    "CCI", "CHOP", "OBV_slope_5", "J",
]


def fetch_klines(pair: str, interval: str = "15m", days: int = 60) -> pd.DataFrame | None:
    """
    拉取 K 线数据。用 15m 间隔：每页 1000 行 ≈ 10.4 天，能覆盖 50+ 天。
    """
    limit = 1000
    all_rows = []
    last_ts = int(time.time())
    needed = days * 24 * 60 // int(interval.replace("m", ""))  # 目标行数

    for _ in range(10):  # 最多 10 页 ≈ 100+ 天
        if len(all_rows) >= needed:
            break
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data)
            # 下一页: 从最早那根 K 线之前继续
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


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 K 线数据上计算 31 维特征"""
    d = df.copy(); eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)

    d["hour_sin"] = np.sin(2*np.pi*d.index.hour/24)
    d["hour_cos"] = np.cos(2*np.pi*d.index.hour/24)
    d["dow_sin"] = np.sin(2*np.pi*d.index.dayofweek/7)
    d["dow_cos"] = np.cos(2*np.pi*d.index.dayofweek/7)

    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()
    macd_line = e12 - e26
    d["MACD"] = 2*(macd_line - macd_line.ewm(span=9, adjust=False).mean()).fillna(0)
    d["macd_hist_change"] = d["MACD"] - d["MACD"].shift(1)

    low_9 = d["low"].rolling(9).min(); high_9 = d["high"].rolling(9).max()
    rsv = (d["close"]-low_9)/(high_9-low_9+eps)*100
    k = rsv.ewm(com=2, adjust=False).mean()
    d_val = k.ewm(com=2, adjust=False).mean()
    d["J"] = 3*k - 2*d_val

    delta = d["close"].diff().fillna(0)
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean().replace(0,eps)
    d["RSI"] = (100-(100/(1+gain/loss))).fillna(50)
    d["rsi_change"] = d["RSI"] - d["RSI"].shift(5)

    mid = d["close"].rolling(20).mean(); std = d["close"].rolling(20).std().fillna(0)
    d["BB_Pos"] = ((d["close"]-(mid-2*std))/(4*std+eps)).clip(0,1).fillna(0.5)
    d["bb_width"] = (((mid+2*std)-(mid-2*std))/(mid+eps)).fillna(0)

    tr = pd.concat([d["high"]-d["low"],(d["high"]-d["close"].shift(1)).abs(),
                    (d["low"]-d["close"].shift(1)).abs()],axis=1).max(axis=1)
    d["NATR"] = (tr.rolling(14).mean()/(d["close"]+eps)).fillna(0)
    d["volatility_ratio"] = d["NATR"]/(d["NATR"].rolling(20).mean()+eps)

    up = d["high"]-d["high"].shift(1); dn = d["low"].shift(1)-d["low"]
    pdm = pd.Series(np.where((up>dn)&(up>0),up,0),index=d.index)
    ndm = pd.Series(np.where((dn>up)&(dn>0),dn,0),index=d.index)
    tr14 = tr.rolling(14).sum().replace(0,eps)
    pdi = 100*pdm.rolling(14).sum()/tr14; ndi = 100*ndm.rolling(14).sum()/tr14
    d["ADX"] = (100*abs(pdi-ndi)/(pdi+ndi+eps)).rolling(14).mean().fillna(20)
    d["adx_change"] = d["ADX"]-d["ADX"].shift(5)

    tp = (d["high"]+d["low"]+d["close"])/3
    vwap = (d["volume"]*tp).cumsum()/(d["volume"].cumsum()+eps)
    d["VWAP_Dist"] = ((d["close"]-vwap)/(vwap+eps)).fillna(0)

    d["MA10"] = d["close"].rolling(10).mean().bfill()
    d["MA20"] = d["close"].rolling(20).mean().bfill()
    d["MA50"] = d["close"].rolling(50).mean().bfill()
    d["close_to_ma50"] = ((d["close"]-d["MA50"])/(d["MA50"]+eps)).fillna(0)
    d["MA_trend"] = np.sign(d["MA10"]-d["MA20"]).fillna(0)
    d["Macro_Trend"] = ((d["close"]-d["close"].ewm(span=100,adjust=False).mean())/
                        (d["close"].ewm(span=100,adjust=False).mean()+eps)).fillna(0)

    d["momentum_3"] = d["close"]-d["close"].shift(3)
    d["ROC_5"] = (d["close"]-d["close"].shift(5))/(d["close"].shift(5)+eps)*100

    d["volume_ratio"] = d["volume"]/(d["volume"].rolling(5).mean()+eps)
    d["VEV"] = d["volume_ratio"]/(d["NATR"]+eps)

    hl = (d["high"]-d["low"])+eps
    buy_raw = (d["close"]-d["low"])/hl*d["volume"]
    sell_raw = (d["high"]-d["close"])/hl*d["volume"]
    for w in [5,15,30]:
        d[f"BSP_{w}"] = np.log((buy_raw.rolling(w).sum()+eps)/(sell_raw.rolling(w).sum()+eps))

    hl_range = d["high"]-d["low"]+eps
    d["wick_upper_ratio"] = (d["high"]-d[["open","close"]].max(axis=1))/hl_range
    d["wick_lower_ratio"] = (d[["open","close"]].min(axis=1)-d["low"])/hl_range
    d["body_ratio"] = (d["close"]-d["open"]).abs()/hl_range

    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
    d["CCI"] = ((tp-tp_sma)/(0.015*tp_mad+eps)).fillna(0)

    atr14 = tr.rolling(14).sum()
    d["CHOP"] = (100*np.log10(atr14/(d["high"].rolling(14).max()-d["low"].rolling(14).min()+eps))/
                 np.log10(14)).fillna(50)

    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"]*obv_dir).cumsum()
    d["OBV_slope_5"] = obv.diff(5)/(obv.shift(5).abs()+eps)

    return d.replace([np.inf,-np.inf],np.nan).dropna()


def build_labels(
    df: pd.DataFrame,
    forward_bars: int = 1,
    min_move_pct: float = 0.0008,
    min_history: int = 50,
) -> pd.DataFrame:
    """
    构建方向标签，每个 entry point 只生成 1 条样本。

    Args:
        df: K 线 DataFrame
        forward_bars: 向前看几根 K 线（1=下一根，对应 15m expiry）
        min_move_pct: 最小价格变动，低于此阈值的样本被排除
        min_history: 最少历史 K 线数

    Returns:
        DataFrame with columns: entry_ts, entry_price, expiry_price,
        move_pct, direction, label_binary
    """
    close = df["close"].values
    if hasattr(df.index, "tz") and df.index.tz is not None:
        timestamps = np.array([int(t.timestamp()*1000) for t in df.index])
    else:
        timestamps = np.array([int(t.timestamp()*1000) for t in df.index])

    start = min_history
    end = len(df) - forward_bars
    if end <= start:
        return pd.DataFrame()

    rows = []
    for i in range(start, end):
        entry_price = float(close[i])
        expiry_price = float(close[i + forward_bars])
        move_pct = (expiry_price - entry_price) / entry_price

        if abs(move_pct) < min_move_pct:
            continue

        direction = 1 if move_pct > 0 else 0
        rows.append({
            "entry_ts": int(timestamps[i]),
            "entry_price": round(entry_price, 6),
            "expiry_price": round(expiry_price, 6),
            "move_pct": round(move_pct, 8),
            "direction": "CALL" if direction == 1 else "PUT",
            "label_binary": direction,
        })

    return pd.DataFrame(rows)


def align_features_with_labels(feat_df, labels_df, min_history=50):
    """对齐特征和标签，返回 X, y, weights, meta"""
    if feat_df.empty or labels_df.empty:
        return np.array([]), np.array([]), np.array([]), []

    X_list, y_list, w_list = [], [], []
    feat_index = feat_df.index
    feat_values = feat_df[FEATURES].values

    n = len(labels_df)
    time_weights = np.exp(np.linspace(-1.0, 0.0, n))

    for idx, (_, row) in enumerate(labels_df.iterrows()):
        entry_dt = pd.Timestamp(row["entry_ts"], unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)
        if feat_index.tz is not None:
            entry_dt = entry_dt.tz_localize("UTC")

        mask = feat_index <= entry_dt
        if not mask.any(): continue
        feat_idx = mask.sum() - 1
        if feat_idx < min_history: continue

        try:
            fv = feat_values[feat_idx]
            if np.any(np.isnan(fv)) or np.any(np.isinf(fv)): continue
            X_list.append(fv)
            y_list.append(row["label_binary"])
            w_list.append(time_weights[idx])
        except (IndexError, KeyError):
            continue

    if not X_list:
        return np.array([]), np.array([]), np.array([])
    return np.array(X_list), np.array(y_list), np.array(w_list)


def train_single_model(X_train, y_train, w_train, X_test, y_test, w_test,
                       symbol, expiry, output_dir):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = (y_train==1).sum(); n_neg = (y_train==0).sum()
    scale_pos_weight = n_neg/max(n_pos,1) if n_pos>0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        class_weight="balanced", random_state=42, verbosity=-1,
    )
    model.fit(X_train_s, y_train, sample_weight=w_train,
              eval_set=[(X_test_s, y_test)], eval_sample_weight=[w_test],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)
    pos_idx = 1 if hasattr(model,"classes_") and 1 in model.classes_ else 0
    proba = y_prob[:, pos_idx]

    acc = accuracy_score(y_test, y_pred)
    call_mask = y_pred==1; put_mask = y_pred==0
    call_wr = y_test[call_mask].mean() if call_mask.sum()>0 else 0.0
    put_wr = (1-y_test[put_mask]).mean() if put_mask.sum()>0 else 0.0

    # 概率分桶分析
    buckets = [0.50,0.52,0.54,0.56,0.58,0.60,0.65,0.70,1.0]
    edge_analysis = []
    for i in range(len(buckets)-1):
        lo, hi = buckets[i], buckets[i+1]
        mask = (proba>=lo) & (proba<hi)
        if mask.sum()>=5:
            actual = y_test[mask].mean()
            edge_analysis.append({"prob_range":f"{lo:.2f}-{hi:.2f}","n":int(mask.sum()),
                                  "pred_wr":float(proba[mask].mean()),"actual_wr":float(actual),
                                  "edge":float(actual-0.5)})

    model_name = f"{symbol.lower()}_{expiry}m_ensemble_v3.pkl"
    model_path = os.path.join(output_dir, model_name)
    joblib.dump({"ensemble":model,"scaler":scaler,"features":FEATURES,
                 "best_threshold":0.50,"label_version":"v3_direction_15m",
                 "expiry_minutes":expiry,"price_source":"gate_io"}, model_path)

    importances = dict(zip(FEATURES, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x:-x[1])[:8]

    return {"symbol":symbol,"expiry":expiry,"train_samples":len(X_train),
            "test_samples":len(X_test),"accuracy":round(acc,4),
            "call_win_rate":round(call_wr,4),"call_count":int(call_mask.sum()),
            "put_win_rate":round(put_wr,4),"put_count":int(put_mask.sum()),
            "brier_score":round(brier_score_loss(y_test,proba),4),
            "log_loss":round(log_loss(y_test,proba),4),
            "model_path":model_path,"model_size_kb":os.path.getsize(model_path)//1024,
            "top_features":top,"pos_ratio_train":round(n_pos/len(y_train),4),
            "pos_ratio_test":round((y_test==1).sum()/len(y_test),4),
            "edge_analysis":edge_analysis}


def main():
    parser = argparse.ArgumentParser(description="EventEdge V3 — 方向预测 + 噪音过滤 + 15m K线")
    parser.add_argument("--days",type=int,default=60)
    parser.add_argument("--expiries",type=str,default="15",
                        help="到期期限(分钟)，逗号分隔。15→预测1根15m, 30→预测2根, 60→预测4根")
    parser.add_argument("--min-move",type=float,default=0.0008,
                        help="最小价格变动 (e.g. 0.0008=0.08%%)")
    parser.add_argument("--output",type=str,default="./models")
    parser.add_argument("--symbols",type=str,default="")
    parser.add_argument("--test-split",type=float,default=0.20)
    parser.add_argument("--interval",type=str,default="15m",
                        help="K线间隔 (15m, 5m, 30m, 1h)")
    args = parser.parse_args()

    expiries = [int(x.strip()) for x in args.expiries.split(",")]
    symbols = args.symbols.split(",") if args.symbols else list(SYMBOLS.keys())
    symbols = [s.strip() for s in symbols if s.strip() and s.strip() in SYMBOLS]
    os.makedirs(args.output, exist_ok=True)

    interval_min = int(args.interval.replace("m","").replace("h",""))
    if "h" in args.interval:
        interval_min *= 60

    print(f"{'='*65}")
    print(f"  EventEdge V3 — 方向预测 + 噪音过滤")
    print(f"  K线: {args.interval} | Expiries: {[f'{e}m' for e in expiries]}")
    print(f"  Min Move: {args.min_move:.4%} | Symbols: {symbols}")
    print(f"  Features: {len(FEATURES)} | Days: {args.days}")
    print(f"{'='*65}\n")

    all_results = []
    for sym in symbols:
        pair = SYMBOLS[sym]
        print(f"\n{'─'*65}\n  {sym} ({pair})\n{'─'*65}")

        print(f"  拉取 {args.interval} K线...", end=" ", flush=True)
        df = fetch_klines(pair, interval=args.interval, days=args.days)
        if df is None or len(df)<100:
            print(f"数据不足 ({len(df) if df is not None else 0} 行)"); continue
        print(f"{len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

        print(f"  计算特征...", end=" ", flush=True)
        feat_df = calc_features(df)
        print(f"{len(feat_df)} 行")

        for expiry in expiries:
            # forward_bars: 15m expiry → 1 bar, 30m → 2 bars, 60m → 4 bars
            forward_bars = max(1, expiry // interval_min)

            print(f"\n  [{expiry}m] 构建标签 (forward={forward_bars} bars, min_move={args.min_move:.4%})...",
                  end=" ", flush=True)
            labels_df = build_labels(df, forward_bars=forward_bars,
                                     min_move_pct=args.min_move, min_history=50)
            if labels_df.empty:
                print("无样本"); continue

            n_call = (labels_df["label_binary"]==1).sum()
            n_put = (labels_df["label_binary"]==0).sum()
            n_total = len(labels_df)
            print(f"{n_total} 样本 (CALL↑={n_call}, PUT↓={n_put})")

            if n_total < 200:
                print(f"  样本不足 ({n_total}<200), 跳过"); continue

            X, y, w = align_features_with_labels(feat_df, labels_df, min_history=50)
            if len(X)==0:
                print("  对齐失败"); continue
            print(f"  对齐: {len(X)} 样本")

            split = int(len(X)*(1-args.test_split))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            w_tr, w_te = w[:split], w[split:]

            print(f"  训练集: {len(X_tr)} | 测试集: {len(X_te)}")
            print(f"  CALL占比: 训练={(y_tr==1).sum()/len(y_tr)*100:.1f}%  "
                  f"测试={(y_te==1).sum()/len(y_te)*100:.1f}%")

            result = train_single_model(X_tr,y_tr,w_tr, X_te,y_te,w_te,
                                        sym,expiry,args.output)
            all_results.append(result)

            print(f"\n  准确率: {result['accuracy']:.1%} | "
                  f"CALL胜率: {result['call_win_rate']:.1%} ({result['call_count']}) | "
                  f"PUT胜率: {result['put_win_rate']:.1%} ({result['put_count']})")
            print(f"  Brier: {result['brier_score']:.4f} | LogLoss: {result['log_loss']:.4f}")

            if result.get("edge_analysis"):
                print(f"\n  📊 概率校准:")
                print(f"  {'区间':<12} {'样本':>6} {'预测':>8} {'实际':>8} {'Edge':>8}")
                print(f"  {'─'*45}")
                for ea in result["edge_analysis"]:
                    print(f"  {ea['prob_range']:<12} {ea['n']:>6} {ea['pred_wr']:>7.1%} "
                          f"{ea['actual_wr']:>7.1%} {ea['edge']:>+8.1%}")

            print(f"\n  重要特征: {', '.join(f'{f}={v:.3f}' for f,v in result['top_features'])}")
            print(f"  模型: {result['model_path']} ({result['model_size_kb']}KB)")

    # ── 汇总 ──
    print(f"\n\n{'='*65}")
    print(f"  训练汇总 (V3: 方向预测 + 噪音过滤 + 15m K线)")
    print(f"{'='*65}")
    if all_results:
        print(f"\n  {'品种':<10} {'期限':>5} {'Train':>7} {'Test':>7} "
              f"{'Acc':>7} {'CALL':>7} {'PUT':>7} {'Brier':>7} {'BestEdge':>9}")
        print(f"  {'─'*72}")
        for r in sorted(all_results, key=lambda x:(x["symbol"],x["expiry"])):
            best_edge = max((ea["edge"] for ea in r.get("edge_analysis",[])), default=0)
            print(f"  {r['symbol']:<10} {str(r['expiry'])+'m':>5} {r['train_samples']:>7} {r['test_samples']:>7} "
                  f"{r['accuracy']:>7.1%} {r['call_win_rate']:>7.1%} {r['put_win_rate']:>7.1%} "
                  f"{r['brier_score']:>7.4f} {best_edge:>+9.1%}")
    else:
        print("  无训练结果")
    print(f"\n  模型目录: {os.path.abspath(args.output)}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()