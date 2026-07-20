#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3 回测 — 多维度过滤: Regime + RSI极值 + 概率阈值

新增过滤:
1. RSI 极值过滤: RSI>70 不做CALL, RSI<30 不做PUT (防反转)
2. Regime 过滤: TREND_DOWN 不做CALL, TREND_UP 不做PUT (不逆势)
3. RSI 背离: 价格新高但RSI未新高 → 不做CALL (顶背离)
4. 概率阈值: 只在置信度>阈值时出手

用法:
    python scripts/backtest_v3.py --days 90 --symbols BTCUSDT,ETHUSDT
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
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"

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

# 特征名到索引的映射
FIDX = {name: i for i, name in enumerate(FEATURES)}

PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}


def fetch_klines(pair, interval="15m", days=90):
    limit = 1000
    all_rows, last_ts = [], int(time.time())
    for _ in range(10):
        if len(all_rows) >= days * 96: break
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200: time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1
            if len(data) < limit: break
        except Exception: time.sleep(3)
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=["ts","qv","close","high","low","open","volume","final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close"])


def calc_features(df):
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


def build_labels(df, forward_bars=1, min_move_pct=0.0005, min_history=50):
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
            "label_binary": direction,
        })
    return pd.DataFrame(rows)


def align(feat_df, labels_df, min_history=50):
    if feat_df.empty or labels_df.empty:
        return np.array([]), np.array([]), np.array([])
    X_list, y_list, idx_list = [], [], []
    feat_index = feat_df.index
    feat_values = feat_df[FEATURES].values
    for _, row in labels_df.iterrows():
        entry_dt = pd.Timestamp(row["entry_ts"], unit="ms")
        if entry_dt.tz is not None: entry_dt = entry_dt.tz_localize(None)
        if feat_index.tz is not None: entry_dt = entry_dt.tz_localize("UTC")
        mask = feat_index <= entry_dt
        if not mask.any(): continue
        feat_idx = mask.sum() - 1
        if feat_idx < min_history: continue
        try:
            fv = feat_values[feat_idx]
            if np.any(np.isnan(fv)) or np.any(np.isinf(fv)): continue
            X_list.append(fv)
            y_list.append(row["label_binary"])
            idx_list.append(feat_idx)
        except (IndexError, KeyError): continue
    if not X_list: return np.array([]), np.array([]), np.array([])
    return np.array(X_list), np.array(y_list), np.array(idx_list)


def classify_regime(row):
    """
    从特征向量推断市场状态。
    返回: (regime, details)
    """
    rsi = float(row[FIDX["RSI"]])
    adx = float(row[FIDX["ADX"]])
    bb_pos = float(row[FIDX["BB_Pos"]])
    bb_width = float(row[FIDX["bb_width"]])
    vol_ratio = float(row[FIDX["volatility_ratio"]])
    macro_trend = float(row[FIDX["Macro_Trend"]])
    ma_trend = float(row[FIDX["MA_trend"]])
    macd = float(row[FIDX["MACD"]])
    cci = float(row[FIDX["CCI"]])
    rsi_change = float(row[FIDX["rsi_change"]])

    details = {
        "rsi": round(rsi, 1),
        "adx": round(adx, 1),
        "bb_pos": round(bb_pos, 3),
        "bb_width": round(bb_width, 4),
        "vol_ratio": round(vol_ratio, 2),
        "macro_trend": round(macro_trend, 4),
        "ma_trend": int(ma_trend),
        "macd": round(macd, 4),
        "cci": round(cci, 1),
        "rsi_change": round(rsi_change, 1),
    }

    # ── 极端波动 ──
    if vol_ratio > 2.5:
        return "HIGH_VOL", details

    # ── 趋势市 ──
    if adx > 25:
        if macro_trend > 0.001 and ma_trend >= 0:
            strength = "STRONG" if adx > 35 else "NORMAL"
            return f"TREND_UP_{strength}", details
        elif macro_trend < -0.001 and ma_trend <= 0:
            strength = "STRONG" if adx > 35 else "NORMAL"
            return f"TREND_DOWN_{strength}", details
        else:
            return "TRENDING_NO_DIR", details

    # ── 震荡市 ──
    if bb_pos > 0.7:
        return "RANGE_HIGH", details
    elif bb_pos < 0.3:
        return "RANGE_LOW", details
    else:
        return "RANGE_MID", details


def check_reversal_risk(row):
    """
    检查反转风险。
    返回: (is_risky_call, is_risky_put, reason)
    """
    rsi = float(row[FIDX["RSI"]])
    rsi_change = float(row[FIDX["rsi_change"]])
    bb_pos = float(row[FIDX["BB_Pos"]])
    cci = float(row[FIDX["CCI"]])
    j_val = float(row[FIDX["J"]])
    wick_upper = float(row[FIDX["wick_upper_ratio"]])
    wick_lower = float(row[FIDX["wick_lower_ratio"]])
    body_ratio = float(row[FIDX["body_ratio"]])

    risky_call = False
    risky_put = False
    reasons = []

    # ── RSI 极值 ──
    if rsi > 70:
        risky_call = True
        reasons.append(f"RSI超买({rsi:.0f})")
    if rsi < 30:
        risky_put = True
        reasons.append(f"RSI超卖({rsi:.0f})")

    # ── RSI 背离（rsi_change 与价格方向相反）──
    if rsi_change < -2 and bb_pos > 0.7:
        risky_call = True
        reasons.append(f"RSI顶背离(rsi_chg={rsi_change:.1f})")
    if rsi_change > 2 and bb_pos < 0.3:
        risky_put = True
        reasons.append(f"RSI底背离(rsi_chg={rsi_change:.1f})")

    # ── CCI 极值 ──
    if cci > 150:
        risky_call = True
        reasons.append(f"CCI极高({cci:.0f})")
    if cci < -150:
        risky_put = True
        reasons.append(f"CCI极低({cci:.0f})")

    # ── 上影线 + 超买 = 顶部信号 ──
    if wick_upper > 0.6 and body_ratio < 0.3 and bb_pos > 0.7:
        risky_call = True
        reasons.append(f"长上影+超买(wick={wick_upper:.2f})")

    # ── 下影线 + 超卖 = 底部信号 ──
    if wick_lower > 0.6 and body_ratio < 0.3 and bb_pos < 0.3:
        risky_put = True
        reasons.append(f"长下影+超卖(wick={wick_lower:.2f})")

    # ── J 值极值 (KDJ) ──
    if j_val > 100:
        risky_call = True
        reasons.append(f"J值超买({j_val:.0f})")
    if j_val < 0:
        risky_put = True
        reasons.append(f"J值超卖({j_val:.0f})")

    return risky_call, risky_put, "; ".join(reasons)


def get_filter_configs():
    """返回多个过滤配置"""
    return [
        # (name, min_prob, use_regime, use_reversal, min_adx)
        ("无过滤(基准)", 0.50, False, False, 0),
        ("概率>52%", 0.52, False, False, 0),
        ("概率>52%+Regime", 0.52, True, False, 0),
        ("概率>52%+反转过滤", 0.52, False, True, 0),
        ("概率>52%+Regime+反转", 0.52, True, True, 0),
        ("概率>54%+Regime+反转", 0.54, True, True, 0),
    ]


def run_backtest(symbol, pair, days, test_split, forward_bars, min_move):
    print(f"\n{'='*65}")
    print(f"  {symbol} ({pair}) — 回测")
    print(f"{'='*65}")

    # 1. 数据
    print(f"  拉取数据...", end=" ", flush=True)
    df = fetch_klines(pair, interval="15m", days=days)
    if df is None or len(df) < 200:
        print(f"数据不足"); return None
    print(f"{len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 2. 特征
    print(f"  计算特征...", end=" ", flush=True)
    feat_df = calc_features(df)
    print(f"{len(feat_df)} 行")

    # 3. 标签
    print(f"  构建标签...", end=" ", flush=True)
    labels_df = build_labels(df, forward_bars=forward_bars, min_move_pct=min_move)
    if labels_df.empty:
        print("无样本"); return None
    n_up = (labels_df["label_binary"]==1).sum()
    n_down = (labels_df["label_binary"]==0).sum()
    print(f"{len(labels_df)} 样本 (↑{n_up} ↓{n_down})")

    # 4. 对齐
    X, y, idxs = align(feat_df, labels_df)
    if len(X)==0:
        print("  对齐失败"); return None
    print(f"  对齐: {len(X)} 样本")

    # 5. 时间序列切分
    split = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    label_split = int(len(labels_df) * (1 - test_split))
    test_labels_df = labels_df.iloc[label_split:].reset_index(drop=True)

    print(f"  训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 6. 训练
    print(f"  训练 LightGBM...", end=" ", flush=True)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = (y_train==1).sum(); n_neg = (y_train==0).sum()
    scale_pos_weight = n_neg/max(n_pos,1)

    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        class_weight="balanced", random_state=42, verbosity=-1,
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              eval_metric="auc",
              callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])
    print("done")

    # 7. 预测
    y_prob = model.predict_proba(X_test_s)
    pos_idx = 1 if hasattr(model,"classes_") and 1 in model.classes_ else 0
    proba = y_prob[:, pos_idx]

    payout = PAYOUTS.get(symbol, 0.80)
    be_prob = 1.0 / (1.0 + payout)

    n_test = len(X_test)
    if len(test_labels_df) > n_test:
        test_labels_df = test_labels_df.iloc[:n_test]

    # ── 预计算每个样本的 regime 和反转风险 ──
    regimes = []
    reversal_risks = []
    for i in range(n_test):
        row = X_test[i]
        regime, details = classify_regime(row)
        regimes.append(regime)
        risky_call, risky_put, reason = check_reversal_risk(row)
        reversal_risks.append((risky_call, risky_put, reason))

    # ── 按 regime 统计实际胜率 ──
    regime_stats = {}
    for i in range(n_test):
        reg = regimes[i]
        if reg not in regime_stats:
            regime_stats[reg] = {"total": 0, "call_up": 0, "call_total": 0}
        regime_stats[reg]["total"] += 1
        if y_test[i] == 1:
            regime_stats[reg]["call_up"] += 1
        regime_stats[reg]["call_total"] += 1

    print(f"\n  📊 市场状态分布:")
    for reg in sorted(regime_stats.keys()):
        s = regime_stats[reg]
        wr = s["call_up"]/s["call_total"]*100 if s["call_total"] > 0 else 0
        print(f"    {reg:<20}: {s['total']:>5} 样本, CALL胜率={wr:.1f}%")

    # ── 8. 多配置回测 ──
    print(f"\n{'='*65}")
    print(f"  回测对比 — 不同过滤配置")
    print(f"  盈亏平衡胜率: {be_prob:.1%} | 赔付率: {payout} | 固定下注 3U")
    print(f"{'='*65}")

    filter_configs = get_filter_configs()
    all_config_results = []

    for cfg_name, min_prob, use_regime, use_reversal, min_adx in filter_configs:
        trades = []
        rejected = {"regime": 0, "reversal": 0, "prob": 0}

        for i in range(n_test):
            p = float(proba[i])
            true_label = int(y_test[i])
            entry_price = float(test_labels_df.iloc[i]["entry_price"]) if i < len(test_labels_df) else 0
            expiry_price = float(test_labels_df.iloc[i]["expiry_price"]) if i < len(test_labels_df) else 0
            true_move = (expiry_price - entry_price) / entry_price if entry_price > 0 else 0

            # 决定方向
            if p >= min_prob:
                direction = "CALL"
                pred_label = 1
            elif p <= (1 - min_prob):
                direction = "PUT"
                pred_label = 0
            else:
                rejected["prob"] += 1
                continue

            # ── Regime 过滤 ──
            regime = regimes[i]
            if use_regime:
                if "DOWN" in regime and direction == "CALL":
                    rejected["regime"] += 1
                    continue
                if "UP" in regime and direction == "PUT":
                    rejected["regime"] += 1
                    continue
                if "HIGH_VOL" in regime:
                    rejected["regime"] += 1
                    continue

            # ── 反转风险过滤 ──
            if use_reversal:
                risky_call, risky_put, reason = reversal_risks[i]
                if direction == "CALL" and risky_call:
                    rejected["reversal"] += 1
                    continue
                if direction == "PUT" and risky_put:
                    rejected["reversal"] += 1
                    continue

            # 结算
            is_win = (pred_label == true_label)
            edge = (p if direction == "CALL" else 1-p) - be_prob
            pnl = 3 * payout if is_win else -3

            trades.append({
                "direction": direction,
                "prob": p if direction == "CALL" else 1-p,
                "is_win": is_win,
                "pnl": pnl,
                "edge": edge,
                "regime": regime,
                "true_move": true_move,
            })

        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t["is_win"])
        wr = n_wins / n_trades if n_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        total_staked = n_trades * 3
        roi = total_pnl / total_staked if total_staked > 0 else 0
        avg_edge = sum(t["edge"] for t in trades) / n_trades if n_trades > 0 else 0

        # 连败
        max_lose = 0; cur = 0
        for t in trades:
            if not t["is_win"]: cur += 1; max_lose = max(max_lose, cur)
            else: cur = 0

        # 按方向
        call_t = [t for t in trades if t["direction"]=="CALL"]
        put_t = [t for t in trades if t["direction"]=="PUT"]
        call_wr = sum(1 for t in call_t if t["is_win"])/len(call_t) if call_t else 0
        put_wr = sum(1 for t in put_t if t["is_win"])/len(put_t) if put_t else 0

        # 按 regime
        regime_trades = {}
        for t in trades:
            reg = t["regime"]
            if reg not in regime_trades:
                regime_trades[reg] = {"wins": 0, "total": 0, "pnl": 0}
            regime_trades[reg]["total"] += 1
            if t["is_win"]: regime_trades[reg]["wins"] += 1
            regime_trades[reg]["pnl"] += t["pnl"]

        config_result = {
            "name": cfg_name,
            "trades": n_trades,
            "wins": n_wins,
            "win_rate": wr,
            "call_wr": call_wr,
            "call_count": len(call_t),
            "put_wr": put_wr,
            "put_count": len(put_t),
            "pnl": total_pnl,
            "roi": roi,
            "avg_edge": avg_edge,
            "max_lose_streak": max_lose,
            "rejected": rejected,
            "regime_trades": regime_trades,
        }
        all_config_results.append(config_result)

    # ── 打印对比表 ──
    print(f"\n  {'配置':<28} {'交易':>6} {'胜率':>8} {'CALL':>8} {'PUT':>8} "
          f"{'PnL':>9} {'ROI':>8} {'连败':>5} {'剔除':>20} {'判定':>10}")
    print(f"  {'─'*115}")

    best_valid = None
    for cr in all_config_results:
        name = cr["name"]
        nt = cr["trades"]
        wr = cr["win_rate"]
        cwr = cr["call_wr"]
        pwr = cr["put_wr"]
        pnl = cr["pnl"]
        roi = cr["roi"]
        streak = cr["max_lose_streak"]
        rej = cr["rejected"]
        rej_str = f"R={rej['regime']} V={rej['reversal']} P={rej['prob']}"

        if nt == 0:
            verdict = "无交易"
        elif wr > be_prob:
            verdict = "✅ 盈利"
            if pnl > 0 and (best_valid is None or (pnl > 0 and cr["trades"] > best_valid.get("trades", 0))):
                best_valid = cr
        elif wr > be_prob - 0.02:
            verdict = "≈ 持平"
        else:
            verdict = "❌ 亏损"

        print(f"  {name:<28} {nt:>6} {wr:>7.1%} "
              f"{cwr:>7.1%} {pwr:>7.1%} "
              f"{pnl:>+9.1f} {roi:>+7.1%} {streak:>5} {rej_str:>20} {verdict:>10}")

    # ── 分 regime 详情 ──
    if best_valid and best_valid.get("regime_trades"):
        print(f"\n  📊 最佳配置 [{best_valid['name']}] 按市场状态:")
        print(f"  {'状态':<22} {'交易':>6} {'胜率':>8} {'PnL':>9}")
        print(f"  {'─'*48}")
        for reg in sorted(best_valid["regime_trades"].keys()):
            rt = best_valid["regime_trades"][reg]
            rwr = rt["wins"]/rt["total"] if rt["total"]>0 else 0
            print(f"  {reg:<22} {rt['total']:>6} {rwr:>7.1%} {rt['pnl']:>+9.1f}")

    return {
        "symbol": symbol,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "be_prob": be_prob,
        "payout": payout,
        "config_results": all_config_results,
        "regime_stats": regime_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="V3 回测 — 多维度过滤")
    parser.add_argument("--days",type=int,default=90)
    parser.add_argument("--symbols",type=str,default="BTCUSDT,ETHUSDT")
    parser.add_argument("--test-split",type=float,default=0.20)
    parser.add_argument("--min-move",type=float,default=0.0005)
    parser.add_argument("--expiries",type=str,default="15")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    expiries = [int(x.strip()) for x in args.expiries.split(",")]

    print(f"{'='*65}")
    print(f"  V3 回测 — 多维度过滤 (Regime + RSI反转 + 概率)")
    print(f"  Symbols: {symbols} | Expiries: {expiries} | Days: {args.days}")
    print(f"{'='*65}")

    all_results = []
    for sym in symbols:
        pair = SYMBOLS.get(sym)
        if not pair: continue
        for expiry in expiries:
            forward_bars = max(1, expiry // 15)
            result = run_backtest(sym, pair, args.days, args.test_split,
                                  forward_bars, args.min_move)
            if result:
                all_results.append(result)

    # ── 最终汇总 ──
    print(f"\n\n{'='*65}")
    print(f"  最终汇总 — 各品种最优配置")
    print(f"{'='*65}")
    for r in all_results:
        sym = r["symbol"]
        be = r["be_prob"]
        print(f"\n  {sym} (盈亏平衡={be:.1%}):")
        # 找最佳配置
        best = None
        for cr in r["config_results"]:
            if cr["pnl"] > 0 and cr["trades"] > 10:
                if best is None or cr["pnl"] > best["pnl"]:
                    best = cr
        if best:
            print(f"    推荐: {best['name']}")
            print(f"    交易{best['trades']}笔 胜率{best['win_rate']:.1%} "
                  f"PnL {best['pnl']:+.1f}U ROI {best['roi']:+.1%} 连败{best['max_lose_streak']}")
        else:
            print(f"    ⚠ 无盈利配置，需要继续优化")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()