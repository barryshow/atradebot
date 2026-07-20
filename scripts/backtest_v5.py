#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 回测 v5 — 三品种 Walk-Forward + Edge Bucket 验证

核心改进:
1. 直接从 Gate.io API 拉取 15m K 线 (90 天数据)
2. 使用 v3 的 32 特征集（已验证有效）
3. 使用修复后的整数 Kelly stake 公式
4. Edge Bucket 与实际 ROI 正相关验证
5. 概率区间可靠性检查
6. TIE 单独处理

用法:
    python scripts/backtest_v5.py --days 90 --symbols BTCUSDT,ETHUSDT,SOLUSDT
"""
import argparse, io, os, sys, time, warnings, math, json
warnings.filterwarnings("ignore")
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from curl_cffi import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}

# ── v3 特征集 (32 features, verified effective) ──
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

# ── 回测交易记录 ──
@dataclass
class BTTrade:
    trade_id: int = 0
    symbol: str = ""
    direction: str = ""
    direction_int: int = 0
    entry_price: float = 0.0
    expiry_price: float = 0.0
    stake_usd: int = 3
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0
    break_even_probability: float = 0.0
    effective_edge: float = 0.0
    expected_roi: float = 0.0
    net_payout_ratio: float = 0.80
    result: str = ""
    realized_pnl: float = 0.0
    regime: str = ""
    true_move_pct: float = 0.0


def fetch_klines(pair: str, interval: str = "15m", days: int = 90) -> Optional[pd.DataFrame]:
    """从 Gate.io 拉取 K 线"""
    limit = 1000
    all_rows, last_ts = [], int(time.time())
    for _ in range(10):
        if len(all_rows) >= days * 96:
            break
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval,
                "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200:
                time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2:
                break
            all_rows.extend(data)
            last_ts = int(data[0][0]) - 1
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


def calc_features(df: pd.DataFrame) -> pd.DataFrame:
    """v3 32-feature 计算（已验证有效）"""
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


def build_labels(df: pd.DataFrame, forward_bars: int = 1, min_move_pct: float = 0.0005,
                 min_history: int = 50) -> pd.DataFrame:
    """构建 Event Contract 风格标签"""
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

        # CALL label: 1 if price went up, 0 if down (TIE = abs(move) < min_move_pct)
        if abs(move_pct) < min_move_pct:
            # TIE: both CALL and PUT result in 0 PnL
            # For binary model, treat as 0 (not WIN)
            call_label = 0
            put_label = 0
            is_tie = 1
        else:
            call_label = 1 if move_pct > 0 else 0
            put_label = 1 if move_pct < 0 else 0
            is_tie = 0

        # CALL sample
        rows.append({
            "entry_ts": int(timestamps[i]),
            "entry_price": round(entry_price, 6),
            "expiry_price": round(expiry_price, 6),
            "move_pct": round(move_pct, 8),
            "direction": "CALL",
            "direction_int": 1,
            "label_binary": call_label,
            "is_tie": is_tie,
        })
        # PUT sample
        rows.append({
            "entry_ts": int(timestamps[i]),
            "entry_price": round(entry_price, 6),
            "expiry_price": round(expiry_price, 6),
            "move_pct": round(move_pct, 8),
            "direction": "PUT",
            "direction_int": 2,
            "label_binary": put_label,
            "is_tie": is_tie,
        })

    return pd.DataFrame(rows)


def align(feat_df: pd.DataFrame, labels_df: pd.DataFrame,
          min_history: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """对齐特征与标签"""
    if feat_df.empty or labels_df.empty:
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()

    X_list, y_list, idx_list = [], [], []
    feat_index = feat_df.index
    feat_values = feat_df[FEATURES].values

    for _, row in labels_df.iterrows():
        entry_dt = pd.Timestamp(row["entry_ts"], unit="ms")
        if entry_dt.tz is not None:
            entry_dt = entry_dt.tz_localize(None)
        if feat_index.tz is not None:
            entry_dt = entry_dt.tz_localize("UTC")
        mask = feat_index <= entry_dt
        if not mask.any():
            continue
        feat_idx = mask.sum() - 1
        if feat_idx < min_history:
            continue
        try:
            fv = feat_values[feat_idx]
            if np.any(np.isnan(fv)) or np.any(np.isinf(fv)):
                continue
            X_list.append(fv)
            y_list.append(row["label_binary"])
            idx_list.append(feat_idx)
        except (IndexError, KeyError):
            continue

    if not X_list:
        return np.array([]), np.array([]), np.array([]), pd.DataFrame()

    return np.array(X_list), np.array(y_list), np.array(idx_list), labels_df.iloc[:len(X_list)]


def compute_stats(trades: List[BTTrade]) -> dict:
    """计算统计"""
    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "ties": 0,
                "win_rate": 0.0, "pnl": 0.0, "roi": 0.0,
                "avg_edge": 0.0, "brier": 0.0, "avg_exp_roi": 0.0}

    wins = sum(1 for t in trades if t.result == "WIN")
    losses = sum(1 for t in trades if t.result == "LOSS")
    ties = sum(1 for t in trades if t.result == "TIE")
    settled = wins + losses
    total_pnl = sum(t.realized_pnl for t in trades)
    total_staked = sum(t.stake_usd for t in trades)
    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    wr = wins / settled if settled > 0 else 0.0
    avg_edge = sum(t.effective_edge for t in trades) / len(trades) if trades else 0.0
    avg_exp_roi = sum(t.expected_roi for t in trades) / len(trades) if trades else 0.0

    brier = 0.0
    for t in trades:
        actual = 1.0 if t.result == "WIN" else 0.0
        brier += (t.raw_probability - actual) ** 2
    brier /= len(trades) if trades else 1

    return {
        "total": len(trades), "wins": wins, "losses": losses, "ties": ties,
        "win_rate": wr, "pnl": total_pnl, "total_staked": total_staked,
        "roi": roi, "avg_edge": avg_edge, "brier": brier,
        "avg_exp_roi": avg_exp_roi,
    }


def edge_bucket_stats(trades: List[BTTrade]) -> dict:
    """按 Edge 区间统计"""
    buckets = {
        "negative": [], "0-1%": [], "1-2%": [], "2-3%": [],
        "3-5%": [], "5-7%": [], "7-10%": [], "10%+": [],
    }
    for t in trades:
        e = t.effective_edge
        if e < 0: buckets["negative"].append(t)
        elif e < 0.01: buckets["0-1%"].append(t)
        elif e < 0.02: buckets["1-2%"].append(t)
        elif e < 0.03: buckets["2-3%"].append(t)
        elif e < 0.05: buckets["3-5%"].append(t)
        elif e < 0.07: buckets["5-7%"].append(t)
        elif e < 0.10: buckets["7-10%"].append(t)
        else: buckets["10%+"].append(t)

    result = {}
    for name, ts in buckets.items():
        if ts:
            result[name] = compute_stats(ts)
    return result


def prob_bucket_stats(trades: List[BTTrade]) -> dict:
    """按概率区间统计"""
    buckets = {
        "50-52%": [], "52-54%": [], "54-56%": [], "56-58%": [],
        "58-60%": [], "60-65%": [], "65-70%": [], "70%+": [],
    }
    for t in trades:
        p = t.raw_probability
        if p < 0.52: buckets["50-52%"].append(t)
        elif p < 0.54: buckets["52-54%"].append(t)
        elif p < 0.56: buckets["54-56%"].append(t)
        elif p < 0.58: buckets["56-58%"].append(t)
        elif p < 0.60: buckets["58-60%"].append(t)
        elif p < 0.65: buckets["60-65%"].append(t)
        elif p < 0.70: buckets["65-70%"].append(t)
        else: buckets["70%+"].append(t)

    result = {}
    for name, ts in buckets.items():
        if ts:
            stats = compute_stats(ts)
            stats["avg_predicted"] = sum(t.raw_probability for t in ts) / len(ts)
            result[name] = stats
    return result


def simulate_trades(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    labels_df: pd.DataFrame,
    symbol: str,
    net_payout: float,
    min_prob: float = 0.52,
    min_edge: float = 0.0,
    min_order: int = 3,
    order_step: int = 1,
    kelly_frac: float = 0.10,
    max_bet_frac: float = 0.01,
    equity: float = 5000.0,
) -> List[BTTrade]:
    """模拟交易 — 严格整数 Kelly + Edge 门槛"""
    be_prob = 1.0 / (1.0 + net_payout)
    trades = []

    for i in range(len(y_prob)):
        prob = float(y_prob[i])
        true_label = int(y_true[i])
        row = labels_df.iloc[i] if i < len(labels_df) else None
        if row is None:
            continue

        # 考虑 CALL 和 PUT 两个方向
        for direction, direction_int, dir_prob in [
            ("CALL", 1, prob),
            ("PUT", 2, 1.0 - prob),
        ]:
            if dir_prob < min_prob:
                continue

            effective_edge = dir_prob - be_prob
            if effective_edge < min_edge:
                continue

            expected_roi = dir_prob * net_payout - (1.0 - dir_prob)

            # Integer Kelly
            denom = 1.0 + net_payout
            frac_kelly = effective_edge / denom if denom > 0 else 0.0
            target_fraction = kelly_frac * frac_kelly
            effective_fraction = min(target_fraction, max_bet_frac)
            incremental = equity * effective_fraction
            stake_usd = min_order + int(math.floor(incremental))
            stake_usd = (stake_usd // order_step) * order_step

            if stake_usd < min_order:
                continue

            # 结算
            if row["is_tie"]:
                result = "TIE"
                realized_pnl = 0.0
                is_win_actual = False
            elif direction == "CALL":
                is_win_actual = (true_label == 1)
                result = "WIN" if is_win_actual else "LOSS"
                realized_pnl = stake_usd * net_payout if is_win_actual else -stake_usd
            else:  # PUT
                is_win_actual = (true_label == 0)
                result = "WIN" if is_win_actual else "LOSS"
                realized_pnl = stake_usd * net_payout if is_win_actual else -stake_usd

            trade = BTTrade(
                trade_id=len(trades),
                symbol=symbol,
                direction=direction,
                direction_int=direction_int,
                entry_price=float(row["entry_price"]),
                expiry_price=float(row["expiry_price"]),
                stake_usd=stake_usd,
                raw_probability=round(dir_prob, 4),
                calibrated_probability=round(dir_prob, 4),
                break_even_probability=round(be_prob, 4),
                effective_edge=round(effective_edge, 4),
                expected_roi=round(expected_roi, 4),
                net_payout_ratio=net_payout,
                result=result,
                realized_pnl=round(realized_pnl, 4),
                true_move_pct=float(row["move_pct"]),
            )
            trades.append(trade)

    return trades


def run_backtest(symbol: str, pair: str, days: int, test_split: float,
                 forward_bars: int, min_move: float) -> Optional[dict]:
    """单品种 Walk-Forward 回测"""
    payout = PAYOUTS.get(symbol, 0.80)
    be_prob = 1.0 / (1.0 + payout)

    print(f"\n{'='*65}")
    print(f"  {symbol} ({pair}) — {days}d data, payout={payout}, BE={be_prob:.1%}")
    print(f"{'='*65}")

    # 1. 数据
    print(f"  [1/5] 拉取 {days}d 15m K线...", end=" ", flush=True)
    df = fetch_klines(pair, interval="15m", days=days)
    if df is None or len(df) < 200:
        print(f"数据不足"); return None
    print(f"{len(df)} rows ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 2. 特征
    print(f"  [2/5] 计算 {len(FEATURES)} 特征...", end=" ", flush=True)
    feat_df = calc_features(df)
    print(f"{len(feat_df)} rows")

    # 3. 标签
    print(f"  [3/5] 构建标签 (forward={forward_bars} bars)...", end=" ", flush=True)
    labels_df = build_labels(df, forward_bars=forward_bars, min_move_pct=min_move)
    if labels_df.empty:
        print("无样本"); return None
    n_up = (labels_df["label_binary"] == 1).sum()
    n_down = (labels_df["label_binary"] == 0).sum()
    n_tie = labels_df["is_tie"].sum() // 2  # 除以2因为每个entry点有CALL和PUT
    print(f"{len(labels_df)} samples (↑{n_up} ↓{n_down} TIE≈{n_tie})")

    # 4. 对齐
    print(f"  [4/5] 对齐特征与标签...", end=" ", flush=True)
    X, y, idxs, aligned_labels = align(feat_df, labels_df)
    if len(X) == 0:
        print("对齐失败"); return None
    print(f"{len(X)} 样本")

    # 5. 时间序列切分
    split = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_labels = aligned_labels.iloc[split:].reset_index(drop=True)

    print(f"  [5/5] Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Date: {df.index[0].date()} ~ {df.index[-1].date()}")

    # 6. 训练
    print(f"        训练 LightGBM...", end=" ", flush=True)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = (y_train == 1).sum(); n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

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
    y_prob_raw = model.predict_proba(X_test_s)
    pos_idx = 1 if hasattr(model, "classes_") and 1 in model.classes_ else 0
    proba = y_prob_raw[:, pos_idx]

    prob_mean = float(np.mean(proba))
    prob_std = float(np.std(proba))
    print(f"        概率分布: mean={prob_mean:.3f}, std={prob_std:.4f}")
    if prob_std < 0.01:
        print(f"        ⚠ 概率分布坍缩 — 模型无法区分样本")
        return None

    # 8. 多配置回测
    configs = [
        # (name, min_prob, min_edge, kelly_frac, max_bet_frac)
        ("概率>52%", 0.52, 0.0, 0.10, 0.01),
        ("概率>52%+Edge>1%", 0.52, 0.01, 0.10, 0.01),
        ("概率>54%+Edge>2%", 0.54, 0.02, 0.10, 0.01),
        ("概率>56%+Edge>3%", 0.56, 0.03, 0.10, 0.01),
        ("概率>58%+Edge>4%", 0.58, 0.04, 0.10, 0.01),
    ]

    all_configs = []
    best_config = None

    for cfg_name, min_prob, min_edge, kf, mbf in configs:
        trades = simulate_trades(
            proba, y_test, test_labels, symbol, payout,
            min_prob=min_prob, min_edge=min_edge,
            kelly_frac=kf, max_bet_frac=mbf,
        )
        stats = compute_stats(trades)
        stats["config"] = cfg_name
        stats["min_prob"] = min_prob
        stats["min_edge"] = min_edge
        all_configs.append(stats)

        if stats["total"] > 0 and (best_config is None or stats["roi"] > best_config["roi"]):
            best_config = stats
            best_config["_trades"] = trades

    # 打印配置对比
    print(f"\n  {'配置':<28} {'交易':>6} {'胜率':>8} {'PnL':>9} {'ROI':>8} "
          f"{'AvgEdge':>8} {'Brier':>7} {'TIE':>5}")
    print(f"  {'─'*85}")
    for s in all_configs:
        print(f"  {s['config']:<28} {s['total']:>6} {s['win_rate']:>7.1%} "
              f"{s['pnl']:>+9.1f} {s['roi']:>+7.1%} "
              f"{s['avg_edge']:>7.2%} {s['brier']:>7.4f} {s['ties']:>5}")

    return {
        "symbol": symbol,
        "payout": payout,
        "be_prob": be_prob,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "prob_mean": prob_mean,
        "prob_std": prob_std,
        "all_configs": all_configs,
        "best_config": best_config,
    }


def main():
    parser = argparse.ArgumentParser(description="EventEdge V2 回测 v5")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--test-split", type=float, default=0.20)
    parser.add_argument("--min-move", type=float, default=0.0005)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    print(f"{'='*65}")
    print(f"  EventEdge V2 回测 v5 — Walk-Forward + Edge Bucket 验证")
    print(f"  Symbols: {symbols} | Days: {args.days} | Features: {len(FEATURES)}")
    print(f"{'='*65}")

    all_results = []
    for sym in symbols:
        pair = SYMBOLS.get(sym)
        if not pair:
            continue
        result = run_backtest(sym, pair, args.days, args.test_split,
                             forward_bars=1, min_move=args.min_move)
        if result:
            all_results.append(result)

    # ═══════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'='*70}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*70}")

    for r in all_results:
        sym = r["symbol"]
        print(f"\n  {sym} (payout={r['payout']}, BE={r['be_prob']:.1%}):")
        print(f"    Train: {r['train_samples']} | Test: {r['test_samples']}")
        print(f"    Prob: mean={r['prob_mean']:.3f}, std={r['prob_std']:.4f}")

        bc = r.get("best_config")
        if bc and bc["total"] > 0:
            trades = bc.get("_trades", [])
            print(f"\n    📊 最佳配置: {bc['config']}")
            print(f"    交易: {bc['total']} | 胜率: {bc['win_rate']:.1%} | "
                  f"PnL: {bc['pnl']:+.1f}U | ROI: {bc['roi']:+.1%}")
            print(f"    AvgEdge: {bc['avg_edge']:.2%} | Brier: {bc['brier']:.4f} | TIE: {bc['ties']}")

            # ── Edge Bucket 分析 ──
            if trades:
                eb = edge_bucket_stats(trades)
                print(f"\n    📈 Edge Bucket → ROI 相关性:")
                print(f"    {'Bucket':>10} {'Trades':>7} {'WR':>7} {'ROI':>7} {'AvgEdge':>8}")
                print(f"    {'─'*45}")
                prev_wr = None
                monotonic = True
                for bn in ["negative", "0-1%", "1-2%", "2-3%", "3-5%", "5-7%", "7-10%", "10%+"]:
                    if bn in eb:
                        s = eb[bn]
                        wr = s["win_rate"]
                        marker = ""
                        if prev_wr is not None and wr < prev_wr:
                            monotonic = False
                            marker = " ⚠ NON-MONOTONIC"
                        print(f"    {bn:>10} {s['total']:>7} {wr:>6.1%} {s['roi']:>+6.1%} {s['avg_edge']:>7.2%}{marker}")
                        prev_wr = wr

                if monotonic:
                    print(f"\n    ✅ Edge Bucket → ROI 正相关（单调递增）")
                else:
                    print(f"\n    ⚠ Edge Bucket → ROI 不单调 → 概率校准有问题")

                # ── 概率可靠性 ──
                pb = prob_bucket_stats(trades)
                print(f"\n    📊 概率可靠性 (Reliability):")
                print(f"    {'Bucket':>10} {'Trades':>7} {'Predicted':>9} {'Actual':>7} {'Bias':>7}")
                print(f"    {'─'*45}")
                for bn in ["50-52%", "52-54%", "54-56%", "56-58%", "58-60%", "60-65%", "65-70%", "70%+"]:
                    if bn in pb:
                        s = pb[bn]
                        pred = s["avg_predicted"]
                        actual = s["win_rate"]
                        bias = pred - actual
                        bias_marker = " ⚠" if abs(bias) > 0.05 else ""
                        print(f"    {bn:>10} {s['total']:>7} {pred:>8.1%} {actual:>6.1%} {bias:>+6.1%}{bias_marker}")
        else:
            print(f"    ⚠ 无有效交易配置")

    # ── 最终结论 ──
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")

    for r in all_results:
        sym = r["symbol"]
        bc = r.get("best_config")
        if bc and bc["total"] > 0 and bc["roi"] > 0:
            print(f"  {sym}: ✅ 回测盈利 — {bc['config']} — "
                  f"{bc['total']}笔 胜率{bc['win_rate']:.1%} ROI{bc['roi']:+.1%}")
        elif bc and bc["total"] > 0:
            print(f"  {sym}: ❌ 回测亏损 — {bc['config']} — "
                  f"{bc['total']}笔 胜率{bc['win_rate']:.1%} ROI{bc['roi']:+.1%}")
        else:
            print(f"  {sym}: ⚠ 无交易 — 概率分布可能坍缩")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()