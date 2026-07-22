#!/usr/bin/env python3
"""Quick check: 15m model probability distribution and edge pass rate"""
import sys, io, os, time, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from curl_cffi import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}
UNC_MARGIN = 0.005; CAL_MARGIN = 0.01; MIN_EDGE = 0.02; MIN_ROI = 0.005

FEATURES = [
    "hour_sin","hour_cos","dow_sin","dow_cos","MACD","macd_hist_change","RSI","rsi_change",
    "ROC_5","momentum_3","Macro_Trend","BB_Pos","bb_width","NATR","volatility_ratio",
    "ADX","adx_change","VWAP_Dist","close_to_ma50","MA_trend",
    "volume_ratio","VEV","BSP_5","BSP_15","BSP_30",
    "wick_upper_ratio","wick_lower_ratio","body_ratio","CCI","CHOP","OBV_slope_5","J",
]

def fetch_klines(pair, interval="15m", days=90):
    limit = 1000; all_rows, last_ts = [], int(time.time())
    for _ in range(10):
        if len(all_rows) >= days * 96: break
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval, "limit": limit, "to": last_ts,
            }, impersonate="chrome110", timeout=15, verify=False)
            if r.status_code != 200: time.sleep(2); continue
            data = r.json()
            if not data or len(data) < 2: break
            all_rows.extend(data); last_ts = int(data[0][0]) - 1
            if len(data) < limit: break
        except Exception: time.sleep(3)
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=["ts","qv","close","high","low","open","volume","final"])
    df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","close"])

def calc_features(df):
    eps = 1e-10; d = df.copy()
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

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    pair = SYMBOLS[sym]; payout = PAYOUTS[sym]; be_prob = 1.0/(1.0+payout)
    print(f"\n{'='*60}")
    print(f"  {sym} (payout={payout}, BE={be_prob:.1%})")
    print(f"{'='*60}")

    df = fetch_klines(pair, interval="15m", days=90)
    feat_df = calc_features(df)
    n = len(feat_df); split = int(n * 0.80)
    close_vals = feat_df["close"].values
    feat_arr = feat_df[FEATURES].values

    X_train_list = []; y_train_list = []
    for i in range(50, split - 1):
        fv = feat_arr[i]
        if np.any(np.isnan(fv)) or np.any(np.isinf(fv)): continue
        entry = close_vals[i]; expiry = close_vals[i+1]
        move = (expiry - entry) / entry
        if abs(move) < 0.0003: continue
        X_train_list.append(fv); y_train_list.append(1 if move>0 else 0)

    X_train = np.array(X_train_list); y_train = np.array(y_train_list)
    n_pos = (y_train==1).sum(); n_neg = (y_train==0).sum()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=n_neg/max(n_pos,1), class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    model.fit(X_train_s, y_train)

    X_test_list = []; test_info = []
    for i in range(split + 50, n - 1):
        fv = feat_arr[i]
        if np.any(np.isnan(fv)) or np.any(np.isinf(fv)): continue
        entry = close_vals[i]; expiry = close_vals[i+1]
        move = (expiry - entry) / entry
        X_test_list.append(fv)
        test_info.append({"pos":i, "move":move, "entry":entry, "expiry":expiry})

    X_test = np.array(X_test_list); X_test_s = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_s)
    pos_idx = 1 if 1 in model.classes_ else 0
    proba = y_prob[:, pos_idx]

    print(f"  Train: {len(X_train)} | Test: {len(proba)}")
    print(f"  Prob: mean={proba.mean():.4f} std={proba.std():.4f} min={proba.min():.4f} max={proba.max():.4f}")

    # Edge pass rate
    edge_pass = 0
    for p in proba:
        for dp in [p, 1.0-p]:
            cons = dp - UNC_MARGIN - CAL_MARGIN
            eff = cons - be_prob
            exp_roi = cons * payout - (1.0 - cons)
            if eff > MIN_EDGE and exp_roi > MIN_ROI:
                edge_pass += 1; break

    print(f"  Edge-passing samples: {edge_pass}/{len(proba)} ({100*edge_pass/len(proba):.1f}%)")

    # Probability buckets with actual win rate
    print(f"  Probability Buckets:")
    for lo, hi in [(0.50,0.52),(0.52,0.54),(0.54,0.56),(0.56,0.58),(0.58,0.60),(0.60,0.65),(0.65,0.70),(0.70,1.0)]:
        mask = (proba >= lo) & (proba < hi)
        n_b = mask.sum()
        if n_b >= 5:
            pred = proba[mask].mean()
            call_actual = np.array([1 if ti["move"] > 0 else 0 for ti in test_info])[mask]
            actual = call_actual.mean()
            tag = " OK" if actual > be_prob else " BELOW_BE"
            print(f"    [{lo:.2f}-{hi:.2f}]: n={n_b:>4} pred={pred:.1%} actual={actual:.1%}{tag}")

print(f"\n{'='*60}")
print(f"  PAYOUT_VERIFIED = false")
print(f"  ASSUMED_PAYOUT: BTC={PAYOUTS['BTCUSDT']}, ETH={PAYOUTS['ETHUSDT']}, SOL={PAYOUTS['SOLUSDT']}")
print(f"{'='*60}")