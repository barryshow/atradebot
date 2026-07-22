#!/usr/bin/env python3
"""ATradeBot Trading Backtest — Full Pipeline with 15m model + constraints"""
import sys, io, os, time, warnings, math
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from curl_cffi import requests
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}
MAX_HOURLY = 4; MAX_ACTIVE = 3; MAX_PER_SYM = 1; COOLDOWN = 60
HOLD_MIN = 15; MIN_EDGE = 0.02; MIN_ROI = 0.005
UNC_MARGIN = 0.005; CAL_MARGIN = 0.01
KELLY_F = 0.50; MAX_BET_F = 0.35; MIN_ORDER = 3; SIM_EQUITY = 5000

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

def stats(trades):
    if not trades: return {"trades":0,"wins":0,"losses":0,"ties":0,"wr":0,"pnl":0,"staked":0,"roi":0}
    W = sum(1 for t in trades if t["result"]=="WIN")
    L = sum(1 for t in trades if t["result"]=="LOSS")
    T = sum(1 for t in trades if t["result"]=="TIE")
    S = W+L
    pnl = sum(t["pnl"] for t in trades)
    stk = sum(t["stake"] for t in trades)
    return {"trades":len(trades),"wins":W,"losses":L,"ties":T,
        "wr":W/S if S>0 else 0,"pnl":round(pnl,2),"staked":stk,
        "roi":pnl/stk if stk>0 else 0}

print("=" * 65)
print("  ATradeBot TRADING BACKTEST — Full Pipeline")
print(f"  MAX_HOURLY={MAX_HOURLY}, MAX_ACTIVE={MAX_ACTIVE}, MAX_PER_SYM={MAX_PER_SYM}")
print(f"  HOLD={HOLD_MIN}min, COOLDOWN={COOLDOWN}s, Edge>={MIN_EDGE}, ROI>={MIN_ROI}")
print("=" * 65)

grand_total = []; all_details = []

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    pair = SYMBOLS[sym]; payout = PAYOUTS[sym]; be_prob = 1.0/(1.0+payout)
    print(f"\n{'─'*65}")
    print(f"  {sym} (payout={payout}, BE={be_prob:.1%})")
    print(f"{'─'*65}")

    df = fetch_klines(pair, interval="15m", days=90)
    feat_df = calc_features(df)
    n = len(feat_df); split = int(n * 0.80)
    close_vals = feat_df["close"].values
    feat_arr = feat_df[FEATURES].values

    # Train labels (skip ties)
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
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train)
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        min_child_samples=30, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=n_neg/max(n_pos,1), class_weight="balanced",
        random_state=42, verbosity=-1,
    )
    model.fit(X_train_s, y_train)

    # Test data
    X_test_list = []; test_info = []
    for i in range(split + 50, n - 1):
        fv = feat_arr[i]
        if np.any(np.isnan(fv)) or np.any(np.isinf(fv)): continue
        X_test_list.append(fv)
        test_info.append({"pos":i, "move":(close_vals[i+1]-close_vals[i])/close_vals[i],
                         "entry":close_vals[i], "expiry":close_vals[i+1],
                         "ts": feat_df.index[i]})

    X_test = np.array(X_test_list); X_test_s = scaler.transform(X_test)
    proba = model.predict_proba(X_test_s)[:, 1 if 1 in model.classes_ else 0]

    print(f"  Train: {len(X_train)} | Test: {len(proba)} | "
          f"Prob: mean={proba.mean():.3f} std={proba.std():.3f}")

    # ── TRADING SIMULATION ──
    trades = []; active = []; hourly = defaultdict(int); hour_start = None; last_trade = {}
    # Track funnel
    fast_scans = 0; candidates = 0; edge_passed = 0; ranker_selected = 0
    risk_passed = 0; cooldown_rejected = 0; sym_max_rejected = 0
    max_active_rejected = 0; max_hourly_rejected = 0

    for i, (p, ti) in enumerate(zip(proba, test_info)):
        fast_scans += 1
        ts = ti["ts"]

        if hour_start is None: hour_start = ts
        if (ts - hour_start).total_seconds() > 3600:
            hourly.clear(); hour_start = ts

        # Settle expired
        for at in list(active):
            if (ts - at["entry_time"]).total_seconds() >= HOLD_MIN * 60:
                settle_price = ti["entry"]
                if at["dir"] == "CALL":
                    is_win = settle_price > at["entry_price"]
                else:
                    is_win = settle_price < at["entry_price"]
                at["result"] = "WIN" if is_win else "LOSS"
                at["pnl"] = at["stake"] * payout if is_win else -at["stake"]
                active.remove(at)

        for direction, dir_int, dir_prob in [("CALL", 1, p), ("PUT", 2, 1.0-p)]:
            candidates += 1
            cons_prob = dir_prob - UNC_MARGIN - CAL_MARGIN
            eff_edge = cons_prob - be_prob
            exp_roi = cons_prob * payout - (1.0 - cons_prob)

            if eff_edge < MIN_EDGE or exp_roi < MIN_ROI: continue
            edge_passed += 1

            # Cooldown
            key = (sym, direction)
            if key in last_trade:
                if (ts - last_trade[key]).total_seconds() < COOLDOWN:
                    cooldown_rejected += 1; continue

            # Per-symbol
            sym_active = sum(1 for t in active if t["sym"] == sym)
            if sym_active >= MAX_PER_SYM:
                sym_max_rejected += 1; continue

            # Global
            if len(active) >= MAX_ACTIVE:
                max_active_rejected += 1; continue

            # Hourly
            if hourly[sym] >= MAX_HOURLY:
                max_hourly_rejected += 1; continue

            ranker_selected += 1; risk_passed += 1

            # Kelly sizing
            r_p = payout
            kelly = max(0.0, (cons_prob * (1.0 + r_p) - 1.0) / r_p)
            target_f = KELLY_F * kelly
            eff_f = min(target_f, MAX_BET_F)
            stake = int(math.floor(SIM_EQUITY * eff_f))
            if stake < MIN_ORDER: stake = MIN_ORDER

            trade = {
                "sym": sym, "dir": direction, "dir_int": dir_int,
                "entry_time": ts, "entry_price": ti["entry"],
                "stake": stake, "prob": round(dir_prob, 4),
                "cons_prob": round(cons_prob, 4), "be": round(be_prob, 4),
                "eff_edge": round(eff_edge, 4), "exp_roi": round(exp_roi, 4),
                "payout": payout,
            }

            is_tie = abs(ti["move"]) < 0.0003
            if is_tie:
                trade["result"] = "TIE"; trade["pnl"] = 0.0
            elif direction == "CALL":
                is_win = ti["move"] > 0
                trade["result"] = "WIN" if is_win else "LOSS"
                trade["pnl"] = stake * payout if is_win else -stake
            else:
                is_win = ti["move"] < 0
                trade["result"] = "WIN" if is_win else "LOSS"
                trade["pnl"] = stake * payout if is_win else -stake

            trades.append(trade); active.append(trade)
            last_trade[key] = ts; hourly[sym] += 1

    s = stats(trades)
    grand_total.append(s)
    all_details.append({"sym": sym, "trades": trades, "stats": s, "payout": payout, "be": be_prob,
        "funnel": {"scans": fast_scans, "candidates": candidates, "edge_passed": edge_passed,
                   "ranker": ranker_selected, "risk": risk_passed,
                   "cooldown_rej": cooldown_rejected, "sym_max_rej": sym_max_rejected,
                   "max_active_rej": max_active_rejected, "max_hourly_rej": max_hourly_rejected}})

    print(f"  FUNNEL: scans={fast_scans} candidates={candidates} edge_passed={edge_passed} "
          f"ranker={ranker_selected} risk={risk_passed}")
    print(f"  REJECTED: cooldown={cooldown_rejected} sym_max={sym_max_rejected} "
          f"max_active={max_active_rejected} max_hourly={max_hourly_rejected}")
    print(f"  TRADES: {s['trades']} | Win={s['wins']} Loss={s['losses']} TIE={s['ties']}")
    print(f"  WR={s['wr']:.1%} | PnL={s['pnl']:+.1f}U | ROI={s['roi']:+.1%} | Staked={s['staked']}U")

    if trades:
        for d in ["CALL","PUT"]:
            dt = [t for t in trades if t["dir"]==d]
            if dt:
                ds = stats(dt)
                print(f"    {d}: {ds['trades']}笔 WR={ds['wr']:.1%} PnL={ds['pnl']:+.1f}U")

        # Edge buckets
        buckets = {"0-1%":[],"1-2%":[],"2-3%":[],"3-5%":[],"5%+":[]}
        for t in trades:
            e = t["eff_edge"]
            if e < 0.01: buckets["0-1%"].append(t)
            elif e < 0.02: buckets["1-2%"].append(t)
            elif e < 0.03: buckets["2-3%"].append(t)
            elif e < 0.05: buckets["3-5%"].append(t)
            else: buckets["5%+"].append(t)
        for bn, ts in buckets.items():
            if ts:
                bs = stats(ts)
                print(f"    Edge {bn:>5}: {bs['trades']:>4}笔 WR={bs['wr']:.1%} PnL={bs['pnl']:+.1f}U")

        # Max DD
        cum = [0]
        for t in trades: cum.append(cum[-1] + t["pnl"])
        peak = cum[0]; max_dd = 0
        for v in cum:
            if v > peak: peak = v
            if peak - v > max_dd: max_dd = peak - v
        max_lose = 0; cur = 0
        for t in trades:
            if t["result"] == "LOSS": cur += 1; max_lose = max(max_lose, cur)
            else: cur = 0
        max_win = 0; cur_w = 0
        for t in trades:
            if t["result"] == "WIN": cur_w += 1; max_win = max(max_win, cur_w)
            else: cur_w = 0
        print(f"    MaxDD={max_dd:.1f}U | MaxLoseStreak={max_lose} | MaxWinStreak={max_win}")

# ── Summary ──
print(f"\n\n{'='*70}")
print(f"  TRADING BACKTEST SUMMARY")
print(f"  (15m model, 80% train / 20% test, ~18 days test period)")
print(f"{'='*70}")
for r in all_details:
    s = r["stats"]
    print(f"\n  {r['sym']} (ASSUMED_PAYOUT={r['payout']}, BE={r['be']:.1%}):")
    print(f"    Trades: {s['trades']} | Win: {s['wins']} | Loss: {s['losses']} | TIE: {s['ties']}")
    print(f"    WR: {s['wr']:.1%} | PnL: {s['pnl']:+.1f}U | ROI: {s['roi']:+.1%} | Staked: {s['staked']}U")

total_t = sum(s["trades"] for s in grand_total)
total_p = sum(s["pnl"] for s in grand_total)
total_s = sum(s["staked"] for s in grand_total)
print(f"\n  OVERALL: {total_t} trades, PnL={total_p:+.1f}U, ROI={total_p/total_s if total_s>0 else 0:+.1%}")
print(f"  PAYOUT_VERIFIED = false")
print(f"  ASSUMED_PAYOUT: BTC={PAYOUTS['BTCUSDT']}, ETH={PAYOUTS['ETHUSDT']}, SOL={PAYOUTS['SOLUSDT']}")
print(f"{'='*70}")