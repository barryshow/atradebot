#!/usr/bin/env python3
"""Backtest NEW vs OLD params for quality control comparison"""
import sys, io, os, time, warnings, math
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np, pandas as pd
from curl_cffi import requests
import joblib

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}
HOLD_MIN = 15; MIN_ORDER = 3

# ── NEW params ──
NEW = {
    "max_hourly": 3, "max_active": 1, "max_per_sym": 1, "cooldown": 300,
    "min_edge": 0.05, "min_roi": 0.03, "unc_margin": 0.03, "cal_margin": 0.015,
    "min_direction": 0.08, "kelly_frac": 0.25, "max_bet_frac": 0.08,
}

# ── OLD params (what lost yesterday) ──
OLD = {
    "max_hourly": 4, "max_active": 3, "max_per_sym": 1, "cooldown": 60,
    "min_edge": 0.02, "min_roi": 0.005, "unc_margin": 0.02, "cal_margin": 0.01,
    "min_direction": 0.0, "kelly_frac": 0.50, "max_bet_frac": 0.10,
}

def fetch_klines(pair, interval="1m", days=7):
    limit = 1000; all_rows, last_ts = [], int(time.time())
    for _ in range(15):
        if len(all_rows) >= days * 24 * 60: break
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

def aggregate_5m(df_1m):
    return df_1m.resample("5min").agg({
        "open":"first","high":"max","low":"min","close":"last","volume":"sum",
    }).dropna()

def stat(trades):
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

def run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, capital, cfg):
    equity = float(capital)
    trades = []; active = []; hourly = defaultdict(int); hour_start = None; last_trade = {}
    edge_rej = 0; cooldown_rej = 0; weak_dir_rej = 0; sym_max_rej = 0
    max_active_rej = 0; max_hourly_rej = 0; total_scans = 0

    for i_idx, j in enumerate(idx_list):
        p = proba[i_idx]
        ts = df_1m.index[j]
        entry_price = float(df_1m["close"].values[j])
        if j + 15 >= n: continue
        if hour_start is None: hour_start = ts
        if (ts - hour_start).total_seconds() > 3600:
            hourly.clear(); hour_start = ts

        # Settle active trades
        for at in list(active):
            if (ts - at["entry_time"]).total_seconds() >= HOLD_MIN * 60:
                sp = entry_price
                is_win = sp > at["entry_price"] if at["dir"] == "CALL" else sp < at["entry_price"]
                at["result"] = "WIN" if is_win else "LOSS"
                at["pnl"] = at["stake"] * payout if is_win else -at["stake"]
                equity += at["pnl"] + at["stake"]
                active.remove(at)

        for direction, dir_prob in [("CALL", p), ("PUT", 1.0-p)]:
            total_scans += 1

            # Direction strength filter (NEW only, OLD has min_direction=0)
            if abs(dir_prob - 0.50) < cfg["min_direction"]:
                weak_dir_rej += 1; continue

            cons_prob = dir_prob - cfg["unc_margin"] - cfg["cal_margin"]
            eff_edge = cons_prob - be_prob
            exp_roi = cons_prob * payout - (1.0 - cons_prob)
            if eff_edge < cfg["min_edge"] or exp_roi < cfg["min_roi"]: edge_rej += 1; continue

            key = (sym, direction)
            if key in last_trade and (ts - last_trade[key]).total_seconds() < cfg["cooldown"]:
                cooldown_rej += 1; continue
            if sum(1 for t in active if t["sym"]==sym) >= cfg["max_per_sym"]:
                sym_max_rej += 1; continue
            if len(active) >= cfg["max_active"]: max_active_rej += 1; continue
            if hourly[sym] >= cfg["max_hourly"]: max_hourly_rej += 1; continue

            r_p = payout
            k = max(0.0, (cons_prob * (1.0 + r_p) - 1.0) / r_p) if r_p > 0 else 0.0
            target_f = cfg["kelly_frac"] * k
            eff_f = min(target_f, cfg["max_bet_frac"])
            raw = equity * eff_f
            stake = int(math.floor(raw))
            if stake < MIN_ORDER: stake = MIN_ORDER
            if stake > equity: stake = int(equity)

            trade = {"sym":sym,"dir":direction,"entry_time":ts,"entry_price":entry_price,
                "stake":stake,"eff_edge":eff_edge,"cons_prob":cons_prob}
            move = (float(df_1m["close"].values[j+15]) - entry_price) / entry_price
            is_tie = abs(move) < 0.0003
            if is_tie: trade["result"]="TIE"; trade["pnl"]=0.0
            elif direction=="CALL": trade["result"]="WIN" if move>0 else "LOSS"; trade["pnl"]=stake*payout if move>0 else -stake
            else: trade["result"]="WIN" if move<0 else "LOSS"; trade["pnl"]=stake*payout if move<0 else -stake
            trades.append(trade); active.append(trade)
            last_trade[key]=ts; hourly[sym]+=1; equity-=stake

    s = stat(trades)
    filters = {"edge_rej":edge_rej//2,"cooldown":cooldown_rej//2,"weak_dir":weak_dir_rej//2,
        "sym_max":sym_max_rej//2,"max_active":max_active_rej//2,"max_hourly":max_hourly_rej//2,"scans":total_scans//2}
    return s, trades, filters

# ── Load models ──
from lib.engine.multi_timeframe_features import compute_fast_entry_features, build_fast_feature_vector

model_dir = "models"
models = {}
for sym in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
    path = os.path.join(model_dir, f"{sym.lower()}_fast_entry.pkl")
    if os.path.exists(path):
        bundle = joblib.load(path)
        models[sym] = (bundle["model"], bundle["scaler"])

print("=" * 80)
print("  QUALITY CONTROL BACKTEST: NEW PARAMS vs OLD PARAMS")
print("  NEW: Edge>5% ROI>3% Margins(3%/1.5%) DirStrength>8% Kelly25% MaxBet8% Cooldown300s MaxActive1 MaxHourly3")
print("  OLD: Edge>2% ROI>0.5% Margins(2%/1%) NoDirFilter Kelly50% MaxBet10% Cooldown60s MaxActive3 MaxHourly4")
print("=" * 80)

capital_levels = [14, 50, 100]

for sym in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
    if sym not in models: continue
    pair = SYMBOLS[sym]; payout = PAYOUTS[sym]; be_prob = 1.0/(1.0+payout)
    model, scaler = models[sym]

    print(f"\n{'─'*80}")
    print(f"  {sym} | Payout={payout}  BreakEven={be_prob:.1%}")
    print(f"{'─'*80}")

    print("  Fetching 7d klines...", end=" ", flush=True)
    df_1m = fetch_klines(pair, interval="1m", days=7)
    if df_1m is None or len(df_1m) < 1000:
        print("SKIP (not enough data)")
        continue
    n = len(df_1m); df_5m = aggregate_5m(df_1m)

    start_idx = int(n * 0.60); step = 5
    print(f"Computing features...", end=" ", flush=True)
    X_list = []; idx_list = []
    for i in range(start_idx + 50, n - 15, step):
        df_1m_before = df_1m.iloc[:i+1].copy()
        df_5m_before = df_5m[df_5m.index <= df_1m.index[i]].copy()
        features = compute_fast_entry_features(sym, None, df_1m_before, df_5m_before,
            slow_context={"probability":0.50,"regime":"RANGE","trend_strength":0,"volatility":0})
        try:
            vec = build_fast_feature_vector(features)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)): continue
        except: continue
        X_list.append(vec); idx_list.append(i)

    X = np.array(X_list); X_s = scaler.transform(X)
    proba = model.predict_proba(X_s)[:, 1 if 1 in model.classes_ else 0]
    print(f"{len(proba)} samples | prob mean={proba.mean():.4f} std={proba.std():.4f}")

    # ── Run both parameter sets ──
    print(f"\n  {'Capital':>5} {'Mode':>4} {'Trades':>6} {'WR':>7} {'PnL':>9} {'AvgBet':>7} {'EdgePass':>9} {'Scans':>7}")
    print(f"  {'─'*5} {'─'*4} {'─'*6} {'─'*7} {'─'*9} {'─'*7} {'─'*9} {'─'*7}")

    for cap in capital_levels:
        for mode_name, cfg in [("NEW", NEW), ("OLD", OLD)]:
            s, trades, filters = run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, cap, cfg)
            avg_bet = s["staked"]/s["trades"] if s["trades"] > 0 else 0
            edge_pass = (filters["scans"] - filters["edge_rej"]) / filters["scans"] * 100 if filters["scans"] > 0 else 0
            pnl_str = f"{s['pnl']:+.2f}U" if s["trades"] > 0 else "N/A"
            print(f"  {cap:>5}U {mode_name:>4} {s['trades']:>6} {s['wr']:>6.1%} {pnl_str:>9} {avg_bet:>6.1f}U {edge_pass:>8.1f}% {filters['scans']:>7}")

            # Show filter breakdown for NEW mode only
            if mode_name == "NEW" and filters["scans"] > 0:
                wd = filters["weak_dir"]; er = filters["edge_rej"]
                cd = filters["cooldown"]; sm = filters["sym_max"]
                ma = filters["max_active"]; mh = filters["max_hourly"]
                passed = filters["scans"] - wd - er - cd - sm - ma - mh
                if wd > 0:
                    print(f"       └─ Filter: weak_dir={wd} edge_rej={er} cooldown={cd} sym_max={sm} max_act={ma} max_h={mh}→passed={passed}")

# ── SUMMARY ──
print(f"\n{'='*80}")
print("  SUMMARY: Expected Impact of Quality Controls")
print(f"{'='*80}")
print("""
  NEW params will:
  1. Reject ~60-80% of old candidates (weak direction + thin edge removed)
  2. Open 1/3 as many trades (max_active=1 vs 3)
  3. Wait 5x longer between same-direction trades (300s vs 60s cooldown)
  4. Bet 1/2 as aggressive (25% kelly vs 50% kelly)
  5. Require actual direction conviction (8% from 0.50)

  These filters CANNOT turn a bad model into a good one.
  If the underlying model is noise (AUC ~0.64, actual wr ~50-55%),
  filtering tighter just reduces trade count — ROI will improve
  but absolute PnL will be lower (fewer bets at same edge).

  KEY QUESTION: Does edge bucket show higher WR at higher edge?
  If YES → filtering works (remove low-edge noise)
  If NO  → model is random, needs retrain, not filtering
""")
