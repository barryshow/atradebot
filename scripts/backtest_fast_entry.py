#!/usr/bin/env python3
"""
ATradeBot Fast Entry Trading Backtest v2
Uses PRE-TRAINED Fast Entry models.
Two modes: (A) Fixed 3U, (B) Kelly with realistic caps.
"""
import sys, io, os, time, warnings, math
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from curl_cffi import requests
import joblib
from collections import defaultdict

from lib.engine.multi_timeframe_features import (
    compute_fast_entry_features, FAST_FEATURES, build_fast_feature_vector,
)

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
PAYOUTS = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}

MAX_HOURLY = 4; MAX_ACTIVE = 3; MAX_PER_SYM = 1; COOLDOWN = 60
HOLD_MIN = 15; MIN_EDGE = 0.02; MIN_ROI = 0.005
UNC_MARGIN = 0.005; CAL_MARGIN = 0.01
MIN_ORDER = 3

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

def run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, capital, fixed_bet=True, kelly_frac=0.50, max_bet_frac=0.35):
    """Run trading simulation. fixed_bet=True uses constant 3U. Otherwise uses Kelly."""
    equity = float(capital)
    start_equity = equity
    trades = []; active = []; hourly = defaultdict(int); hour_start = None; last_trade = {}
    edge_rej = 0; cooldown_rej = 0; sym_max_rej = 0; max_active_rej = 0; max_hourly_rej = 0

    for i_idx, j in enumerate(idx_list):
        p = proba[i_idx]
        ts = df_1m.index[j]
        entry_price = float(df_1m["close"].values[j])
        if j + 15 >= n: continue
        move = (float(df_1m["close"].values[j+15]) - entry_price) / entry_price

        if hour_start is None: hour_start = ts
        if (ts - hour_start).total_seconds() > 3600:
            hourly.clear(); hour_start = ts

        # Settle
        for at in list(active):
            if (ts - at["entry_time"]).total_seconds() >= HOLD_MIN * 60:
                sp = entry_price
                if at["dir"] == "CALL": is_win = sp > at["entry_price"]
                else: is_win = sp < at["entry_price"]
                at["result"] = "WIN" if is_win else "LOSS"
                at["pnl"] = at["stake"] * payout if is_win else -at["stake"]
                equity += at["pnl"] + at["stake"]
                active.remove(at)

        for direction, dir_int, dir_prob in [("CALL", 1, p), ("PUT", 2, 1.0-p)]:
            cons_prob = dir_prob - UNC_MARGIN - CAL_MARGIN
            eff_edge = cons_prob - be_prob
            exp_roi = cons_prob * payout - (1.0 - cons_prob)
            if eff_edge < MIN_EDGE or exp_roi < MIN_ROI: edge_rej += 1; continue

            key = (sym, direction)
            if key in last_trade and (ts - last_trade[key]).total_seconds() < COOLDOWN:
                cooldown_rej += 1; continue
            if sum(1 for t in active if t["sym"]==sym) >= MAX_PER_SYM:
                sym_max_rej += 1; continue
            if len(active) >= MAX_ACTIVE: max_active_rej += 1; continue
            if hourly[sym] >= MAX_HOURLY: max_hourly_rej += 1; continue

            if fixed_bet:
                stake = MIN_ORDER
            else:
                r_p = payout
                kelly = max(0.0, (cons_prob * (1.0 + r_p) - 1.0) / r_p) if r_p > 0 else 0.0
                target_f = kelly_frac * kelly
                eff_f = min(target_f, max_bet_frac)
                raw = equity * eff_f
                stake = int(math.floor(raw))
                if stake < MIN_ORDER: stake = MIN_ORDER
                if stake > equity: stake = int(equity)

            trade = {"sym":sym,"dir":direction,"entry_time":ts,"entry_price":entry_price,
                "stake":stake,"eff_edge":eff_edge,"payout":payout}
            is_tie = abs(move) < 0.0003
            if is_tie: trade["result"]="TIE"; trade["pnl"]=0.0
            elif direction=="CALL": trade["result"]="WIN" if move>0 else "LOSS"; trade["pnl"]=stake*payout if move>0 else -stake
            else: trade["result"]="WIN" if move<0 else "LOSS"; trade["pnl"]=stake*payout if move<0 else -stake
            trades.append(trade); active.append(trade)
            last_trade[key]=ts; hourly[sym]+=1; equity-=stake

    s = stat(trades)
    # Max DD
    cum = [0]
    for t in trades: cum.append(cum[-1] + t["pnl"])
    peak = cum[0]; max_dd = 0
    for v in cum:
        if v > peak: peak = v
        if peak - v > max_dd: max_dd = peak - v
    max_lose = 0; cur = 0
    for t in trades:
        if t["result"]=="LOSS": cur += 1; max_lose = max(max_lose, cur)
        else: cur = 0
    return s, max_dd, max_lose, edge_rej//2, trades

print("=" * 70)
print("  ATradeBot FAST ENTRY TRADING BACKTEST v2")
print(f"  PRE-TRAINED models | {len(FAST_FEATURES)} features | 1-min entry")
print(f"  Mode A: FIXED 3U  |  Mode B: KELLY (half, 10% max bet)")
print("=" * 70)

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    pair = SYMBOLS[sym]; payout = PAYOUTS[sym]; be_prob = 1.0/(1.0+payout)
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "models", f"{sym.lower()}_fast_entry.pkl")
    if not os.path.exists(model_path):
        print(f"\n  {sym}: SKIP — no model"); continue
    bundle = joblib.load(model_path)
    model = bundle["model"]; scaler = bundle["scaler"]

    print(f"\n{'─'*70}")
    print(f"  {sym} (payout={payout}, BE={be_prob:.1%})")
    print(f"{'─'*70}")

    df_1m = fetch_klines(pair, interval="1m", days=7)
    if df_1m is None or len(df_1m) < 1000: continue
    n = len(df_1m); df_5m = aggregate_5m(df_1m)

    # Build features
    start_idx = int(n * 0.60); step = 5
    print(f"  Computing features (every {step}min)...", end=" ", flush=True)
    X_list = []; idx_list = []
    for i in range(start_idx + 50, n - 15, step):
        df_1m_before = df_1m.iloc[:i+1].copy()
        df_5m_before = df_5m[df_5m.index <= df_1m.index[i]].copy()
        features = compute_fast_entry_features(
            sym, None, df_1m_before, df_5m_before,
            slow_context={"probability":0.50,"regime":"RANGE","trend_strength":0,"volatility":0})
        try:
            vec = build_fast_feature_vector(features)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)): continue
        except: continue
        X_list.append(vec); idx_list.append(i)

    X = np.array(X_list); X_s = scaler.transform(X)
    proba = model.predict_proba(X_s)[:, 1 if 1 in model.classes_ else 0]
    print(f"{len(proba)} samples | prob mean={proba.mean():.4f} std={proba.std():.4f}")

    # Probability Reliability
    print(f"  Prob Reliability (CALL):")
    for lo, hi in [(0.50,0.52),(0.52,0.54),(0.54,0.56),(0.56,0.58),(0.58,0.60),(0.60,0.65),(0.65,1.0)]:
        mask = (proba >= lo) & (proba < hi)
        n_b = mask.sum()
        if n_b >= 5:
            pred = float(proba[mask].mean())
            actuals = []
            for jj in range(len(proba)):
                if mask[jj]:
                    ii = idx_list[jj]
                    entry = float(df_1m["close"].values[ii])
                    expiry = float(df_1m["close"].values[ii+15]) if ii+15 < n else entry
                    actuals.append(1 if (expiry-entry)/entry > 0.0003 else 0)
            act = np.mean(actuals)
            tag = " OK" if act > be_prob else " BELOW_BE"
            if n_b < 20: tag += " LOW_SAMPLE"
            print(f"    [{lo:.2f}-{hi:.2f}]: n={n_b:>4} pred={pred:.1%} actual={act:.1%}{tag}")

    # Edge pass rate
    ep = 0
    for p in proba:
        for dp in [p, 1.0-p]:
            cons = dp - UNC_MARGIN - CAL_MARGIN
            if (cons - be_prob) > MIN_EDGE and (cons * payout - (1.0 - cons)) > MIN_ROI:
                ep += 1; break
    print(f"  Edge Pass: {ep}/{len(proba)} ({100*ep/len(proba):.1f}%)")

    # ── MODE A: Fixed 3U ──
    print(f"\n  MODE A: FIXED 3U (no Kelly, no compounding)")
    print(f"  {'Capital':>6} {'Trades':>6} {'WR':>7} {'PnL':>8} {'ROI':>7} {'MaxDD':>8} {'MaxLose':>8}")
    print(f"  {'─'*65}")
    for cap in [14, 50, 100, 300]:
        s, dd, ml, er, _ = run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, cap, fixed_bet=True)
        print(f"  {cap:>6}U: {s['trades']:>6} {s['wr']:>6.1%} {s['pnl']:>+8.1f}U {s['roi']:>+6.1%} {dd:>8.1f}U {ml:>8}")

    # ── MODE B: Kelly with realistic cap (10% max bet) ──
    print(f"\n  MODE B: KELLY (half, 10% max bet, 3U floor)")
    print(f"  {'Capital':>6} {'Trades':>6} {'WR':>7} {'PnL':>8} {'ROI':>7} {'MaxDD':>8} {'MaxLose':>8} {'AvgBet':>7}")
    print(f"  {'─'*75}")
    for cap in [14, 50, 100, 300]:
        s, dd, ml, er, trades = run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, cap, fixed_bet=False, kelly_frac=0.50, max_bet_frac=0.10)
        avg_bet = s['staked']/s['trades'] if s['trades']>0 else 0
        print(f"  {cap:>6}U: {s['trades']:>6} {s['wr']:>6.1%} {s['pnl']:>+8.1f}U {s['roi']:>+6.1%} {dd:>8.1f}U {ml:>8} {avg_bet:>7.1f}U")

    # ── Edge Bucket (MODE A, 50U) ──
    _, _, _, _, trades_50 = run_trades(proba, idx_list, df_1m, n, sym, payout, be_prob, 50, fixed_bet=True)
    if trades_50:
        print(f"\n  Edge Bucket (Fixed 3U, 50U capital):")
        buckets = {"0-1%":[],"1-2%":[],"2-3%":[],"3-5%":[],"5%+":[]}
        for t in trades_50:
            e = t["eff_edge"]
            if e < 0.01: buckets["0-1%"].append(t)
            elif e < 0.02: buckets["1-2%"].append(t)
            elif e < 0.03: buckets["2-3%"].append(t)
            elif e < 0.05: buckets["3-5%"].append(t)
            else: buckets["5%+"].append(t)
        monotonic = True; prev_wr = None
        for bn, ts in buckets.items():
            if ts:
                bs = stat(ts)
                wr = bs["wr"]; tag = ""
                if prev_wr is not None and wr < prev_wr: monotonic = False; tag = " ⚠NON-MONOTONIC"
                print(f"    {bn:>5}: {bs['trades']:>4}笔 WR={bs['wr']:.1%} PnL={bs['pnl']:+.1f}U{tag}")
                prev_wr = wr
        if not monotonic: print(f"    ⚠ Edge→WR not monotonic → calibration/uncertainty issue")

print(f"\n{'='*70}")
print(f"  PAYOUT_VERIFIED = false")
print(f"  ASSUMED_PAYOUT: BTC={PAYOUTS['BTCUSDT']}, ETH={PAYOUTS['ETHUSDT']}, SOL={PAYOUTS['SOLUSDT']}")
print(f"  Test period: last 40% of 7d 1m data, every 5th minute")
print(f"  ⚠ 7-day test period is SHORT — results may not generalize")
print(f"{'='*70}")