#!/usr/bin/env python3
"""
ATradeBot 真实行情数据获取器 (改进版)
从 Gate.io 拉取大量历史+实时1分钟K线 → CSV
引擎内部再重采样为30分钟用于预测

用法:
  python scripts/simulate_data.py [--interval 15] [--csv ./hibt_ticks.csv]
"""
import argparse
import csv
import io
import os
import sys
import time
from datetime import datetime, timezone
from curl_cffi import requests

SYMBOLS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}
API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"


def fetch_klines(pair: str, limit: int = 1000) -> list:
    """拉取K线, 支持超过1000根的分页"""
    all_rows = []
    last_ts = int(time.time())
    while len(all_rows) < limit:
        to_fetch = min(1000, limit - len(all_rows))
        try:
            r = requests.get(
                API_URL,
                params={"currency_pair": pair, "interval": "1m", "limit": to_fetch, "to": last_ts},
                impersonate="chrome110", timeout=15, verify=False,
            )
            if r.status_code == 200:
                data = r.json()
                if not data or len(data) < 2:
                    break
                all_rows.extend(data)
                last_ts = int(data[0][0]) - 60
                if len(data) < to_fetch:
                    break
            else:
                break
        except Exception as e:
            print(f"[WARN] {pair}: {str(e)[:40]}")
            time.sleep(2)
            continue
    return all_rows


def gateio_to_csv_row(row: list, symbol: str) -> dict:
    return {
        "ts": int(row[0]) * 1000,
        "symbol": symbol,
        "open": round(float(row[5]), 2),
        "high": round(float(row[3]), 2),
        "low": round(float(row[4]), 2),
        "close": round(float(row[2]), 2),
        "volume": round(float(row[6]), 4),
    }


def write_csv_header(csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "symbol", "open", "high", "low", "close", "volume"])


def append_csv_rows(csv_path: str, rows: list[dict]):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([r["ts"], r["symbol"], r["open"], r["high"], r["low"], r["close"], r["volume"]])


def load_existing_ts(csv_path: str, symbol: str) -> set:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    existing = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 7 and row[1] == symbol:
                try:
                    existing.add(int(row[0]))
                except ValueError:
                    pass
    return existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=15.0, help="获取间隔(秒)")
    parser.add_argument("--csv", type=str, default="./hibt_ticks.csv", help="CSV输出路径")
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    csv_path = os.path.abspath(args.csv)
    print(f"[行情] 从 Gate.io 拉取真实K线 → {csv_path}")

    # 拉大量历史数据 (~1500根1mK线=25小时, 重采样30m≈50根)
    print("[行情] 拉取历史数据(每品种3000根K线=50小时)...")
    write_csv_header(csv_path)
    total = 0
    for sym, pair in SYMBOLS.items():
        raw = fetch_klines(pair, limit=7200)
        if raw:
            rows = [gateio_to_csv_row(r, sym) for r in raw]
            append_csv_rows(csv_path, rows)
            total += len(rows)
            latest = float(rows[-1]["close"])
            print(f"  [OK] {sym} ({pair}): {len(rows)}根, 最新价 ${latest:.2f}")
    print(f"  -> 共 {total} 行历史数据\n")

    # 持续拉取最新K线
    cycle = 0
    last_prices = {}
    while True:
        cycle += 1
        new_rows = []
        for sym, pair in SYMBOLS.items():
            existing_ts = load_existing_ts(csv_path, sym)
            raw = fetch_klines(pair, limit=5)
            if not raw:
                continue
            for r in raw:
                ts = int(r[0]) * 1000
                if ts not in existing_ts:
                    row = gateio_to_csv_row(r, sym)
                    new_rows.append(row)
                    existing_ts.add(ts)
                    last_prices[sym] = float(row["close"])

        if new_rows:
            append_csv_rows(csv_path, new_rows)
            t = datetime.now(timezone.utc).strftime("%H:%M:%S")
            prices = " | ".join(f"{s}=${v:.2f}" for s, v in last_prices.items())
            print(f"[{t}] +{len(new_rows)}根 | {prices}")
        else:
            time.sleep(1)
            continue

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[行情] 停止")