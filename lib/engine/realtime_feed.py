# -*- coding: utf-8 -*-
"""
RealtimeFeed — 实时行情数据源

使用 Gate.io 1m K线 API 高频轮询，维护 1m/5m/15m 多周期 K 线。
- 每 5 秒拉取最新 1m K 线
- 在内存中维护 closed candles (1m/5m/15m)
- 提供实时价格（最新 ticker）
- 允许访问 forming 1m candle 的实时状态
"""
import time
import threading
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from curl_cffi import requests

API_URL = "https://api.gateio.ws/api/v4/spot/candlesticks"
TICKER_URL = "https://api.gateio.ws/api/v4/spot/tickers"
SYMBOL_PAIRS = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "SOLUSDT": "SOL_USDT"}

# 各周期需要的 K 线数量
KLINE_LIMITS = {"1m": 120, "5m": 96, "15m": 200}


@dataclass
class RealtimePrice:
    """实时价格快照"""
    symbol: str = ""
    price: float = 0.0
    timestamp: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume_24h: float = 0.0
    change_pct: float = 0.0


@dataclass
class KlineStore:
    """多周期 K 线存储"""
    symbol: str = ""
    # closed candles: interval -> DataFrame
    closed: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # forming 1m candle
    forming_1m: Optional[dict] = None
    # realtime price
    realtime: Optional[RealtimePrice] = None
    # last update timestamps
    last_update: Dict[str, float] = field(default_factory=dict)


class RealtimeFeed:
    """
    实时行情数据源。

    维护 1m/5m/15m 多周期 K 线，每 5 秒刷新。
    1m 和 5m 数据从 1m API 聚合得到，15m 从 15m API 获取。
    """

    def __init__(self, symbols: List[str], scan_interval: float = 5.0):
        self.symbols = symbols
        self.scan_interval = scan_interval
        self.stores: Dict[str, KlineStore] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_count = 0
        self._last_scan = 0.0

        for sym in symbols:
            self.stores[sym] = KlineStore(symbol=sym)

    def start(self):
        """启动后台数据拉取线程"""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # 初始拉取
        self._fetch_all()

    def stop(self):
        self._running = False

    def _run(self):
        """后台线程：定期拉取数据"""
        while self._running:
            try:
                self._fetch_all()
                self._update_count += 1
                self._last_scan = time.time()
            except Exception:
                pass
            time.sleep(self.scan_interval)

    def _fetch_all(self):
        """拉取所有品种的 1m K 线和实时价格"""
        for sym in self.symbols:
            pair = SYMBOL_PAIRS.get(sym)
            if not pair:
                continue
            try:
                self._fetch_klines(sym, pair, "1m", 120)
                self._fetch_ticker(sym, pair)
            except Exception:
                pass

    def _fetch_klines(self, sym: str, pair: str, interval: str, limit: int):
        """拉取 K 线"""
        try:
            r = requests.get(API_URL, params={
                "currency_pair": pair, "interval": interval, "limit": limit,
            }, impersonate="chrome110", timeout=5, verify=False)
            if r.status_code != 200:
                return
            data = r.json()
            if not data or len(data) < 2:
                return
        except Exception:
            return

        df = pd.DataFrame(data, columns=["ts", "qv", "close", "high", "low", "open", "volume", "final"])
        df["dt"] = pd.to_datetime(df["ts"].astype(int), unit="s", utc=True)
        df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])

        with self._lock:
            store = self.stores[sym]
            store.closed["1m"] = df
            store.last_update["1m"] = time.time()

            # 聚合 5m
            if len(df) >= 5:
                df_5m = df.resample("5min").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum",
                }).dropna()
                store.closed["5m"] = df_5m
                store.last_update["5m"] = time.time()

    def _fetch_ticker(self, sym: str, pair: str):
        """拉取实时价格"""
        try:
            r = requests.get(TICKER_URL, params={"currency_pair": pair},
                           impersonate="chrome110", timeout=5, verify=False)
            if r.status_code != 200:
                return
            data = r.json()
            if not data:
                return
            ticker = data[0] if isinstance(data, list) else data
            price = float(ticker.get("last", 0))
            if price > 0:
                with self._lock:
                    self.stores[sym].realtime = RealtimePrice(
                        symbol=sym, price=price, timestamp=time.time(),
                        bid=float(ticker.get("highest_bid", 0)),
                        ask=float(ticker.get("lowest_ask", 0)),
                        volume_24h=float(ticker.get("base_volume", 0)),
                        change_pct=float(ticker.get("change_percentage", 0)),
                    )
        except Exception:
            pass

    def get_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """获取已收盘 K 线"""
        with self._lock:
            store = self.stores.get(symbol)
            if store is None:
                return None
            return store.closed.get(interval)

    def get_realtime_price(self, symbol: str) -> Optional[RealtimePrice]:
        """获取实时价格"""
        with self._lock:
            store = self.stores.get(symbol)
            if store is None:
                return None
            return store.realtime

    def get_realtime_features(self, symbol: str) -> dict:
        """
        获取实时特征（用于 Fast Entry Model）。

        返回:
            dict with:
            - price: 当前价格
            - return_1m, return_5m, return_15m: 短期回报
            - price_vs_1m_close: 当前价格 vs 最后 1m 收盘价
            - seconds_since_1m_close: 距最后 1m 收盘的秒数
            - closed_1m_features: 最后 closed 1m candle 的特征
            - closed_5m_features: 最后 closed 5m candle 的特征
        """
        with self._lock:
            store = self.stores.get(symbol)
            if store is None:
                return {}

            rt = store.realtime
            price = rt.price if rt else 0.0

            features = {"price": price, "symbol": symbol}

            # 1m closed candles
            df_1m = store.closed.get("1m")
            if df_1m is not None and len(df_1m) >= 5:
                closes = df_1m["close"].values
                features["return_1m"] = (price / closes[-1] - 1) if closes[-1] > 0 else 0
                features["return_3m"] = (price / closes[-3] - 1) if len(closes) >= 3 and closes[-3] > 0 else 0
                features["return_5m"] = (price / closes[-5] - 1) if len(closes) >= 5 and closes[-5] > 0 else 0
                features["last_1m_close"] = float(closes[-1])
                features["last_1m_volume"] = float(df_1m["volume"].values[-1])
                features["volume_ratio_1m"] = float(df_1m["volume"].values[-1] / max(df_1m["volume"].values[-5:].mean(), 0.001))
                features["price_vs_1m_close"] = features["return_1m"]

                # 计算 RSI(14) on 1m
                if len(closes) >= 15:
                    deltas = np.diff(closes[-15:])
                    gains = np.maximum(deltas, 0).mean()
                    losses = np.maximum(-deltas, 0).mean()
                    rs = gains / max(losses, 1e-10)
                    features["rsi_1m"] = float(100 - 100 / (1 + rs))
                else:
                    features["rsi_1m"] = 50.0

                # 1m 波动率
                if len(closes) >= 15:
                    returns = np.diff(closes[-15:]) / closes[-16:-1]
                    features["volatility_1m"] = float(np.std(returns))
                else:
                    features["volatility_1m"] = 0.0

                # 1m 动量
                features["momentum_1m"] = features["return_1m"]
                features["momentum_3m"] = features["return_3m"]

                # 价格加速度
                if len(closes) >= 3:
                    ret_1 = (closes[-1] / closes[-2] - 1) if closes[-2] > 0 else 0
                    ret_2 = (closes[-2] / closes[-3] - 1) if closes[-3] > 0 else 0
                    features["price_acceleration"] = ret_1 - ret_2
                else:
                    features["price_acceleration"] = 0.0

                # 距最后 1m 收盘的秒数
                last_ts = df_1m.index[-1]
                features["seconds_since_1m_close"] = (time.time() - last_ts.timestamp()) if hasattr(last_ts, 'timestamp') else 0

            # 5m closed candles
            df_5m = store.closed.get("5m")
            if df_5m is not None and len(df_5m) >= 3:
                closes_5 = df_5m["close"].values
                features["return_5m"] = (price / closes_5[-1] - 1) if closes_5[-1] > 0 else 0
                features["return_15m"] = (price / closes_5[-3] - 1) if len(closes_5) >= 3 and closes_5[-3] > 0 else 0

            return features

    def get_stats(self) -> dict:
        """获取数据源统计"""
        return {
            "running": self._running,
            "update_count": self._update_count,
            "last_scan": self._last_scan,
            "symbols": {s: {"last_update": self.stores[s].last_update, "has_price": self.stores[s].realtime is not None} for s in self.symbols},
        }


# 全局单例
_feed: Optional[RealtimeFeed] = None


def get_realtime_feed(symbols: Optional[List[str]] = None) -> RealtimeFeed:
    global _feed
    if _feed is None and symbols:
        _feed = RealtimeFeed(symbols)
    return _feed