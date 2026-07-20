# -*- coding: utf-8 -*-
"""
LabelBuilder — Event Contract 真实结算标签构建器

为 HIBT Event Contract 构建训练标签，替代旧版 "next candle up/down" 标签。

核心原则:
1. 标签必须模拟 HIBT Event Contract 到期时的真实结算逻辑
2. 严格防止 Look-Ahead Bias: 所有 feature 只能使用 entry_timestamp 之前的数据
3. 多个 Expiry Horizon: 5m, 15m, 30m, 60m
4. TIE 单独记录，不纳入二分类训练
5. 价格源标注: "gate_io" (代理) 或 "hibt_official" (如有)

CALL:  expiry_price > entry_price  → WIN
PUT:   expiry_price < entry_price  → WIN
TIE:   expiry_price == entry_price → TIE (PnL=0)

用法:
    from lib.engine.label_builder import LabelBuilder

    builder = LabelBuilder(price_source="gate_io")
    labels = builder.build_labels(df_1m, symbol="BTCUSDT")
    # labels 是一个 DataFrame，每行一条样本
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from .models import TrainLabel


# HIBT 支持的到期期限（分钟后缀）
EXPIRY_HORIZONS = {
    5: "5m",
    15: "15m",
    30: "30m",
    60: "60m",
}

# 可用的到期期限（仅已验证的）
DEFAULT_EXPIRIES = [15]  # 当前仅 15m 已验证


@dataclass
class LabelStats:
    """标签统计"""
    symbol: str = ""
    expiry_minutes: int = 0
    total_samples: int = 0
    call_samples: int = 0
    put_samples: int = 0
    tie_samples: int = 0
    call_win_rate: float = 0.0
    put_win_rate: float = 0.0
    call_win_count: int = 0
    put_win_count: int = 0
    avg_entry_price: float = 0.0
    avg_expiry_price: float = 0.0
    price_source: str = ""


class LabelBuilder:
    """
    构建 Event Contract 训练标签。

    工作原理:
    1. 读取 1 分钟 OHLCV 数据
    2. 对每一根 K 线（entry_timestamp），向前看 expiry_minutes 根 K 线
    3. 计算 CALL 和 PUT 的结算结果
    4. 记录所有样本，包括 TIE

    Look-Ahead Bias 防护:
    - 标签使用 entry_timestamp 之后的数据（expiry 价格）
    - 但训练时，feature 只能使用 entry_timestamp 之前的数据
    - 这个分离由训练脚本保证（使用时间序列切分）
    """

    def __init__(
        self,
        price_source: str = "gate_io",
        expiries: Optional[List[int]] = None,
    ):
        """
        Args:
            price_source: 价格来源标识 ("gate_io", "binance", "hibt_official")
            expiries: 要构建的到期期限列表，默认 [15]
        """
        self.price_source = price_source
        self.expiries = expiries or DEFAULT_EXPIRIES

    def build_labels(
        self,
        df_1m: pd.DataFrame,
        symbol: str = "",
        min_samples: int = 200,
    ) -> pd.DataFrame:
        """
        从 1 分钟 OHLCV 数据构建标签。

        Args:
            df_1m: 1 分钟 K 线 DataFrame，必须包含:
                - datetime index
                - open, high, low, close, volume 列
            symbol: 品种名称
            min_samples: 最少需要的历史 K 线数（用于特征计算）

        Returns:
            DataFrame，每行一条样本，包含:
            - symbol, entry_ts, entry_price
            - expiry_ts, expiry_price, expiry_minutes
            - direction (1=CALL, 2=PUT), result (WIN/LOSS/TIE)
            - price_source
        """
        if len(df_1m) < min_samples:
            print(f"[LabelBuilder] {symbol}: 数据不足 ({len(df_1m)} < {min_samples})")
            return pd.DataFrame()

        all_samples = []

        for expiry_minutes in self.expiries:
            samples = self._build_for_expiry(df_1m, symbol, expiry_minutes, min_samples)
            all_samples.extend(samples)

        if not all_samples:
            return pd.DataFrame()

        df = pd.DataFrame(all_samples)
        df = df.sort_values("entry_ts").reset_index(drop=True)
        return df

    def _build_for_expiry(
        self,
        df_1m: pd.DataFrame,
        symbol: str,
        expiry_minutes: int,
        min_samples: int,
    ) -> list:
        """
        针对单个 expiry horizon 构建标签。

        对每一根 K 线作为 entry 点，向前看 expiry_minutes 根 K 线作为 expiry 点。
        要求 entry 点之前至少有 min_samples 根 K 线（用于特征计算）。
        """
        samples = []
        close_prices = df_1m["close"].values
        # 统一提取时间戳为毫秒（处理 tz-aware 和 tz-naive）
        if hasattr(df_1m.index, "tz") and df_1m.index.tz is not None:
            timestamps = np.array([int(t.timestamp() * 1000) for t in df_1m.index])
        else:
            timestamps = np.array([int(t.timestamp() * 1000) for t in df_1m.index])

        # 从 min_samples 开始，到倒数 expiry_minutes 根结束
        start_idx = min_samples
        end_idx = len(df_1m) - expiry_minutes

        if end_idx <= start_idx:
            return []

        for i in range(start_idx, end_idx):
            entry_ts = int(timestamps[i]) if isinstance(timestamps[i], (int, float, np.integer, np.floating)) else int(pd.Timestamp(timestamps[i]).timestamp() * 1000)
            entry_price = float(close_prices[i])

            # 到期价格 = entry 后 expiry_minutes 根 K 线的收盘价
            expiry_idx = i + expiry_minutes
            expiry_ts = int(timestamps[expiry_idx]) if isinstance(timestamps[expiry_idx], (int, float, np.integer, np.floating)) else int(pd.Timestamp(timestamps[expiry_idx]).timestamp() * 1000)
            expiry_price = float(close_prices[expiry_idx])

            # ── CALL 标签 ──
            if expiry_price > entry_price:
                call_result = "WIN"
            elif expiry_price < entry_price:
                call_result = "LOSS"
            else:
                call_result = "TIE"

            # ── PUT 标签 ──
            if expiry_price < entry_price:
                put_result = "WIN"
            elif expiry_price > entry_price:
                put_result = "LOSS"
            else:
                put_result = "TIE"

            # 记录 CALL 和 PUT 方向各一条样本
            for direction, direction_int, result in [
                ("CALL", 1, call_result),
                ("PUT", 2, put_result),
            ]:
                samples.append({
                    "symbol": symbol,
                    "entry_ts": entry_ts,
                    "entry_price": round(entry_price, 6),
                    "expiry_ts": expiry_ts,
                    "expiry_price": round(expiry_price, 6),
                    "expiry_minutes": expiry_minutes,
                    "direction": direction,
                    "direction_int": direction_int,
                    "result": result,
                    "price_source": self.price_source,
                    # 用于训练时的标签编码
                    "label_binary": 1 if result == "WIN" else 0,  # WIN=1, LOSS/TIE=0
                    "label_ternary": 0 if result == "LOSS" else (1 if result == "TIE" else 2),  # LOSS=0, TIE=1, WIN=2
                    "is_tie": 1 if result == "TIE" else 0,
                })

        return samples

    def filter_binary_labels(self, df_labels: pd.DataFrame) -> pd.DataFrame:
        """
        排除 TIE 样本，只保留 WIN/LOSS 用于二分类训练。

        返回的 DataFrame 中 label_binary=1 表示 WIN, label_binary=0 表示 LOSS。
        """
        return df_labels[df_labels["result"] != "TIE"].copy()

    def get_stats(self, df_labels: pd.DataFrame) -> List[LabelStats]:
        """获取标签统计信息"""
        stats_list = []
        if df_labels.empty:
            return stats_list

        for expiry in df_labels["expiry_minutes"].unique():
            df_e = df_labels[df_labels["expiry_minutes"] == expiry]

            call_df = df_e[df_e["direction"] == "CALL"]
            put_df = df_e[df_e["direction"] == "PUT"]

            call_wins = (call_df["result"] == "WIN").sum()
            put_wins = (put_df["result"] == "WIN").sum()
            ties = (df_e["result"] == "TIE").sum() // 2  # 除以2因为每个entry点有CALL和PUT两条

            stats = LabelStats(
                symbol=df_e["symbol"].iloc[0] if len(df_e) > 0 else "",
                expiry_minutes=int(expiry),
                total_samples=len(df_e),
                call_samples=len(call_df),
                put_samples=len(put_df),
                tie_samples=ties,
                call_win_rate=round(call_wins / len(call_df[call_df["result"] != "TIE"]), 4) if len(call_df[call_df["result"] != "TIE"]) > 0 else 0.0,
                put_win_rate=round(put_wins / len(put_df[put_df["result"] != "TIE"]), 4) if len(put_df[put_df["result"] != "TIE"]) > 0 else 0.0,
                call_win_count=int(call_wins),
                put_win_count=int(put_wins),
                avg_entry_price=round(df_e["entry_price"].mean(), 4),
                avg_expiry_price=round(df_e["expiry_price"].mean(), 4),
                price_source=self.price_source,
            )
            stats_list.append(stats)

        return stats_list

    def print_stats(self, df_labels: pd.DataFrame):
        """打印标签统计"""
        stats_list = self.get_stats(df_labels)
        print(f"\n{'='*65}")
        print(f"  Event Contract 标签统计 (price_source={self.price_source})")
        print(f"{'='*65}")
        for s in stats_list:
            total_non_tie = s.call_samples + s.put_samples - s.tie_samples * 2
            print(f"\n  {s.symbol} | {s.expiry_minutes}分钟到期:")
            print(f"    总样本: {s.total_samples} (CALL: {s.call_samples}, PUT: {s.put_samples})")
            print(f"    TIE: {s.tie_samples} ({s.tie_samples/max(s.total_samples//2,1)*100:.1f}%)")
            print(f"    CALL 胜率: {s.call_win_rate:.1%} ({s.call_win_count}/{s.call_samples - s.tie_samples})")
            print(f"    PUT 胜率:  {s.put_win_rate:.1%} ({s.put_win_count}/{s.put_samples - s.tie_samples})")
            print(f"    平均入场价: {s.avg_entry_price:.2f}")
            print(f"    平均到期价: {s.avg_expiry_price:.2f}")


def make_labels_from_df(
    df_1m: pd.DataFrame,
    symbol: str = "",
    expiries: Optional[List[int]] = None,
    price_source: str = "gate_io",
    exclude_ties: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    便捷函数：从 1m K 线 DataFrame 构建标签。

    Args:
        df_1m: 1 分钟 OHLCV DataFrame
        symbol: 品种名称
        expiries: 到期期限列表
        price_source: 价格来源
        exclude_ties: 是否排除 TIE 样本

    Returns:
        (all_labels_df, binary_labels_df)
        - all_labels_df: 所有样本（含 TIE）
        - binary_labels_df: 仅 WIN/LOSS（用于二分类训练）
    """
    builder = LabelBuilder(price_source=price_source, expiries=expiries)
    all_labels = builder.build_labels(df_1m, symbol=symbol)
    if all_labels.empty:
        return all_labels, all_labels

    binary_labels = builder.filter_binary_labels(all_labels) if exclude_ties else all_labels
    return all_labels, binary_labels