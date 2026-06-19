# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskGateResult:
    level: int
    name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class Prediction:
    """原始模型预测（来自 predictor.py）"""
    symbol: str
    prob_long: float
    direction: int  # 1=CALL, 2=PUT
    prob_win: float
    flipped: bool = False


@dataclass
class TradeSignal:
    """经过 SignalValidator 验证后的交易信号"""
    symbol: str
    direction: int          # 1=CALL, 2=PUT
    dir_str: str            # 用于日志和通知
    prob_win: float         # 可能是模型原始胜率，也可能是翻转后重置的0.55
    original_prob: float    # 模型原始胜率（用于日志对比）
    is_reversal: bool       # 是否经过了极值翻转
    action: str             # "open" | "close_and_open" | "close" (add removed — 二元期权每单独立)
    entry_price: float      # 当前价格
    indicators: dict        # 指标快照
    confluence: float       # 共振分
    row: dict               # 特征行（传递给 OrderExecutor 用于 PnL 计算）
    flip_note: str = ""     # 翻转说明


@dataclass
class PositionState:
    """品种当前持仓状态"""
    symbol: str
    direction: int          # 1=CALL, 2=PUT  (0=无持仓)
    amount: float           # 投入金额
    entry_price: float      # 开仓价
    open_time_ms: int       # 开仓时间戳
    entry_bar_ts: object    # 开仓时的 K 线时间戳
    unrealized_pnl: float = 0.0      # 未实现盈亏
    unrealized_roi: float = 0.0      # 未实现收益率
    pending_close: bool = False      # 等待平仓（反向信号触发）


@dataclass
class GateResult:
    """单道风控门结果"""
    level: int
    name: str
    passed: bool
    reason: str
