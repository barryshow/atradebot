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
    action: str             # "open" | "close_and_open" | "close"
    entry_price: float      # 当前价格
    indicators: dict        # 指标快照
    confluence: float       # 共振分
    row: dict               # 特征行
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
    unrealized_pnl: float = 0.0
    unrealized_roi: float = 0.0
    pending_close: bool = False


@dataclass
class GateResult:
    """单道风控门结果"""
    level: int
    name: str
    passed: bool
    reason: str


# ═══════════════════════════════════════════════════════════
# EventEdge V2 新增数据类
# ═══════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    """市场状态分类"""
    regime: str = ""                        # "TREND_UP" / "TREND_DOWN" / "RANGE" / "HIGH_VOLATILITY" / "LOW_LIQUIDITY" / "EVENT_RISK"
    confidence: float = 0.0                 # 分类置信度
    atr: float = 0.0
    adx: float = 0.0
    volatility: float = 0.0
    ema_slope: float = 0.0
    bb_width: float = 0.0
    volume_zscore: float = 0.0
    spread: float = 0.0
    order_book_imbalance: float = 0.0       # 如果有 order book 数据
    details: dict = field(default_factory=dict)


@dataclass
class ExpertPrediction:
    """单个 Expert Model 的输出"""
    expert_name: str = ""                   # "trend" / "mean_reversion" / "order_flow" / "volatility_breakout"
    symbol: str = ""
    direction: int = 0                      # 1=CALL, 2=PUT
    direction_str: str = ""                 # "CALL" / "PUT"
    raw_probability: float = 0.0            # 专家原始概率
    calibrated_probability: float = 0.0     # 校准后概率
    confidence: float = 0.0                 # 专家自身置信度
    features_used: list = field(default_factory=list)
    model_version: str = ""


@dataclass
class EnsemblePrediction:
    """Meta Model 输出"""
    symbol: str = ""
    direction: int = 0                      # 1=CALL, 2=PUT
    ensemble_probability: float = 0.0       # Meta Model 集成概率
    calibrated_probability: float = 0.0     # 校准后概率
    conservative_probability: float = 0.0   # 保守概率（扣除所有 margin）
    expert_votes: dict = field(default_factory=dict)  # {expert: prob}
    meta_model_version: str = ""
    regime: str = ""


@dataclass
class EdgeResult:
    """Edge 计算结果"""
    symbol: str = ""
    expiry_minutes: int = 15
    direction: str = ""                     # "CALL" / "PUT"
    direction_int: int = 0
    entry_price: float = 0.0

    # 概率
    calibrated_probability: float = 0.0
    conservative_probability: float = 0.0

    # 赔付率
    payout_ratio: float = 0.0               # 总返还比例 (含本金)
    net_payout_ratio: float = 0.0           # 净盈利比例
    payout_source: str = ""                 # "api" / "hardcoded"
    payout_verified: bool = False           # 仅 API 返回的 payout 才是 verified
    payout_flag: str = ""                   # "CONFIG_ASSUMED" | "VERIFIED_HIBT"

    # Edge
    break_even_probability: float = 0.0     # 盈亏平衡概率
    probability_edge: float = 0.0           # 概率优势
    raw_edge: float = 0.0                   # 原始优势
    effective_edge: float = 0.0             # 有效优势 (penalty 后)
    expected_roi: float = 0.0               # 每 1U 期望 ROI
    edge_flag: str = ""                     # "SIMULATED_EDGE" | "VERIFIED_EDGE"

    # Margin
    uncertainty_margin: float = 0.0
    calibration_margin: float = 0.0
    model_degradation_margin: float = 0.0
    sample_uncertainty_margin: float = 0.0

    # 判定
    passed: bool = False                    # 是否通过 Edge 门槛
    reject_reason: str = ""                 # 未通过原因

    # 元数据
    regime: str = ""
    expert_votes: dict = field(default_factory=dict)


@dataclass
class Opportunity:
    """候选交易机会（用于排序）"""
    symbol: str = ""
    expiry_minutes: int = 15
    direction: str = ""
    direction_int: int = 0
    calibrated_probability: float = 0.0
    break_even_probability: float = 0.0
    effective_edge: float = 0.0
    expected_roi: float = 0.0
    uncertainty: float = 0.0
    regime: str = ""
    rank_score: float = 0.0                 # 排序得分
    selected: bool = False                  # 是否被选中
    risk_adjusted_ev: float = 0.0           # 风险调整后期望值


@dataclass
class TrainLabel:
    """Event Contract 真实训练标签"""
    symbol: str = ""
    entry_timestamp: int = 0                # 入场时间戳 (ms)
    entry_reference_price: float = 0.0      # 入场参考价格
    expiry_timestamp: int = 0               # 到期时间戳 (ms)
    expiry_reference_price: float = 0.0     # 到期参考价格
    expiry_minutes: int = 15                # 期限
    direction: int = 0                      # 1=CALL, 2=PUT
    result: str = ""                        # "WIN" / "LOSS" / "TIE"
    payout_ratio: float = 0.0               # 实际赔付率
    stake: float = 0.0                      # 下注金额
    realized_pnl: float = 0.0               # 已实现盈亏
    price_source: str = ""                  # "hibt_official" / "gate_io" / "binance"


@dataclass
class ModelHealthReport:
    """模型健康报告"""
    window: int = 0                         # 窗口大小 (50/100/500)
    trade_count: int = 0
    actual_win_rate: float = 0.0
    predicted_win_rate: float = 0.0         # 平均预测概率
    win_rate_delta: float = 0.0             # 实际 - 预测
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0
    ev: float = 0.0
    actual_pnl: float = 0.0
    roi: float = 0.0
    max_drawdown: float = 0.0
    is_degraded: bool = False
    degradation_reason: str = ""
