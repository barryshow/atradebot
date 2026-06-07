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
    symbol: str
    prob_long: float
    direction: int  # 1=CALL, 2=PUT
    prob_win: float
    flipped: bool = False


@dataclass
class TradeDecision:
    symbol: str
    direction: int
    dir_str: str
    entry_price: float
    ml_prob: float
    ai_approval: float
    ai_reason: str
    risk_gates: list
    flipped: bool
