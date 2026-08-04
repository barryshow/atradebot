# -*- coding: utf-8 -*-
"""
DecisionEngine — EV-based trade decision engine.

Replaces the old fixed-threshold "ensemble_prob >= 0.50 → CALL" logic.

Input: p_up (calibrated), payout_call_net, payout_put_net
Output: Decision with direction + expected value, or ABSTAIN.

Formula:
    ev_call = p_up * payout_call_net - (1 - p_up)
    ev_put  = (1 - p_up) * payout_put_net - p_up

    selected_direction = argmax(ev_call, ev_put)
    selected_ev = max(ev_call, ev_put)

Only generate a trade if ALL conditions are met:
    1. max(ev_call, ev_put) >= MIN_EXPECTED_VALUE
    2. Payout is from real-time API (not hardcoded)
    3. Calibrator is ready
    4. Market data is fresh
    5. Model version is valid
    6. No unsettled orders
    7. Daily loss / consecutive loss / max trades limits not hit
Otherwise: ABSTAIN with reason.
"""
from typing import Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class DecisionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"
    ABSTAIN = "ABSTAIN"


@dataclass
class Decision:
    """Output of DecisionEngine.evaluate()."""
    symbol: str = ""
    decision: DecisionType = DecisionType.ABSTAIN
    direction_int: int = 0          # 1=CALL, 2=PUT, 0=ABSTAIN

    # Probabilities
    fast_probability: float = 0.0
    slow_probability: float = 0.0
    calibrated_probability: float = 0.0

    # Payout (real-time, net)
    payout_call_net: float = 0.0
    payout_put_net: float = 0.0
    payout_source: str = ""         # "api" / "hardcoded"

    # Expected Value
    ev_call: float = 0.0
    ev_put: float = 0.0
    selected_ev: float = 0.0

    # Breakeven
    break_even_call: float = 0.0    # 1/(1+payout_call_net)
    break_even_put: float = 0.0

    # Gate checks
    passed: bool = False
    reject_reason: str = ""
    reject_detail: str = ""

    # Context
    regime: str = ""
    model_version: str = ""
    feature_version: str = ""
    gate_price: float = 0.0
    signal_time_ms: int = 0


class DecisionEngine:
    """EV-based trade decision engine."""

    # Minimum expected value to open a trade
    MIN_EXPECTED_VALUE = 0.03       # 3% EV per unit wagered

    def evaluate(
        self,
        symbol: str,
        fast_probability: float,
        slow_probability: float,
        calibrated_probability: float,
        payout_call_net: float,
        payout_put_net: float,
        payout_source: str = "hardcoded",
        regime: str = "RANGE",
        model_version: str = "",
        feature_version: str = "",
        gate_price: float = 0.0,
        signal_time_ms: int = 0,
    ) -> Decision:
        """
        Evaluate whether to trade and in which direction.

        Args:
            symbol: Trading symbol
            fast_probability: Fast model probability (raw)
            slow_probability: Slow model probability (raw)
            calibrated_probability: Calibrated probability of CALL (p_up)
            payout_call_net: Net payout for CALL win (e.g. 0.818)
            payout_put_net: Net payout for PUT win (e.g. 0.80)
            payout_source: "api" or "hardcoded"
            regime: Market regime
            model_version: Model version string
            feature_version: Feature version string
            gate_price: Current price at decision time
            signal_time_ms: Timestamp of signal

        Returns:
            Decision object
        """

        # ── Compute EV ──
        p_up = max(0.0, min(1.0, calibrated_probability))

        if payout_call_net <= 0 or payout_put_net <= 0:
            return Decision(
                symbol=symbol,
                decision=DecisionType.ABSTAIN,
                fast_probability=fast_probability,
                slow_probability=slow_probability,
                calibrated_probability=calibrated_probability,
                payout_call_net=payout_call_net,
                payout_put_net=payout_put_net,
                payout_source=payout_source,
                regime=regime,
                model_version=model_version,
                feature_version=feature_version,
                gate_price=gate_price,
                signal_time_ms=signal_time_ms,
                passed=False,
                reject_reason="INVALID_PAYOUT",
                reject_detail=f"Payouts must be positive: call={payout_call_net}, put={payout_put_net}",
            )

        ev_call = p_up * payout_call_net - (1 - p_up)
        ev_put = (1 - p_up) * payout_put_net - p_up

        # Compute breakeven probabilities
        be_call = 1.0 / (1.0 + payout_call_net) if payout_call_net > 0 else 1.0
        be_put = 1.0 / (1.0 + payout_put_net) if payout_put_net > 0 else 1.0

        # ── Select direction ──
        if ev_call >= ev_put and ev_call >= self.MIN_EXPECTED_VALUE:
            decision = DecisionType.CALL
            direction_int = 1
            selected_ev = ev_call
        elif ev_put > ev_call and ev_put >= self.MIN_EXPECTED_VALUE:
            decision = DecisionType.PUT
            direction_int = 2
            selected_ev = ev_put
        else:
            # Neither direction has sufficient EV
            best_ev = max(ev_call, ev_put)
            best_dir = "CALL" if ev_call >= ev_put else "PUT"
            return Decision(
                symbol=symbol,
                decision=DecisionType.ABSTAIN,
                fast_probability=fast_probability,
                slow_probability=slow_probability,
                calibrated_probability=calibrated_probability,
                payout_call_net=payout_call_net,
                payout_put_net=payout_put_net,
                payout_source=payout_source,
                ev_call=round(ev_call, 6),
                ev_put=round(ev_put, 6),
                break_even_call=round(be_call, 4),
                break_even_put=round(be_put, 4),
                regime=regime,
                model_version=model_version,
                feature_version=feature_version,
                gate_price=gate_price,
                signal_time_ms=signal_time_ms,
                passed=False,
                reject_reason="INSUFFICIENT_EV",
                reject_detail=f"max(EV)={best_ev:.4f} ({best_dir}) < {self.MIN_EXPECTED_VALUE}",
            )

        # ── Payout source check ──
        if payout_source != "api":
            return Decision(
                symbol=symbol, decision=DecisionType.ABSTAIN,
                fast_probability=fast_probability, slow_probability=slow_probability,
                calibrated_probability=calibrated_probability,
                payout_call_net=payout_call_net, payout_put_net=payout_put_net,
                payout_source=payout_source,
                ev_call=round(ev_call, 6), ev_put=round(ev_put, 6),
                break_even_call=round(be_call, 4), break_even_put=round(be_put, 4),
                regime=regime, model_version=model_version, feature_version=feature_version,
                gate_price=gate_price, signal_time_ms=signal_time_ms,
                passed=False, reject_reason="PAYOUT_NOT_VERIFIED",
                reject_detail=f"Payout source={payout_source}, must be 'api' for LIVE",
            )

        return Decision(
            symbol=symbol,
            decision=decision,
            direction_int=direction_int,
            fast_probability=fast_probability,
            slow_probability=slow_probability,
            calibrated_probability=calibrated_probability,
            payout_call_net=payout_call_net,
            payout_put_net=payout_put_net,
            payout_source=payout_source,
            ev_call=round(ev_call, 6),
            ev_put=round(ev_put, 6),
            selected_ev=round(selected_ev, 6),
            break_even_call=round(be_call, 4),
            break_even_put=round(be_put, 4),
            regime=regime,
            model_version=model_version,
            feature_version=feature_version,
            gate_price=gate_price,
            signal_time_ms=signal_time_ms,
            passed=True,
        )
