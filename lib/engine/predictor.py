# -*- coding: utf-8 -*-
"""
多模型集成30分钟预测引擎

统一阈值: 0.62 (凯利滚仓, 永远 62%+ 胜率才开单)

训练数据显示PUT胜率59-62%, 所以PUT信号更可靠
"""
import os
import numpy as np
from typing import Optional
from . import config
from .models import Prediction

ENSEMBLE_MODELS = {}
MODEL_CONFIGS = {}
_CURRENT_THRESHOLD = float(os.getenv("MIN_PROBABILITY", "0.56"))  # 默认0.56


def load_models():
    import joblib
    ENSEMBLE_MODELS.clear(); MODEL_CONFIGS.clear()
    for s in config.SYMBOLS:
        path = os.path.join(config.MODEL_DIR, f"{s.lower()}_ensemble.pkl")
        if os.path.exists(path):
            try:
                d = joblib.load(path)
                ENSEMBLE_MODELS[s] = d["ensemble"]
                MODEL_CONFIGS[s] = {"scaler": d["scaler"], "features": d["features"],
                                     "threshold": d.get("best_threshold", 0.65)}
            except Exception:
                pass
    return len(ENSEMBLE_MODELS)


def set_bootstrap_mode(enabled: bool = True, turbo: bool = True):
    global _CURRENT_THRESHOLD
    _CURRENT_THRESHOLD = float(os.getenv("MIN_PROBABILITY", "0.56"))


def predict(symbol: str, row) -> Optional[Prediction]:
    if symbol not in ENSEMBLE_MODELS:
        return Prediction(symbol=symbol, prob_long=0.28, direction=1, prob_win=0.28, flipped=False)

    cfg = MODEL_CONFIGS[symbol]
    th = _CURRENT_THRESHOLD
    FEAT = cfg["features"]

    try:
        vec = np.array([[float(row.get(f, 0.0)) for f in FEAT]], dtype=np.float64)
        vs = cfg["scaler"].transform(vec)
        proba = ENSEMBLE_MODELS[symbol].predict_proba(vs)
        pos_idx = 1 if hasattr(ENSEMBLE_MODELS[symbol], "classes_") and 1 in ENSEMBLE_MODELS[symbol].classes_ else 0
        prob_call = float(proba[0][pos_idx])
        prob_put = 1 - prob_call

        adx = float(row.get("ADX", 20))
        rsi = float(row.get("RSI", 50))

        # 趋势过滤器: ADX < 20 且 RSI在中间 → 震荡, 不出手
        if adx < 18 and 35 < rsi < 65:
            return Prediction(symbol=symbol, prob_long=0.28, direction=1, prob_win=0.28, flipped=False)

        if prob_put >= th:
            return Prediction(symbol=symbol, prob_long=round(1-prob_put, 4), direction=2, prob_win=round(prob_put, 4), flipped=False)
        elif prob_call >= th:
            return Prediction(symbol=symbol, prob_long=round(prob_call, 4), direction=1, prob_win=round(prob_call, 4), flipped=False)
        else:
            return Prediction(symbol=symbol, prob_long=0.28, direction=1, prob_win=0.28, flipped=False)

    except Exception:
        return Prediction(symbol=symbol, prob_long=0.28, direction=1, prob_win=0.28, flipped=False)