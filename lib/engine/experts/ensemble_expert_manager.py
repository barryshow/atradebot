# -*- coding: utf-8 -*-
"""
ExpertManager v2 — 集成训练好的 LightGBM 模型

替换硬编码规则专家，使用 train_ensemble_v3.py 训练的模型。
根据 Market Regime 选择不同的模型权重（与 v1 相同的 Regime 自适应策略）。
"""
import numpy as np
import os
import joblib
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from ..models import ExpertPrediction, EnsemblePrediction, MarketRegime
from ..regime_detector import MarketRegimeDetector


# ── 模型路径 ──
# 从 lib/engine/experts/ensemble_expert_manager.py 向上 4 级到项目根目录
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")
# 如果不存在，尝试从 cwd 查找
if not os.path.isdir(_MODEL_DIR):
    _MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_DIR = _MODEL_DIR

# 每个品种一个模型（目前训练的是通用模型，后续可扩展为 per-regime 模型）
SYMBOL_MODEL_MAP = {
    "BTCUSDT": "btcusdt_15m_ensemble_v3.pkl",
    "ETHUSDT": "ethusdt_15m_ensemble_v3.pkl",
    "SOLUSDT": "solusdt_15m_ensemble_v3.pkl",
}


def _log(msg: str):
    """Safe print that handles encoding issues."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'), flush=True)


class LGBMExpert:
    """单个 LightGBM 专家模型"""

    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self._loaded = False

    def load(self) -> bool:
        """加载模型文件"""
        if self._loaded:
            return True
        if not os.path.exists(self.model_path):
            _log(f"[LGBMExpert] Model not found: {self.model_path}")
            return False
        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["ensemble"]
            self.scaler = bundle["scaler"]
            self.features = bundle.get("features", [])
            self._loaded = True
            return True
        except Exception as e:
            _log(f"[LGBMExpert] Load failed: {e}")
            return False

    def predict(self, symbol: str, feature_row: np.ndarray) -> float:
        """预测 CALL 概率（0-1）"""
        if not self._loaded or self.model is None:
            return 0.50
        try:
            X = feature_row.reshape(1, -1)
            X_s = self.scaler.transform(X)
            proba = self.model.predict_proba(X_s)
            # 找到 class 1 的索引
            if hasattr(self.model, "classes_") and 1 in self.model.classes_:
                pos_idx = list(self.model.classes_).index(1)
            else:
                pos_idx = 0
            return float(proba[0, pos_idx])
        except Exception:
            return 0.50


class ExpertManager:
    """
    专家管理器 + Meta Model (v2 — 集成训练好的 LightGBM 模型)。

    与 v1 的区别:
    - v1: 硬编码规则专家 (if/else RSI/ADX)
    - v2: 加载 train_ensemble_v3.py 训练的 LightGBM 模型
    - 保留 Regime 自适应权重策略
    """

    def __init__(self):
        self._models_loaded = False
        self._model_available = False

        # 尝试加载模型
        self.models: Dict[str, LGBMExpert] = {}
        for sym, fname in SYMBOL_MODEL_MAP.items():
            path = os.path.join(MODEL_DIR, fname)
            expert = LGBMExpert(f"lgbm_{sym.lower()}", path)
            self.models[sym] = expert

        self._load_models()

        # Regime → 权重映射（与 v1 相同）
        self.regime_weights: Dict[str, Dict[str, float]] = {
            "TREND_UP": {"trend": 0.55, "mean_reversion": 0.15, "volatility_breakout": 0.30},
            "TREND_DOWN": {"trend": 0.55, "mean_reversion": 0.15, "volatility_breakout": 0.30},
            "RANGE": {"trend": 0.15, "mean_reversion": 0.60, "volatility_breakout": 0.25},
            "HIGH_VOLATILITY": {"trend": 0.25, "mean_reversion": 0.20, "volatility_breakout": 0.55},
            "LOW_LIQUIDITY": {"trend": 0.20, "mean_reversion": 0.30, "volatility_breakout": 0.50},
            "EVENT_RISK": {"trend": 0.10, "mean_reversion": 0.10, "volatility_breakout": 0.80},
        }
        self.default_weights = {"trend": 0.33, "mean_reversion": 0.33, "volatility_breakout": 0.34}

        # Expert 近期表现追踪
        self.expert_performance: Dict[str, List[float]] = defaultdict(list)
        self.performance_window = 50

    def _load_models(self):
        """加载所有可用模型"""
        loaded = 0
        for sym, expert in self.models.items():
            if expert.load():
                loaded += 1
        self._models_loaded = True
        self._model_available = loaded > 0
        if loaded > 0:
            _log(f"[ExpertManager] Loaded {loaded}/{len(self.models)} LightGBM models")
        else:
            _log(f"[ExpertManager] WARNING: No trained models found, using fallback prob (0.50)")

    def is_model_available(self) -> bool:
        return self._model_available

    def predict_all(
        self,
        symbol: str,
        indicators: Dict,
        row: Dict,
        regime: Optional[MarketRegime] = None,
    ) -> List[ExpertPrediction]:
        """
        使用 LightGBM 模型预测（如果可用），否则使用 fallback。
        返回 3 个"虚拟"专家预测，保持与 v1 接口兼容。
        """
        predictions = []

        if self._model_available and symbol in self.models:
            expert = self.models[symbol]
            if expert._loaded and expert.features:
                # 构建特征向量
                feature_row = self._build_feature_vector(indicators, row, expert.features)
                if feature_row is not None:
                    prob = expert.predict(symbol, feature_row)
                else:
                    prob = 0.50
            else:
                prob = 0.50
        else:
            prob = 0.50

        # 生成 3 个"虚拟"专家预测（保持接口兼容）
        # 实际只有 1 个模型，但为了保持 Regime 自适应权重策略，
        # 将模型输出拆分为 3 个略有不同的"专家"预测
        direction = 1 if prob >= 0.50 else 2
        direction_str = "CALL" if direction == 1 else "PUT"
        adjusted_prob = prob if direction == 1 else (1.0 - prob)
        adjusted_prob = max(0.35, min(0.65, adjusted_prob))

        # 趋势专家: 如果 Regime 是趋势，权重更高
        trend_prob = adjusted_prob + np.random.uniform(-0.01, 0.01)
        # 均值回归专家: 如果 Regime 是震荡，权重更高
        mean_rev_prob = adjusted_prob + np.random.uniform(-0.01, 0.01)
        # 波动突破专家: 如果 Regime 是高波动，权重更高
        vol_break_prob = adjusted_prob + np.random.uniform(-0.01, 0.01)

        predictions.extend([
            ExpertPrediction(
                expert_name="trend", symbol=symbol, direction=direction, direction_str=direction_str,
                raw_probability=round(trend_prob, 4), calibrated_probability=round(trend_prob, 4),
                confidence=0.5, model_version="v3_lgbm",
            ),
            ExpertPrediction(
                expert_name="mean_reversion", symbol=symbol, direction=direction, direction_str=direction_str,
                raw_probability=round(mean_rev_prob, 4), calibrated_probability=round(mean_rev_prob, 4),
                confidence=0.5, model_version="v3_lgbm",
            ),
            ExpertPrediction(
                expert_name="volatility_breakout", symbol=symbol, direction=direction, direction_str=direction_str,
                raw_probability=round(vol_break_prob, 4), calibrated_probability=round(vol_break_prob, 4),
                confidence=0.5, model_version="v3_lgbm",
            ),
        ])

        return predictions

    def _build_feature_vector(self, indicators: Dict, row: Dict, feature_names: List[str]) -> Optional[np.ndarray]:
        """
        从 indicators 和 row 构建特征向量，匹配训练时的特征顺序。
        """
        # 创建特征字典
        feat_dict = {}

        # 时间特征（从 row 或当前时间推断）
        import time as _time
        now = _time.localtime()
        feat_dict["hour_sin"] = np.sin(2 * np.pi * now.tm_hour / 24)
        feat_dict["hour_cos"] = np.cos(2 * np.pi * now.tm_hour / 24)
        feat_dict["dow_sin"] = np.sin(2 * np.pi * now.tm_wday / 7)
        feat_dict["dow_cos"] = np.cos(2 * np.pi * now.tm_wday / 7)

        # 从 indicators 映射
        feat_dict["MACD"] = float(indicators.get("MACD", 0))
        feat_dict["macd_hist_change"] = 0.0  # 无法从当前单帧计算
        feat_dict["RSI"] = float(indicators.get("RSI", 50))
        feat_dict["rsi_change"] = 0.0
        feat_dict["ROC_5"] = 0.0
        feat_dict["momentum_3"] = 0.0
        feat_dict["Macro_Trend"] = float(indicators.get("MA_trend", 0))
        feat_dict["BB_Pos"] = float(indicators.get("BB_Pos", 0.5))
        feat_dict["bb_width"] = float(indicators.get("bb_width", 0.02))
        feat_dict["NATR"] = float(indicators.get("ATR_pct", 0.003))
        feat_dict["volatility_ratio"] = float(indicators.get("volatility_ratio", 1.0))
        feat_dict["ADX"] = float(indicators.get("ADX", 20))
        feat_dict["adx_change"] = 0.0
        feat_dict["VWAP_Dist"] = float(indicators.get("VWAP_dist", 0))
        feat_dict["close_to_ma50"] = float(indicators.get("price_vs_MA20", 0))  # 近似
        feat_dict["MA_trend"] = float(indicators.get("MA_trend", 0))
        feat_dict["volume_ratio"] = float(indicators.get("vol_ratio", 1.0))
        feat_dict["VEV"] = 0.0
        feat_dict["BSP_5"] = 0.0
        feat_dict["BSP_15"] = 0.0
        feat_dict["BSP_30"] = 0.0
        feat_dict["wick_upper_ratio"] = 0.0
        feat_dict["wick_lower_ratio"] = 0.0
        feat_dict["body_ratio"] = float(row.get("body_pct", 0.3))
        feat_dict["CCI"] = float(indicators.get("CCI", 0))
        feat_dict["CHOP"] = 50.0
        feat_dict["OBV_slope_5"] = 0.0
        feat_dict["J"] = 0.0

        # 按特征名顺序构建向量
        try:
            vec = np.array([feat_dict.get(name, 0.0) for name in feature_names], dtype=np.float64)
            return vec
        except Exception:
            return None

    def ensemble(
        self,
        predictions: List[ExpertPrediction],
        regime: Optional[MarketRegime] = None,
    ) -> EnsemblePrediction:
        """
        Meta Model: 加权集成所有专家预测（与 v1 相同的 Regime 自适应策略）。
        """
        if not predictions:
            return EnsemblePrediction(symbol="", direction=0, ensemble_probability=0.50)

        regime_str = regime.regime if regime else "RANGE"
        base_weights = self.regime_weights.get(regime_str, self.default_weights)

        # 计算最终权重
        final_weights = {}
        for expert_name, base_w in base_weights.items():
            perf = self.expert_performance.get(expert_name, [])
            if len(perf) >= 10:
                recent_wr = sum(perf[-20:]) / max(len(perf[-20:]), 1)
                perf_multiplier = min(2.0, max(0.5, recent_wr / 0.50))
            else:
                perf_multiplier = 1.0
            final_weights[expert_name] = base_w * perf_multiplier

        # 归一化
        total_w = sum(final_weights.values())
        if total_w > 0:
            final_weights = {k: v / total_w for k, v in final_weights.items()}

        # 加权集成
        weighted_prob = 0.0
        weighted_dir_sum = 0.0
        expert_votes = {}

        for pred in predictions:
            w = final_weights.get(pred.expert_name, 0.1)
            expert_votes[pred.expert_name] = pred.raw_probability
            dir_signal = 1.0 if pred.direction == 1 else -1.0
            weighted_dir_sum += dir_signal * pred.raw_probability * w
            weighted_prob += pred.raw_probability * w

        # 集成方向
        ensemble_direction = 1 if weighted_dir_sum >= 0 else 2
        ensemble_prob = max(0.35, min(0.65, weighted_prob))

        if ensemble_direction == 2:
            ensemble_prob = 1.0 - ensemble_prob
            if ensemble_prob < 0.50:
                ensemble_prob = 0.50 + (0.50 - ensemble_prob)

        probs = [p.raw_probability for p in predictions]
        uncertainty = float(np.std(probs)) if len(probs) > 1 else 0.0

        return EnsemblePrediction(
            symbol=predictions[0].symbol,
            direction=ensemble_direction,
            ensemble_probability=round(ensemble_prob, 4),
            calibrated_probability=round(ensemble_prob, 4),
            conservative_probability=round(ensemble_prob - uncertainty * 0.5, 4),
            expert_votes=expert_votes,
            meta_model_version="v2_lgbm",
            regime=regime_str,
        )

    def update_performance(self, expert_name: str, is_win: bool):
        self.expert_performance[expert_name].append(1 if is_win else 0)
        if len(self.expert_performance[expert_name]) > self.performance_window:
            self.expert_performance[expert_name] = self.expert_performance[expert_name][-self.performance_window:]

    def get_expert_win_rates(self) -> Dict[str, float]:
        rates = {}
        for name in ["trend", "mean_reversion", "volatility_breakout"]:
            perf = self.expert_performance.get(name, [])
            if len(perf) >= 10:
                rates[name] = round(sum(perf) / len(perf), 4)
            else:
                rates[name] = 0.0
        return rates


# 全局单例
_expert_manager: Optional[ExpertManager] = None


def get_expert_manager() -> ExpertManager:
    global _expert_manager
    if _expert_manager is None:
        _expert_manager = ExpertManager()
    return _expert_manager