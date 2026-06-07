#!/usr/bin/env python3
"""
生成轻量模拟LightGBM模型，让引擎能加载预测。

用法: python scripts/generate_mock_models.py [--dir /root/quant_bot]
"""
import argparse
import os
import sys

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import lightgbm as lgb


FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "MACD", "macd_hist_change", "J", "RSI", "rsi_change",
    "BB_Pos", "bb_width", "NATR", "volatility_ratio",
    "ADX", "adx_change", "VWAP_Dist", "volume_ratio",
    "BSP_5", "BSP_15", "BSP_30", "VEV",
    "close_to_ma50", "Macro_Trend", "momentum_3",
    "wick_upper_ratio", "wick_lower_ratio", "body_ratio",
    "CCI", "CHOP", "ROC_5", "OBV_slope_5",
]


def generate_mock_model(symbol: str, seed: int) -> lgb.Booster:
    """生成一个轻量模拟模型，用随机数据训练一棵浅树"""
    np.random.seed(seed)

    # 生成模拟训练数据
    n_samples = 5000
    X = np.random.randn(n_samples, len(FEATURES))
    X = X.astype(np.float64)

    # 制造有意义的标签：价格在布林带低位+超卖 → 涨；高位+超买 → 跌
    bb_pos_idx = FEATURES.index("BB_Pos")
    rsi_idx = FEATURES.index("RSI")
    macd_idx = FEATURES.index("MACD")
    bsp5_idx = FEATURES.index("BSP_5")

    # 简单规则+噪声生成标签
    y_raw = (
        - (X[:, bb_pos_idx] - 0.5) * 1.5      # BB低位倾向于涨
        + (X[:, rsi_idx] - 50) / 100 * 0.8    # RSI超卖倾向于涨
        + X[:, macd_idx] * 3.0                 # MACD正→涨
        + X[:, bsp5_idx] * 1.2                # 资金流入→涨
        + np.random.randn(n_samples) * 0.3    # 噪声
    )
    y = (y_raw > 0).astype(int)  # 二分类

    # 训练一个极简模型
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": seed,
    }
    ds = lgb.Dataset(X, label=y, feature_name=list(FEATURES))
    model = lgb.train(params, ds, num_boost_round=30)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/root/quant_bot", help="模型输出目录")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    symbols = [
        ("BTCUSDT", 42),
        ("ETHUSDT", 123),
        ("SOLUSDT", 777),
    ]

    for sym, seed in symbols:
        path = os.path.join(args.dir, f"{sym.lower()}_model.pkl")
        print(f"Generate {sym} model -> {path}")
        model = generate_mock_model(sym, seed)
        joblib.dump(model, path)
        print(f"  [OK] {sym} 模型完成 ({os.path.getsize(path)//1024}KB)")

    print("\nAll mock models generated!")


if __name__ == "__main__":
    main()