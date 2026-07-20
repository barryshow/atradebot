#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EventEdge V2 Walk-Forward Backtester

严格时间序列 Walk-Forward 回测框架。

核心原则:
1. 禁止随机 Train/Test Split — 使用时间序列窗口滚动
2. 整数下注规则: ≥3U, step=1, 不使用小数
3. 支持多 expiry horizon 独立回测
4. 多维度统计: by symbol, by expiry, by regime, by direction, by probability_bucket, by edge_bucket
5. 验证: effective_edge 越高 → 真实 ROI 是否越高
6. 回测与实盘使用完全相同的整数下注规则

用法:
    from lib.engine.backtester import WalkForwardBacktester

    bt = WalkForwardBacktester(
        symbol="BTCUSDT",
        expiries=[15],
        train_window_days=30,
        test_window_days=7,
        step_days=7,
    )
    report = bt.run(df_1m)
    bt.print_report(report)
"""
import numpy as np
import pandas as pd
import time
import math
import os
import sys
import json
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.engine.label_builder import LabelBuilder
from lib.engine.models import TrainLabel


# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    """单笔回测交易"""
    trade_id: int = 0
    symbol: str = ""
    expiry_minutes: int = 15
    direction: str = ""
    direction_int: int = 0
    entry_time: str = ""
    entry_price: float = 0.0
    expiry_time: str = ""
    expiry_price: float = 0.0
    stake_usd: int = 3
    raw_probability: float = 0.0
    calibrated_probability: float = 0.0
    break_even_probability: float = 0.0
    effective_edge: float = 0.0
    expected_roi: float = 0.0
    net_payout_ratio: float = 0.80
    result: str = ""                     # "WIN" / "LOSS" / "TIE"
    realized_pnl: float = 0.0
    regime: str = ""
    model_version: str = ""
    test_window: int = 0


@dataclass
class WindowResult:
    """单个 Walk-Forward 窗口结果"""
    window_id: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    train_samples: int = 0
    test_samples: int = 0
    trades: List[BacktestTrade] = field(default_factory=list)
    # 统计
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    win_rate: float = 0.0
    break_even_win_rate: float = 0.0
    total_pnl: float = 0.0
    total_staked: int = 0
    roi: float = 0.0
    avg_effective_edge: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    max_drawdown: float = 0.0
    model_path: str = ""


@dataclass
class BacktestReport:
    """完整回测报告"""
    symbol: str = ""
    expiry_minutes: int = 15
    # 全局
    total_windows: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_ties: int = 0
    overall_win_rate: float = 0.0
    break_even_win_rate: float = 0.0
    total_pnl: float = 0.0
    total_staked: int = 0
    overall_roi: float = 0.0
    avg_expected_roi: float = 0.0
    avg_effective_edge: float = 0.0
    overall_brier_score: float = 0.0
    overall_log_loss: float = 0.0
    max_drawdown: float = 0.0
    longest_losing_streak: int = 0
    longest_winning_streak: int = 0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    # 分维度统计
    by_direction: dict = field(default_factory=dict)
    by_probability_bucket: dict = field(default_factory=dict)
    by_edge_bucket: dict = field(default_factory=dict)
    by_regime: dict = field(default_factory=dict)
    # 窗口
    windows: List[WindowResult] = field(default_factory=list)
    # 交易
    all_trades: List[BacktestTrade] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# Walk-Forward Backtester
# ═══════════════════════════════════════════════════════════

class WalkForwardBacktester:
    """
    Walk-Forward 回测器。

    工作流程:
    1. 加载数据 → 计算特征 → 构建标签
    2. 按时间窗口滚动:
       - Train on [t0, t1]
       - Test on [t1, t2]
       - 窗口向前滚动 step_days
    3. 每个窗口: 训练模型 → 对测试集做预测 → 模拟交易 → 记录结果
    4. 汇总所有窗口结果
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        expiries: Optional[List[int]] = None,
        train_window_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 7,
        min_history: int = 200,
        min_samples_train: int = 500,
        # 整数下注
        min_order_usd: int = 3,
        order_step: int = 1,
        # 交易参数
        net_payout_ratio: float = 0.80,
        min_probability: float = 0.50,       # 最低预测概率阈值
        min_effective_edge: float = 0.0,      # 最低有效优势（回测时可设0）
        # 凯利
        kelly_fraction: float = 0.10,
        max_bet_fraction: float = 0.01,
        # 输出
        output_dir: str = "./backtest_results",
        verbose: bool = True,
    ):
        self.symbol = symbol
        self.expiries = expiries or [15]
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.min_history = min_history
        self.min_samples_train = min_samples_train
        self.min_order_usd = min_order_usd
        self.order_step = order_step
        self.net_payout_ratio = net_payout_ratio
        self.min_probability = min_probability
        self.min_effective_edge = min_effective_edge
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.output_dir = output_dir
        self.verbose = verbose

        # 盈亏平衡概率
        self.break_even_probability = 1.0 / (1.0 + self.net_payout_ratio)

        os.makedirs(self.output_dir, exist_ok=True)

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def _compute_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """计算特征 — 先重采样到特征粒度（15m）再计算"""
        from scripts.train_ensemble_v2 import calc_features

        # 重采样到 15m（与标签构建的 expiry 粒度一致）
        feature_minutes = self.expiries[0]  # 使用第一个 expiry 作为特征粒度
        resampled = df_1m.resample(f"{feature_minutes}min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        if len(resampled) < 50:
            self._log(f"    重采样后数据不足 ({len(resampled)} < 50)")
            return pd.DataFrame()

        return calc_features(resampled)

    def _build_labels(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """构建标签 — 在重采样数据上构建"""
        builder = LabelBuilder(price_source="gate_io", expiries=[1])  # 1 根 K 线 = 15m 到期
        labels = builder.build_labels(df_1m, symbol=self.symbol, min_samples=self.min_history)
        if labels.empty:
            return labels
        return builder.filter_binary_labels(labels)

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        window_id: int,
    ) -> Tuple[object, object, np.ndarray]:
        """训练 LightGBM 模型（针对小数据集优化）"""
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()

        # 小数据集: 降低正则化，减少 min_child_samples
        n_samples = len(X_train)
        if n_samples < 5000:
            model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.03,
                num_leaves=15, min_child_samples=5,
                subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.0, reg_lambda=0.0,
                objective="binary",
                random_state=42, verbosity=-1,
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                objective="binary",
                random_state=42, verbosity=-1,
            )

        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            eval_metric="auc",
            callbacks=[lgb.log_evaluation(0)],
        )

        # 预测概率
        pos_idx = 1 if hasattr(model, "classes_") and 1 in model.classes_ else 0
        y_prob = model.predict_proba(X_test_s)[:, pos_idx]

        # 诊断：检查概率分布是否坍缩
        prob_mean = float(np.mean(y_prob))
        prob_std = float(np.std(y_prob))
        if prob_std < 0.01:
            self._log(f"    ⚠ 概率分布坍缩 (mean={prob_mean:.3f}, std={prob_std:.4f}) — 数据可能不足以学习")

        return model, scaler, y_prob

    def _simulate_trades(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        aligned_df: pd.DataFrame,
        window_id: int,
        equity: float = 100.0,
        expiry_minutes: int = 15,
    ) -> List[BacktestTrade]:
        """
        模拟交易，严格遵循整数下注规则。

        规则:
        1. 预测概率 >= min_probability 才考虑
        2. 计算 effective_edge = prob - break_even_prob
        3. effective_edge >= min_effective_edge 才下单
        4. Integer Kelly: frac_kelly = effective_edge / (1 + net_payout)
           → target = equity * kelly_fraction * frac_kelly
           → capped at max_bet_fraction
           → floored to integer
           → min 3U check
        5. TIE 单独处理 (true_label == 2 or both CALL/PUT lose)
        """
        trades = []
        from scripts.train_ensemble_v2 import FEATURES  # noqa: F811

        for i in range(len(y_prob)):
            prob = float(y_prob[i])
            true_label = int(y_true[i])

            # 概率阈值
            if prob < self.min_probability:
                continue

            # 考虑 CALL 和 PUT 两个方向
            for direction, direction_int, dir_prob in [
                ("CALL", 1, prob),
                ("PUT", 2, 1.0 - prob),
            ]:
                if dir_prob < self.min_probability:
                    continue

                # ── Edge 计算 ──
                effective_edge = dir_prob - self.break_even_probability
                if effective_edge < self.min_effective_edge:
                    continue

                expected_roi = dir_prob * self.net_payout_ratio - (1.0 - dir_prob)

                # ── Integer Kelly 仓位 ──
                # stake = MIN_ORDER_USD + floor(equity * kelly * edge / (1 + payout))
                # 这确保即使 edge 很小，最低也下 3U（如果 edge 通过门槛）
                denom = 1.0 + self.net_payout_ratio
                frac_kelly = effective_edge / denom if denom > 0 else 0.0
                target_fraction = self.kelly_fraction * frac_kelly
                effective_fraction = min(target_fraction, self.max_bet_fraction)

                # 基于 equity 的增量部分
                incremental = equity * effective_fraction
                stake_usd = self.min_order_usd + int(math.floor(incremental))
                stake_usd = (stake_usd // self.order_step) * self.order_step

                # 最小订单检查 — 不允许自动提高到 3U
                if stake_usd < self.min_order_usd:
                    continue

                # 获取对齐数据
                row = aligned_df.iloc[i] if i < len(aligned_df) else None

                # ── 结算结果 ──
                if direction == "CALL":
                    if true_label == 1:
                        result = "WIN"
                        realized_pnl = stake_usd * self.net_payout_ratio
                    else:
                        result = "LOSS"
                        realized_pnl = -stake_usd
                else:  # PUT
                    # PUT wins when true_label == 0 (CALL loses)
                    if true_label == 0:
                        result = "WIN"
                        realized_pnl = stake_usd * self.net_payout_ratio
                    else:
                        result = "LOSS"
                        realized_pnl = -stake_usd

                trade = BacktestTrade(
                    trade_id=len(trades),
                    symbol=self.symbol,
                    expiry_minutes=expiry_minutes,
                    direction=direction,
                    direction_int=direction_int,
                    entry_time=str(row.get("entry_ts", "")) if row is not None else "",
                    entry_price=float(row.get("entry_price", 0)) if row is not None else 0.0,
                    expiry_price=float(row.get("expiry_price", 0)) if row is not None else 0.0,
                    stake_usd=stake_usd,
                    raw_probability=round(dir_prob, 4),
                    calibrated_probability=round(dir_prob, 4),
                    break_even_probability=round(self.break_even_probability, 4),
                    effective_edge=round(effective_edge, 4),
                    expected_roi=round(expected_roi, 4),
                    net_payout_ratio=self.net_payout_ratio,
                    result=result,
                    realized_pnl=round(realized_pnl, 4),
                    test_window=window_id,
                )
                trades.append(trade)

        return trades

    def _compute_window_stats(self, trades: List[BacktestTrade]) -> dict:
        """计算窗口统计"""
        if not trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "ties": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "total_staked": 0,
                "roi": 0.0, "avg_effective_edge": 0.0,
                "brier_score": 0.0, "max_drawdown": 0.0,
            }

        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        ties = sum(1 for t in trades if t.result == "TIE")
        settled = wins + losses

        total_pnl = sum(t.realized_pnl for t in trades)
        total_staked = sum(t.stake_usd for t in trades)
        roi = total_pnl / total_staked if total_staked > 0 else 0.0
        win_rate = wins / settled if settled > 0 else 0.0
        avg_edge = sum(t.effective_edge for t in trades) / len(trades) if trades else 0.0

        # Brier Score
        brier = 0.0
        for t in trades:
            actual = 1.0 if t.result == "WIN" else 0.0
            brier += (t.raw_probability - actual) ** 2
        brier /= len(trades) if trades else 1

        # Max Drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            cum_pnl += t.realized_pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "total_staked": total_staked,
            "roi": round(roi, 4),
            "avg_effective_edge": round(avg_edge, 4),
            "brier_score": round(brier, 4),
            "max_drawdown": round(max_dd, 4),
        }

    def _compute_edge_bucket_stats(self, all_trades: List[BacktestTrade]) -> dict:
        """按 Edge 区间统计"""
        buckets = {
            "negative": [],
            "0-1%": [],
            "1-2%": [],
            "2-3%": [],
            "3-5%": [],
            "5-7%": [],
            "7-10%": [],
            "10%+": [],
        }
        for t in all_trades:
            edge = t.effective_edge
            if edge < 0:
                buckets["negative"].append(t)
            elif edge < 0.01:
                buckets["0-1%"].append(t)
            elif edge < 0.02:
                buckets["1-2%"].append(t)
            elif edge < 0.03:
                buckets["2-3%"].append(t)
            elif edge < 0.05:
                buckets["3-5%"].append(t)
            elif edge < 0.07:
                buckets["5-7%"].append(t)
            elif edge < 0.10:
                buckets["7-10%"].append(t)
            else:
                buckets["10%+"].append(t)

        result = {}
        for name, trades in buckets.items():
            if not trades:
                continue
            stats = self._compute_window_stats(trades)
            stats["avg_expected_roi"] = round(sum(t.expected_roi for t in trades) / len(trades), 4)
            result[name] = stats

        return result

    def _compute_probability_bucket_stats(self, all_trades: List[BacktestTrade]) -> dict:
        """按概率区间统计"""
        buckets = {
            "50-52%": [], "52-54%": [], "54-56%": [], "56-58%": [],
            "58-60%": [], "60-65%": [], "65-70%": [], "70%+": [],
        }
        for t in all_trades:
            prob = t.raw_probability
            if prob < 0.52:
                buckets["50-52%"].append(t)
            elif prob < 0.54:
                buckets["52-54%"].append(t)
            elif prob < 0.56:
                buckets["54-56%"].append(t)
            elif prob < 0.58:
                buckets["56-58%"].append(t)
            elif prob < 0.60:
                buckets["58-60%"].append(t)
            elif prob < 0.65:
                buckets["60-65%"].append(t)
            elif prob < 0.70:
                buckets["65-70%"].append(t)
            else:
                buckets["70%+"].append(t)

        result = {}
        for name, trades in buckets.items():
            if not trades:
                continue
            stats = self._compute_window_stats(trades)
            stats["avg_predicted_prob"] = round(sum(t.raw_probability for t in trades) / len(trades), 4)
            result[name] = stats

        return result

    def run(self, df_1m: pd.DataFrame, equity: float = 100.0) -> BacktestReport:
        """
        执行 Walk-Forward 回测。

        数据流: 1m 数据 → 重采样到 15m → 计算特征 → 构建标签 → 对齐 → 训练 → 预测 → 模拟交易
        """
        from scripts.train_ensemble_v2 import calc_features, FEATURES

        self._log(f"\n{'='*65}")
        self._log(f"  Walk-Forward Backtest: {self.symbol} {self.expiries}m")
        self._log(f"  Train: {self.train_window_days}d | Test: {self.test_window_days}d | Step: {self.step_days}d")
        self._log(f"  Min Order: {self.min_order_usd}U | Max Bet: {self.max_bet_fraction*100:.0f}%")
        self._log(f"  Break-even: {self.break_even_probability:.1%} | Payout: {self.net_payout_ratio:.0%}")
        self._log(f"{'='*65}")

        # ── 1. 重采样 1m → 15m ──
        self._log(f"\n[1/4] 重采样 1m → 15m...")
        resampled = df_1m.resample("15min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(resampled) < self.min_history:
            self._log(f"  ERROR: 重采样后数据不足 ({len(resampled)} < {self.min_history})")
            return BacktestReport(symbol=self.symbol)
        self._log(f"  Resampled: {len(resampled)} rows ({resampled.index[0]} ~ {resampled.index[-1]})")

        # ── 2. 计算特征 + 构建标签 ──
        self._log(f"\n[2/4] 计算特征和标签...")
        feat_df = calc_features(resampled)
        # 在 15m 数据上构建标签时，expiry_minutes=1 表示向前看 1 根 K 线（=15 分钟到期）
        builder = LabelBuilder(price_source="gate_io", expiries=[1])
        labels = builder.build_labels(resampled, symbol=self.symbol, min_samples=self.min_history)
        if labels.empty:
            self._log("  ERROR: 标签构建失败")
            return BacktestReport(symbol=self.symbol)
        binary = builder.filter_binary_labels(labels)
        self._log(f"  Features: {len(feat_df)} rows | Labels: {len(binary)} samples")

        # ── 3. 对齐特征与标签 ──
        self._log(f"\n[3/4] 对齐特征与标签...")
        feat_index = feat_df.index
        feat_values = feat_df[FEATURES].values
        X_list, y_list = [], []
        aligned_rows = []

        for _, row in binary.iterrows():
            entry_ts_ms = row["entry_ts"]
            entry_dt = pd.Timestamp(entry_ts_ms, unit="ms")
            if entry_dt.tz is not None:
                entry_dt = entry_dt.tz_localize(None)
            if feat_index.tz is not None:
                entry_dt = entry_dt.tz_localize("UTC")

            mask = feat_index <= entry_dt
            if not mask.any():
                continue
            feat_idx = mask.sum() - 1
            if feat_idx < self.min_history:
                continue

            try:
                feat_row = feat_values[feat_idx]
                if np.any(np.isnan(feat_row)) or np.any(np.isinf(feat_row)):
                    continue
                X_list.append(feat_row)
                y_list.append(row["label_binary"])
                aligned_rows.append(row.to_dict())
            except (IndexError, KeyError):
                continue

        if not X_list:
            self._log("  ERROR: 对齐失败")
            return BacktestReport(symbol=self.symbol)

        X_all = np.array(X_list)
        y_all = np.array(y_list)
        aligned_df = pd.DataFrame(aligned_rows)
        self._log(f"  Aligned: {len(X_all)} samples")

        # ── 4. Walk-Forward 回测 ──
        self._log(f"\n[4/4] Walk-Forward 回测...")

        total_days = (resampled.index[-1] - resampled.index[0]).days
        n_windows = max(1, (total_days - self.train_window_days - self.test_window_days) // self.step_days + 1)

        aligned_df = aligned_df.sort_values("entry_ts")
        all_trades = []
        windows = []

        for w in range(n_windows):
            train_start = resampled.index[0] + pd.Timedelta(days=w * self.step_days)
            train_end = train_start + pd.Timedelta(days=self.train_window_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_window_days)

            train_mask = (
                (pd.to_datetime(aligned_df["entry_ts"], unit="ms") >= train_start) &
                (pd.to_datetime(aligned_df["entry_ts"], unit="ms") < train_end)
            )
            test_mask = (
                (pd.to_datetime(aligned_df["entry_ts"], unit="ms") >= test_start) &
                (pd.to_datetime(aligned_df["entry_ts"], unit="ms") < test_end)
            )

            train_idx = np.where(train_mask.values)[0]
            test_idx = np.where(test_mask.values)[0]

            if len(train_idx) < self.min_samples_train or len(test_idx) < 20:
                continue

            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]
            test_df = aligned_df.iloc[test_idx].reset_index(drop=True)

            self._log(f"\n  Window {w+1}/{n_windows}: "
                      f"Train[{train_start.strftime('%m-%d')}~{train_end.strftime('%m-%d')}] "
                      f"({len(X_train)}) → Test[{test_start.strftime('%m-%d')}~{test_end.strftime('%m-%d')}] "
                      f"({len(X_test)})")

            try:
                model, scaler, y_prob = self._train_model(X_train, y_train, X_test, y_test, w)
            except Exception as e:
                self._log(f"    ⚠ 训练失败: {e}")
                continue

            trades = self._simulate_trades(y_prob, y_test, test_df, w, equity, self.expiries[0])
            stats = self._compute_window_stats(trades)

            window_result = WindowResult(
                window_id=w,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
                train_samples=len(X_train),
                test_samples=len(X_test),
                trades=trades,
                **stats,
            )
            windows.append(window_result)
            all_trades.extend(trades)

            self._log(f"    Trades: {stats['total_trades']} | "
                      f"Win: {stats['wins']}/{stats['losses']} | "
                      f"WR: {stats['win_rate']:.1%} | "
                      f"PnL: {stats['total_pnl']:+.2f}U | "
                      f"ROI: {stats['roi']:+.1%}")

        # 4. 汇总
        self._log(f"\n[4/4] 汇总报告...")

        if not all_trades:
            self._log("  No trades generated")
            return BacktestReport(symbol=self.symbol, expiry_minutes=self.expiries[0])

        # 整体统计
        total_wins = sum(1 for t in all_trades if t.result == "WIN")
        total_losses = sum(1 for t in all_trades if t.result == "LOSS")
        total_ties = sum(1 for t in all_trades if t.result == "TIE")
        settled = total_wins + total_losses
        overall_wr = total_wins / settled if settled > 0 else 0.0
        total_pnl = sum(t.realized_pnl for t in all_trades)
        total_staked = sum(t.stake_usd for t in all_trades)
        overall_roi = total_pnl / total_staked if total_staked > 0 else 0.0
        avg_edge = sum(t.effective_edge for t in all_trades) / len(all_trades) if all_trades else 0.0
        avg_expected_roi = sum(t.expected_roi for t in all_trades) / len(all_trades) if all_trades else 0.0

        # Brier Score
        brier = 0.0
        for t in all_trades:
            actual = 1.0 if t.result == "WIN" else 0.0
            brier += (t.raw_probability - actual) ** 2
        brier /= len(all_trades)

        # Max Drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in all_trades:
            cum_pnl += t.realized_pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

        # Streaks
        longest_lose = 0
        longest_win = 0
        current_lose = 0
        current_win = 0
        for t in all_trades:
            if t.result == "WIN":
                current_win += 1
                current_lose = 0
                longest_win = max(longest_win, current_win)
            elif t.result == "LOSS":
                current_lose += 1
                current_win = 0
                longest_lose = max(longest_lose, current_lose)

        # Profit Factor
        gross_profit = sum(t.realized_pnl for t in all_trades if t.realized_pnl > 0)
        gross_loss = abs(sum(t.realized_pnl for t in all_trades if t.realized_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (简化)
        pnl_series = [t.realized_pnl for t in all_trades]
        sharpe = (np.mean(pnl_series) / np.std(pnl_series)) * np.sqrt(len(pnl_series)) if np.std(pnl_series) > 0 else 0.0

        # 分维度
        by_direction = {
            "CALL": self._compute_window_stats([t for t in all_trades if t.direction == "CALL"]),
            "PUT": self._compute_window_stats([t for t in all_trades if t.direction == "PUT"]),
        }
        by_edge = self._compute_edge_bucket_stats(all_trades)
        by_prob = self._compute_probability_bucket_stats(all_trades)

        report = BacktestReport(
            symbol=self.symbol,
            expiry_minutes=self.expiries[0],
            total_windows=len(windows),
            total_trades=len(all_trades),
            total_wins=total_wins,
            total_losses=total_losses,
            total_ties=total_ties,
            overall_win_rate=round(overall_wr, 4),
            break_even_win_rate=round(self.break_even_probability, 4),
            total_pnl=round(total_pnl, 4),
            total_staked=total_staked,
            overall_roi=round(overall_roi, 4),
            avg_expected_roi=round(avg_expected_roi, 4),
            avg_effective_edge=round(avg_edge, 4),
            overall_brier_score=round(brier, 4),
            max_drawdown=round(max_dd, 4),
            longest_losing_streak=longest_lose,
            longest_winning_streak=longest_win,
            sharpe_ratio=round(sharpe, 4),
            profit_factor=round(profit_factor, 4),
            by_direction=by_direction,
            by_probability_bucket=by_prob,
            by_edge_bucket=by_edge,
            windows=windows,
            all_trades=all_trades,
        )

        return report

    def print_report(self, report: BacktestReport):
        """打印回测报告"""
        # 懒计算：如果 report 中没有预计算 bucket stats，则从 all_trades 计算
        if report.all_trades:
            if not report.by_edge_bucket:
                report.by_edge_bucket = self._compute_edge_bucket_stats(report.all_trades)
            if not report.by_probability_bucket:
                report.by_probability_bucket = self._compute_probability_bucket_stats(report.all_trades)
            if not report.by_direction:
                report.by_direction = {
                    "CALL": self._compute_window_stats([t for t in report.all_trades if t.direction == "CALL"]),
                    "PUT": self._compute_window_stats([t for t in report.all_trades if t.direction == "PUT"]),
                }
        print(f"\n{'='*65}")
        print(f"  Walk-Forward Backtest Report")
        print(f"  {report.symbol} | {report.expiry_minutes}m Expiry")
        print(f"{'='*65}")

        print(f"\n  ── 整体统计 ──")
        print(f"  Walk-Forward 窗口: {report.total_windows}")
        print(f"  总交易: {report.total_trades}")
        print(f"  胜: {report.total_wins} | 负: {report.total_losses} | 平: {report.total_ties}")
        print(f"  胜率: {report.overall_win_rate:.1%}")
        print(f"  盈亏平衡胜率: {report.break_even_win_rate:.1%}")
        print(f"  总盈亏: {report.total_pnl:+.2f}U")
        print(f"  总投入: {report.total_staked}U")
        print(f"  ROI: {report.overall_roi:+.2%}")
        print(f"  平均预期 ROI: {report.avg_expected_roi:+.2%}")
        print(f"  平均 Edge: {report.avg_effective_edge:.2%}")
        print(f"  Brier Score: {report.overall_brier_score:.4f}")
        print(f"  Max Drawdown: {report.max_drawdown:+.2f}U")
        print(f"  最长连败: {report.longest_losing_streak}")
        print(f"  最长连胜: {report.longest_winning_streak}")
        print(f"  Profit Factor: {report.profit_factor:.2f}")
        print(f"  Sharpe Ratio: {report.sharpe_ratio:.2f}")

        print(f"\n  ── 按方向 ──")
        for direction, stats in report.by_direction.items():
            if stats["total_trades"] > 0:
                print(f"  {direction}: {stats['total_trades']}笔 | "
                      f"胜率{stats['win_rate']:.1%} | "
                      f"ROI {stats['roi']:+.2%} | "
                      f"PnL {stats['total_pnl']:+.2f}U")

        print(f"\n  ── 按 Edge 区间 ──")
        edge_order = ["negative", "0-1%", "1-2%", "2-3%", "3-5%", "5-7%", "7-10%", "10%+"]
        for bucket in edge_order:
            if bucket in report.by_edge_bucket:
                s = report.by_edge_bucket[bucket]
                print(f"  {bucket:>8}: {s['total_trades']:>4}笔 | "
                      f"胜率{s['win_rate']:>6.1%} | "
                      f"ROI {s['roi']:>+7.2%} | "
                      f"预期ROI {s.get('avg_expected_roi', 0):>+7.2%}")

        print(f"\n  ── 按概率区间 (Reliability Check) ──")
        prob_order = ["50-52%", "52-54%", "54-56%", "56-58%", "58-60%", "60-65%", "65-70%", "70%+"]
        for bucket in prob_order:
            if bucket in report.by_probability_bucket:
                s = report.by_probability_bucket[bucket]
                calibration_error = s.get("avg_predicted_prob", 0) - s["win_rate"]
                print(f"  {bucket:>8}: {s['total_trades']:>4}笔 | "
                      f"预测{s.get('avg_predicted_prob', 0):.1%} | "
                      f"实际{s['win_rate']:.1%} | "
                      f"偏差{calibration_error:+.1%}")

        print(f"\n  ── 逐窗口 ──")
        print(f"  {'Win':>3} {'Train':>25} {'Test':>25} {'Trades':>6} {'WR':>7} {'PnL':>8} {'ROI':>7}")
        for w in report.windows:
            print(f"  {w.window_id:>3} {w.train_start+'~'+w.train_end:>25} "
                  f"{w.test_start+'~'+w.test_end:>25} "
                  f"{w.total_trades:>6} {w.win_rate:>6.1%} "
                  f"{w.total_pnl:>+8.2f} {w.roi:>+6.1%}")

        print(f"\n{'='*65}")

    def save_report(self, report: BacktestReport, filename: Optional[str] = None):
        """保存报告到 JSON"""
        if filename is None:
            filename = f"{self.symbol.lower()}_{self.expiries[0]}m_backtest.json"

        filepath = os.path.join(self.output_dir, filename)

        # 转换为可序列化的字典
        report_dict = asdict(report)
        # 移除 trades 列表（太大），只保留统计
        if "all_trades" in report_dict:
            del report_dict["all_trades"]
        if "windows" in report_dict:
            for w in report_dict["windows"]:
                if "trades" in w:
                    del w["trades"]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)

        self._log(f"\n  Report saved: {filepath}")
        return filepath