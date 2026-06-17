# -*- coding: utf-8 -*-
"""
ATradeBot 引擎 v3 — SignalValidator → RiskManager → OrderExecutor 流水线

核心规则:
  1. SignalValidator (L0-L2): 防接刀 → 概率门槛 → 极值翻转+概率重置
  2. RiskManager (L3-L5): 共振分 → 双重冷却 → 持仓管理
  3. OrderExecutor: 按 action 执行开仓/加仓/反手
  4. 盈亏来自HIBT余额变化, 不来自CSV收盘价
"""
import time
import json
import sys
import os
import numpy as np
import pandas as pd
from . import config
from . import predictor
from .signal_validator import validate_signal
from .risk_manager import manage_signal
from .order_executor import OrderExecutor
from .ai_judge import judge
from .exchange import fetch_balance, place_order
from .notifier import notify_trade, notify_result
from .models import Prediction


def emit(event_type: str, payload: dict):
    event = {"type": event_type, "ts": int(time.time() * 1000), "payload": payload}
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _calc_30m_features(data_feat: pd.DataFrame) -> dict | None:
    """特征计算 (在 FEATURE_INTERVAL_MIN 聚合K线上计算, 保持与训练特征一致)"""
    d = data_feat.copy()
    eps = 1e-10
    d["volume"] = d["volume"].fillna(0).replace(0, 0.001)
    d["ret_1"] = d["close"].pct_change(1).fillna(0)
    d["ret_3"] = d["close"].pct_change(3).fillna(0)
    d["ret_6"] = d["close"].pct_change(6).fillna(0)
    e12 = d["close"].ewm(span=12, adjust=False).mean()
    e26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    d["MACD"] = (2 * (macd - macd.ewm(span=9, adjust=False).mean())).fillna(0)
    d["MACD_hist"] = d["MACD"].fillna(0)
    delta = d["close"].diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, eps)
    d["RSI"] = (100 - (100 / (1 + gain / loss))).fillna(50)
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std().fillna(0)
    d["BB_Pos"] = ((d["close"] - (mid - 2 * std)) / (4 * std + eps)).clip(0, 1)
    d["BB_width"] = (((mid + 2 * std) - (mid - 2 * std)) / (mid + eps)).fillna(0)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift(1)).abs(),
        (d["low"] - d["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    d["ATR_pct"] = (tr.rolling(14).mean() / (d["close"] + eps)).fillna(0)
    up = d["high"] - d["high"].shift(1)
    dn = d["low"].shift(1) - d["low"]
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=d.index)
    tr14 = tr.rolling(14).sum().replace(0, eps)
    pdi = 100 * pdm.rolling(14).sum() / tr14
    ndi = 100 * ndm.rolling(14).sum() / tr14
    d["ADX"] = (100 * abs(pdi - ndi) / (pdi + ndi + eps)).rolling(14).mean().fillna(20)
    d["MA10"] = d["close"].rolling(10).mean().bfill()
    d["MA20"] = d["close"].rolling(20).mean().bfill()
    d["MA50"] = d["close"].rolling(50).mean().bfill()
    d["price_vs_MA20"] = ((d["close"] - d["MA20"]) / (d["MA20"] + eps)).fillna(0)
    d["price_vs_MA50"] = ((d["close"] - d["MA50"]) / (d["MA50"] + eps)).fillna(0)
    d["MA_trend"] = np.sign(d["MA10"] - d["MA20"]).fillna(0)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    vwap = (d["volume"] * tp).cumsum() / (d["volume"].cumsum() + eps)
    d["VWAP_dist"] = ((d["close"] - vwap) / (vwap + eps)).fillna(0)
    d["vol_ratio"] = (d["volume"] / (d["volume"].rolling(5).mean() + eps)).fillna(1)
    obv_dir = np.sign(d["close"].diff().fillna(0))
    obv = (d["volume"] * obv_dir).cumsum()
    d["OBV_trend"] = np.sign(obv - obv.shift(5)).fillna(0)
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["CCI"] = ((tp - tp_sma) / (0.015 * tp_mad + eps)).fillna(0)
    atr14 = tr.rolling(14).sum()
    d["CHOP"] = (100 * np.log10(atr14 / (d["high"].rolling(14).max() - d["low"].rolling(14).min() + eps)) / np.log10(14)).fillna(50)
    body = (d["close"] - d["open"]).abs()
    d["body_pct"] = (body / (d["high"] - d["low"] + eps)).fillna(0.5)
    d["is_green"] = (d["close"] > d["open"]).astype(int)
    result = d.replace([np.inf, -np.inf], np.nan).dropna()
    return result.iloc[-1].to_dict() if not result.empty else None


class TradingEngine:
    def __init__(self):
        self.executor = OrderExecutor()
        self.reset_state()

    def reset_state(self):
        """完全重置状态"""
        self.running = False
        self.paused = False
        self.balance = 0.0
        self.start_balance = 0.0
        # 旧版 active_trades 保留仅用于结算检测（余额变化法）
        self.active_trades: list[dict] = []
        self.last_trade_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_reject_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_signal_bar_ts: dict[str, object] = {s: None for s in config.SYMBOLS}
        self.recent_results: list[bool] = []
        self.consecutive_losses = 0
        self.halted = False
        self.pause_until = 0
        self.bet_mode = "kelly"
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        self._warmup_done: dict[str, bool] = {s: False for s in config.SYMBOLS}
        self._csv_last_mtime = 0.0
        # 重建 executor（清空内部状态）
        self.executor = OrderExecutor()

    def start(self):
        self.reset_state()
        self.running = True
        n = predictor.load_models()
        predictor.set_bootstrap_mode()
        emit("status", {"state": "running", "models_loaded": n})
        self.balance = fetch_balance()
        if self.balance < 0:
            self.balance = 0.0
        self.start_balance = self.balance
        emit("balance_update", {"balance": self.balance})
        self._warmup_done = {s: False for s in config.SYMBOLS}
        emit("log", {
            "msg": f"启动! 余额{self.balance:.2f}U | 底仓3U | "
                   f"防接刀|共振分≥{config.CONFLUENCE_MIN}|概率重置{config.REVERSAL_PROB}|"
                   f"结算冷却{config.SETTLEMENT_COOLDOWN_SEC}s"
        })

    def stop(self):
        self.running = False
        self.paused = False
        emit("status", {"state": "stopped"})

    def pause(self):
        self.paused = True
        emit("status", {"state": "paused"})

    def resume(self):
        self.paused = False
        self.halted = False
        self.pause_until = 0
        self.consecutive_losses = 0
        emit("status", {"state": "running"})
        emit("log", {"msg": "恢复运行"})

    def _can_trade(self, bet: int) -> bool:
        needed = bet * (len(self.active_trades) + 1)
        return self.balance >= needed + 0.5

    def _check_settlement(self):
        """
        通过HIBT余额变化来判断订单是否已结算。
        结算后通知 OrderExecutor.on_settlement() 清理持仓并执行反向信号。
        """
        if not self.active_trades:
            return

        new_balance = fetch_balance()
        if new_balance < 0:
            return

        prev_balance = self.balance
        balance_changed = abs(new_balance - prev_balance) > 0.005

        for i in range(len(self.active_trades) - 1, -1, -1):
            t = self.active_trades[i]
            elapsed = time.time() * 1000 - t["start_ts"]

            if elapsed < config.HOLD_MINUTES * 60000:
                continue

            if not balance_changed:
                continue

            pnl = new_balance - prev_balance
            is_win = pnl > 0

            if is_win:
                self.total_wins += 1
            else:
                self.total_losses += 1
            self.total_pnl += pnl
            self._record_result(is_win)
            result = "win" if is_win else "loss"
            emit("trade_result", {
                "symbol": t["symbol"], "result": result, "pnl": round(pnl, 4),
                "entryPrice": t["entry"], "dir": t["dir"],
            })
            notify_result(t["symbol"], is_win, pnl)
            self.active_trades.pop(i)
            self.balance = new_balance
            emit("balance_update", {"balance": new_balance})
            emit("log", {"msg": f"结算 {t['symbol']}: {'赢' if is_win else '输'} {abs(pnl):.2f}U | 余额→{new_balance:.2f}U"})

            # 关键: 通知 OrderExecutor 结算完成
            self.executor.on_settlement(t["symbol"])

    def _record_result(self, is_win: bool):
        self.recent_results.append(is_win)
        if len(self.recent_results) > config.RECENT_WINDOW:
            self.recent_results = self.recent_results[-config.RECENT_WINDOW:]
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= config.CONSECUTIVE_LOSS_HALT:
                self.halted = True
                emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 暂停! 需手动恢复"})
            elif self.consecutive_losses >= 3:
                pause_ms = config.CONSECUTIVE_LOSS_PAUSE_SEC * 1000
                self.pause_until = int(time.time() * 1000) + pause_ms
                emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 冷冻{pause_ms//1000}秒"})

    def process_symbol(self, symbol: str, full_df: pd.DataFrame, current_ts: int):
        # ── 全局风控 ──
        if self.halted or current_ts < self.pause_until:
            return
        if not self.active_trades and current_ts < self.pause_until:
            return

        # 余额检查
        if not self._can_trade(config.BET_MIN):
            return

        # 品种模型检查
        if symbol not in predictor.ENSEMBLE_MODELS:
            return

        # ── 数据准备（与原版一致） ──
        df_s = full_df[full_df["symbol"] == symbol].copy()
        if len(df_s) < 200:
            return
        df_s.set_index("datetime", inplace=True)

        candle_min = config.CANDLE_INTERVAL_MIN
        detect_data = df_s.resample(f"{candle_min}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(detect_data) < 10:
            return

        current_bar_ts = detect_data.index[-1]

        if not self._warmup_done.get(symbol, False):
            self.last_signal_bar_ts[symbol] = current_bar_ts
            self._warmup_done[symbol] = True
            emit("log", {"msg": f"预热完成 {symbol}: 跳过当前{candle_min}分K线 {current_bar_ts}"})
            return

        if self.last_signal_bar_ts.get(symbol) == current_bar_ts:
            return

        feat_min = config.FEATURE_INTERVAL_MIN
        feature_data = df_s.resample(f"{feat_min}min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(feature_data) < 200:
            return

        row = _calc_30m_features(feature_data)
        if row is None:
            return

        # ── 模型预测 ──
        pred = predictor.predict(symbol, row)
        if pred is None or pred.prob_win < 0.30:
            return

        # ── 指标提取 ──
        indicators = {
            "ADX": float(row.get("ADX", 20)),
            "BB_Pos": float(row.get("BB_Pos", 0.5)),
            "RSI": float(row.get("RSI", 50)),
            "MACD": float(row.get("MACD", 0)),
            "macd_hist_change": float(row.get("MACD_hist", 0)),
            "BSP_5": float(row.get("ret_1", 0)),
        }
        emit("features", {"symbol": symbol, "indicators": indicators})

        current_price = float(row.get("close", 0))

        # ═══════════════════════════════════════
        # 流水线第一站: SignalValidator (L0-L2)
        # ═══════════════════════════════════════
        existing_pos = self.executor.positions.get(symbol)
        signal, val_gates = validate_signal(
            symbol, pred, row, indicators,
            current_price=current_price,
            existing_position=existing_pos,
        )

        # 发送 L0-L2 门结果
        for g in val_gates:
            emit("risk_gate", {
                "symbol": symbol, "level": g.level, "name": g.name,
                "passed": g.passed, "reason": g.reason,
            })
            if not g.passed:
                self.executor.last_reject_ts[symbol] = current_ts
                self.last_reject_ts[symbol] = current_ts
                return

        if signal is None:
            return

        # ═══════════════════════════════════════
        # 流水线第二站: RiskManager (L3-L5)
        # ═══════════════════════════════════════
        signal, mgmt_gates = manage_signal(
            signal,
            current_ts=current_ts,
            last_reject_ts=self.executor.last_reject_ts,
            last_settlement_ts=self.executor.last_settlement_ts,
            existing_position=existing_pos,
            current_price=current_price,
        )

        for g in mgmt_gates:
            emit("risk_gate", {
                "symbol": symbol, "level": g.level, "name": g.name,
                "passed": g.passed, "reason": g.reason,
            })
            if not g.passed:
                if g.level == 4 or g.name == "Reject Cooldown":
                    self.executor.last_reject_ts[symbol] = current_ts
                    self.last_reject_ts[symbol] = current_ts
                return

        if signal is None:
            return

        # 发送预测（让前端看到信号方向）
        emit("prediction", {
            "symbol": symbol,
            "prob_long": 1 - signal.direction / 3.0,
            "direction": signal.direction,
            "prob_win": signal.prob_win,
            "is_reversal": signal.is_reversal,
        })
        emit("tick", {
            "symbol": symbol, "dir": signal.dir_str,
            "ml_prob": signal.prob_win, "phase": "trade",
        })

        # AI仅展示（不否决）
        try:
            approval, reason, _ = judge(
                symbol, signal.direction, signal.prob_win,
                indicators, signal.is_reversal,
                confluence=signal.confluence,
            )
        except Exception:
            approval, reason = 0.0, "AI不可用"
        emit("risk_gate", {
            "symbol": symbol, "level": 6, "name": "AI分析",
            "passed": True, "reason": f"AI: {reason}",
        })

        # ═══════════════════════════════════════
        # 流水线第三站: OrderExecutor
        # ═══════════════════════════════════════
        ok = self.executor.execute(signal, current_ts, current_bar_ts)

        if ok:
            # 记录到旧的 active_trades 用于结算检测
            self.active_trades.append({
                "symbol": symbol, "dir": signal.direction,
                "start_ts": current_ts, "amount": config.BET_MIN,
                "entry": current_price,
                "pre_balance": self.balance,
            })
            self.last_trade_ts[symbol] = current_ts
            self.last_signal_bar_ts[symbol] = current_bar_ts

            # 查余额
            self.balance = fetch_balance()
            if self.balance < 0:
                self.balance = self.balance - config.BET_MIN

            emit("trade_executed", {
                "symbol": symbol, "direction": "CALL" if signal.direction == 1 else "PUT",
                "entryPrice": current_price, "amount": config.BET_MIN,
                "mlProb": signal.prob_win, "balance": self.balance,
                "confluence": signal.confluence,
                "flipped": signal.is_reversal,
                "action": signal.action,
            })
            notify_trade(
                symbol, signal.dir_str, current_price, config.BET_MIN,
                signal.prob_win, indicators, reason, self.balance,
                len(self.active_trades), signal.is_reversal,
            )
            emit("balance_update", {"balance": self.balance})

            # 如果 action=close_and_open 且下单的是反向开仓:
            # 但 close_and_open 被设计为: 先标记 pending_close, 等结算后再开反向
            # 此时 execute() 不会立即下单，所以不应记录到 active_trades
            if signal.action == "close_and_open":
                # 回滚刚才的记录（实际未下单）
                self.active_trades.pop()
        else:
            emit("log", {"msg": f"拒单 {symbol}"})

    def tick(self):
        current_ts = int(time.time() * 1000)
        if not os.path.exists(config.CSV_FILE) or os.path.getsize(config.CSV_FILE) == 0:
            return

        try:
            self._check_settlement()

            full_df = pd.read_csv(config.CSV_FILE, engine="python", on_bad_lines="skip")
            full_df = full_df.iloc[:, :7]
            full_df.columns = ["ts", "symbol", "open", "high", "low", "close", "volume"]
            if str(full_df.iloc[0, 1]).strip() == "symbol":
                full_df = full_df.iloc[1:].copy()
            full_df = full_df.dropna(subset=["ts", "symbol", "close"])
            full_df["datetime"] = pd.to_datetime(full_df["ts"], unit="ms", errors="coerce")
            full_df = full_df.dropna(subset=["datetime"])

            if len(full_df) < 50:
                return

            for s in config.SYMBOLS:
                ds = full_df[full_df["symbol"] == s]
                if not ds.empty:
                    last = ds.iloc[-1]
                    emit("candle_update", {
                        "symbol": s, "ts": int(last["ts"]),
                        "open": float(last["open"]), "high": float(last["high"]),
                        "low": float(last["low"]), "close": float(last["close"]),
                        "volume": float(last["volume"]) if len(last) > 6 else 0,
                    })

            for s in config.SYMBOLS:
                self.process_symbol(s, full_df, current_ts)

        except Exception as e:
            emit("error", {"msg": f"Error: {str(e)[:100]}"})

    def get_status(self) -> dict:
        total = self.total_wins + self.total_losses
        wr = f"{(self.total_wins / total * 100):.1f}%" if total > 0 else "0.0%"
        state = "halted" if self.halted else (
            "paused" if self.paused else (
                "running" if self.running else "stopped"
            )
        )
        if self.pause_until > int(time.time() * 1000):
            state = "cooling"
        profit = self.balance - self.start_balance if self.start_balance > 0 else 0
        return {
            "state": state, "balance": self.balance,
            "wins": self.total_wins, "losses": self.total_losses,
            "winRate": wr, "activeTrades": len(self.active_trades),
            "maxConcurrentTrades": 999,
            "consecutiveLosses": self.consecutive_losses,
            "currentBet": config.BET_MIN,
            "betMode": "fixed_3u",
            "profit": round(profit, 2),
        }
