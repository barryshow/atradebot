# -*- coding: utf-8 -*-
"""
ATradeBot 引擎 v2 — 尊重HIBT真实余额和结算

核心规则:
  1. 下单前检查余额: bet * (持仓+1) <= 余额 × 0.9
  2. 不下模拟单: 不下单就是不下单, 不虚构settle
  3. 盈亏来自HIBT余额变化, 不来自CSV收盘价
  4. 方向: 集成模型说啥就是啥, 不额外翻转
"""
import time
import json
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from . import config
from . import predictor
from .risk_gates import run_risk_pipeline
from .ai_judge import judge
from .exchange import fetch_balance, place_order
from .notifier import notify_trade, notify_result
from .models import Prediction


def emit(event_type: str, payload: dict):
    event = {"type": event_type, "ts": int(time.time() * 1000), "payload": payload}
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _calc_30m_features(data_30m: pd.DataFrame) -> dict | None:
    """30分钟特征计算 (同predictor训练用特征一致)"""
    d = data_30m.copy()
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
        self.reset_state()

    def reset_state(self):
        """完全重置状态"""
        self.running = False
        self.paused = False
        self.balance = 0.0
        self.start_balance = 0.0  # 启动时的余额快照
        self.active_trades: list[dict] = []
        self.last_trade_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_reject_ts: dict[str, int] = {s: 0 for s in config.SYMBOLS}
        self.last_signal_bar_ts: dict[str, object] = {s: None for s in config.SYMBOLS}
        self.recent_results: list[bool] = []
        self.consecutive_losses = 0
        self.halted = False
        self.pause_until = 0
        self.bet_mode = "fixed"
        self.turbo = config.BOOTSTRAP_MODE.lower() == "turbo"
        # 已结算的盈亏
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        # 启动预热: 第一根K线不交易, 等下一根新K线
        self._warmup_done: dict[str, bool] = {s: False for s in config.SYMBOLS}
        self._csv_last_mtime = 0.0  # CSV最后修改时间, 用于检测数据新鲜度

    def start(self):
        self.reset_state()
        self.running = True
        n = predictor.load_models()
        predictor.set_bootstrap_mode(True, turbo=self.turbo)
        emit("status", {"state": "running", "models_loaded": n})
        # 从HIBT拿真实余额
        self.balance = fetch_balance()
        if self.balance < 0:
            self.balance = 0.0
        self.start_balance = self.balance
        emit("balance_update", {"balance": self.balance})
        # 重置预热标记
        self._warmup_done = {s: False for s in config.SYMBOLS}
        profit_target = config.BOOTSTRAP_TARGET - config.INITIAL_CAPITAL
        label = "🚀 TURBO" if self.turbo else "普通"
        emit("log", {"msg": f"{label}启动! 余额{self.balance:.2f}U | 3U定投 +{profit_target}U→凯利 | 阈值={'0.50' if self.turbo else '0.55'} | 预热中: 等下一根新K线再开单"})

    def stop(self):
        self.running = False
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
        """检查余额是否够下单, 防止余额不足还开单"""
        needed = bet * (len(self.active_trades) + 1)
        return self.balance >= needed + 0.5  # 留0.5U余量

    def _check_settlement(self):
        """
        通过HIBT余额变化来判断订单是否已结算。
        对比 (当前余额 - 前次余额快照) 来判断盈亏, 赢和输都能正确检测。
        """
        if not self.active_trades:
            return

        new_balance = fetch_balance()
        if new_balance < 0:
            # HIBT接口故障, 无法结算判断
            return

        prev_balance = self.balance
        balance_changed = abs(new_balance - prev_balance) > 0.005

        for i in range(len(self.active_trades) - 1, -1, -1):
            t = self.active_trades[i]
            elapsed = time.time() * 1000 - t["start_ts"]

            # 还没到到期时间, 跳过
            if elapsed < config.HOLD_MINUTES * 60000:
                continue

            # 到期了, 判断余额是否发生了变化
            # 赢: 余额增加 (本金返还 + 盈利)
            # 输: 余额减少 (本金亏掉, 无返还)
            # 都没变化: 可能还没结算完, 继续等

            if not balance_changed:
                # 余额没动, 还没结算完
                continue

            # 余额有变化, 判断盈亏
            pnl = new_balance - prev_balance
            # 输单: 余额少了约 bet 的量
            # 赢单: 余额多了约 bet * payout 的量
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
                pause_ms = (config.CONSECUTIVE_LOSS_PAUSE_SEC * 1000) // (2 if self.turbo and self.bet_mode == "fixed" else 1)
                self.pause_until = int(time.time() * 1000) + pause_ms
                emit("log", {"msg": f"连亏{self.consecutive_losses}笔, 冷冻{pause_ms//1000}秒"})

    def _get_bet(self) -> int:
        profit = self.balance - self.start_balance
        profit_target = config.BOOTSTRAP_TARGET - config.INITIAL_CAPITAL

        # 检测启动成功
        if self.bet_mode == "fixed" and profit >= profit_target:
            self.bet_mode = "kelly"
            predictor.set_bootstrap_mode(False)
            emit("log", {"msg": f"启动成功! 利润+{profit:.1f}U >= +{profit_target}U, 切换凯利"})

        if self.bet_mode == "fixed":
            return config.FIXED_BET_MIN

        # 凯利
        recent = self.recent_results[-50:] or self.recent_results
        if len(recent) < 5:
            return max(3, min(int(self.balance * 0.05), 10))

        win_rate = sum(recent) / len(recent)
        b = 0.80
        kelly = max(0, (win_rate * b - (1 - win_rate)) / b) * config.KELLY_FRACTION
        bet = int(self.balance * kelly)
        bet = max(config.BET_MIN, min(config.BET_MAX, bet))
        return bet

    def process_symbol(self, symbol: str, full_df: pd.DataFrame, current_ts: int):
        # 风控检查
        if self.halted or current_ts < self.pause_until:
            return
        if not self.active_trades and current_ts < self.pause_until:
            return
        bet = self._get_bet()
        if not self._can_trade(bet):
            return
        # 不限单数, 只看余额够不够
        # HIBT限制: 每个品种同一时间最多一单
        if any(t["symbol"] == symbol for t in self.active_trades):
            return
        if symbol not in predictor.ENSEMBLE_MODELS:
            return
        cooldown = config.TRADE_COOLDOWN_SEC // (2 if self.turbo and self.bet_mode == "fixed" else 1)
        if current_ts - self.last_trade_ts.get(symbol, 0) < cooldown * 1000:
            return
        rej_cd = config.REJECT_COOLDOWN_SEC // (2 if self.turbo and self.bet_mode == "fixed" else 1)
        if current_ts - self.last_reject_ts.get(symbol, 0) < rej_cd * 1000:
            return

        # 数据准备
        df_s = full_df[full_df["symbol"] == symbol].copy()
        if len(df_s) < 200:
            return
        df_s.set_index("datetime", inplace=True)
        data = df_s.resample("15min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        if len(data) < 200:
            return

        current_bar_ts = data.index[-1]

        # 启动预热: 跳过当前这根已存在的K线, 等下一根新K线再交易
        # 避免用半残的旧K线误判方向
        if not self._warmup_done.get(symbol, False):
            self.last_signal_bar_ts[symbol] = current_bar_ts
            self._warmup_done[symbol] = True
            emit("log", {"msg": f"预热完成 {symbol}: 跳过当前K线 {current_bar_ts}, 等待下一根新K线"})
            return

        if self.last_signal_bar_ts.get(symbol) == current_bar_ts:
            return

        row = _calc_30m_features(data)
        if row is None:
            return

        # 模型预测
        pred = predictor.predict(symbol, row)
        if pred is None or pred.prob_win < 0.30:
            return

        # 风控门
        indicators = {
            "ADX": float(row.get("ADX", 20)),
            "BB_Pos": float(row.get("BB_Pos", 0.5)),
            "RSI": float(row.get("RSI", 50)),
            "MACD": float(row.get("MACD", 0)),
            "macd_hist_change": float(row.get("MACD_hist", 0)),
            "BSP_5": float(row.get("ret_1", 0)),
        }
        emit("features", {"symbol": symbol, "indicators": indicators})

        gates, pred, c_score = run_risk_pipeline(pred, row, indicators)
        for g in gates:
            emit("risk_gate", {"symbol": symbol, "level": g.level, "name": g.name, "passed": g.passed, "reason": g.reason})
            if not g.passed:
                self.last_reject_ts[symbol] = current_ts
                return

        # 现在才发prediction(防止前端看到翻转前的方向)
        emit("prediction", {
            "symbol": symbol, "prob_long": pred.prob_long,
            "direction": pred.direction, "prob_win": pred.prob_win,
        })

        dir_str = "做多(CALL)" if pred.direction == 1 else "做空(PUT)"
        emit("tick", {"symbol": symbol, "dir": dir_str, "ml_prob": pred.prob_win, "phase": "trade"})

        # AI仅展示
        try:
            approval, reason, _ = judge(symbol, pred.direction, pred.prob_win, indicators, pred.flipped, confluence=c_score)
        except Exception:
            approval, reason = 0.0, "AI不可用"
        emit("risk_gate", {"symbol": symbol, "level": 3, "name": "AI分析", "passed": True, "reason": f"AI: {reason}"})

        # 在HIBT实盘下单
        res = place_order(symbol, pred.direction, bet, config.HOLD_MINUTES)
        self.last_trade_ts[symbol] = current_ts
        self.last_signal_bar_ts[symbol] = current_bar_ts

        if res.ok:
            pre_balance = self.balance
            entry_price = float(row.get("close", 0))
            self.active_trades.append({
                "symbol": symbol, "dir": pred.direction,
                "start_ts": current_ts, "amount": bet,
                "entry": entry_price, "pre_balance": pre_balance,
            })
            # 查HIBT余额更新
            self.balance = fetch_balance()
            if self.balance < 0:
                self.balance = pre_balance - bet
            emit("trade_executed", {
                "symbol": symbol, "direction": dir_str,
                "entryPrice": entry_price, "amount": bet,
                "mlProb": pred.prob_win, "balance": self.balance,
                "confluence": c_score, "flipped": pred.flipped,
            })
            notify_trade(symbol, dir_str, entry_price, bet, pred.prob_win, indicators, reason, self.balance, len(self.active_trades), pred.flipped)
            emit("balance_update", {"balance": self.balance})
        else:
            emit("log", {"msg": f"拒单 {symbol}: {res.msg}"})

    def tick(self):
        current_ts = int(time.time() * 1000)
        if not os.path.exists(config.CSV_FILE) or os.path.getsize(config.CSV_FILE) == 0:
            return

        try:
            # 先查结算
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

            # 推K线到前端
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

            # 检查信号
            for s in config.SYMBOLS:
                self.process_symbol(s, full_df, current_ts)

        except Exception as e:
            emit("error", {"msg": f"Error: {str(e)[:100]}"})

    def get_status(self) -> dict:
        total = self.total_wins + self.total_losses
        wr = f"{(self.total_wins / total * 100):.1f}%" if total > 0 else "0.0%"
        state = "halted" if self.halted else ("paused" if self.paused else ("running" if self.running else "stopped"))
        if self.pause_until > int(time.time() * 1000):
            state = "cooling"
        profit = self.balance - self.start_balance if self.start_balance > 0 else 0
        profit_target = config.BOOTSTRAP_TARGET - config.INITIAL_CAPITAL
        return {
            "state": state, "balance": self.balance,
            "wins": self.total_wins, "losses": self.total_losses,
            "winRate": wr, "activeTrades": len(self.active_trades),
            "maxConcurrentTrades": 999,
            "consecutiveLosses": self.consecutive_losses,
            "currentBet": self._get_bet(), "betMode": self.bet_mode,
            "profit": round(profit, 2), "bootstrapProfitTarget": profit_target,
        }