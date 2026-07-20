# -*- coding: utf-8 -*-
"""
TradeLedger — 交易决策完整记录系统

每笔交易（无论是否下单）都记录完整决策快照，支持：
- 按 symbol / expiry / regime / direction / probability_bucket 查询
- 按 edge_bucket 统计
- 回测与实盘统一数据结构
- 拒绝交易也记录（含 reject_reason）

存储: JSON Lines 文件 (.jsonl)，简单可靠，不依赖数据库
"""
import json
import os
import time
import uuid
from typing import Optional
from dataclasses import dataclass, field, asdict
from . import config


@dataclass
class TradeRecord:
    """单笔交易完整记录"""
    # ── 标识 ──
    trade_id: str = ""                          # 内部唯一 ID (UUID)
    hibt_order_id: Optional[str] = None         # HIBT 返回的 order_id (如果有)
    client_order_id: str = ""                   # 我们生成的订单号

    # ── 品种与方向 ──
    symbol: str = ""
    direction: str = ""                         # "CALL" or "PUT"
    direction_int: int = 0                      # 1=CALL, 2=PUT

    # ── 时间 ──
    entry_time_ms: int = 0                      # 下单时间戳 (ms)
    expiry_time_ms: int = 0                     # 预计到期时间戳 (ms)
    expiry_minutes: int = 15                    # 持仓期限

    # ── 价格 ──
    entry_price: float = 0.0                    # 入场参考价格 (下单时)
    expiry_price: Optional[float] = None        # 到期结算价格 (结算后填入)

    # ── 金额 ──
    stake_usd: int = 0                          # 下注金额 (整数 USDT, >= 3)
    bet_fraction: float = 0.0                   # 占净值比例

    # ── 赔付率 ──
    payout_ratio: float = 0.0                   # 总返还比例 (含本金, 如 1.80)
    net_payout_ratio: float = 0.0               # 净盈利比例 (如 0.80)
    payout_source: str = ""                     # "api" / "hardcoded" / "estimated"

    # ── 概率 (模型侧) ──
    raw_probability: float = 0.0                # 模型原始预测概率
    calibrated_probability: float = 0.0         # 校准后概率
    conservative_probability: float = 0.0       # 保守概率 (扣除 margin)

    # ── Edge ──
    break_even_probability: float = 0.0         # 盈亏平衡概率
    probability_edge: float = 0.0               # 概率优势 (calibrated - be)
    expected_roi: float = 0.0                   # 每投入1U的期望ROI
    effective_edge: float = 0.0                 # 有效优势 (所有penalty后)

    # ── 不确定性 ──
    uncertainty_margin: float = 0.0             # 不确定性折扣 (probability point)
    calibration_margin: float = 0.0             # 校准折扣
    model_degradation_margin: float = 0.0       # 模型退化折扣
    sample_uncertainty_margin: float = 0.0      # 样本不足折扣

    # ── 模型版本 ──
    model_version: str = ""                     # 模型版本标识
    feature_version: str = ""                   # 特征版本
    regime: str = ""                            # 市场状态
    meta_model_probability: float = 0.0         # Meta Model 输出

    # ── Expert 投票 ──
    expert_votes: dict = field(default_factory=dict)  # {expert_name: probability}

    # ── 决策快照 ──
    decision_snapshot: dict = field(default_factory=dict)  # 完整决策上下文

    # ── 结果 ──
    result: str = ""                            # "WIN" / "LOSS" / "TIE" / "PENDING" / "REJECTED"
    realized_pnl: Optional[float] = None        # 已实现盈亏
    settlement_status: str = ""                 # "PENDING" / "CONFIRMED" / "ESTIMATED" / "REJECTED"

    # ── 拒绝原因 (如果被拒) ──
    reject_reason: str = ""                     # 拒绝原因代码
    reject_detail: str = ""                     # 拒绝详细说明

    # ── 元数据 ──
    tags: list = field(default_factory=list)    # 分类标签
    notes: str = ""                             # 备注
    created_at: str = ""                        # ISO 时间戳
    settled_at: Optional[str] = None            # 结算时间

    def to_dict(self) -> dict:
        d = asdict(self)
        # 清理 None 值
        return {k: v for k, v in d.items() if v is not None or k in ("expiry_price", "realized_pnl", "settled_at", "hibt_order_id")}

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class TradeLedger:
    """
    交易账本 — 记录所有交易决策。

    特点:
    - JSON Lines 格式 (.jsonl)，每行一条记录
    - 支持追加写入（不重写整个文件）
    - 支持查询过滤
    - 拒绝交易也记录
    - 线程安全（通过文件锁）
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "data", "trade_ledger.jsonl"
        )
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        self._lock = __import__('threading').Lock()

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _generate_client_order_id(self, symbol: str, direction: str) -> str:
        ts = int(time.time() * 1000)
        return f"{symbol.lower().replace('usdt','')}_{direction.lower()}_{ts}"

    def create_record(
        self,
        symbol: str,
        direction: str,
        direction_int: int,
        entry_time_ms: int,
        entry_price: float,
        stake_usd: int,
        expiry_minutes: int,
        raw_probability: float = 0.0,
        calibrated_probability: float = 0.0,
        break_even_probability: float = 0.0,
        expected_roi: float = 0.0,
        effective_edge: float = 0.0,
        payout_ratio: float = 0.0,
        net_payout_ratio: float = 0.0,
        payout_source: str = "hardcoded",
        regime: str = "",
        expert_votes: Optional[dict] = None,
        model_version: str = "",
        reject_reason: str = "",
        reject_detail: str = "",
        decision_snapshot: Optional[dict] = None,
        **kwargs,
    ) -> TradeRecord:
        """创建一条交易记录"""
        trade_id = self._generate_id()
        client_order_id = self._generate_client_order_id(symbol, direction)

        record = TradeRecord(
            trade_id=trade_id,
            client_order_id=client_order_id,
            symbol=symbol,
            direction=direction,
            direction_int=direction_int,
            entry_time_ms=entry_time_ms,
            expiry_time_ms=entry_time_ms + expiry_minutes * 60000,
            expiry_minutes=expiry_minutes,
            entry_price=entry_price,
            stake_usd=stake_usd,
            payout_ratio=payout_ratio,
            net_payout_ratio=net_payout_ratio,
            payout_source=payout_source,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            break_even_probability=break_even_probability,
            expected_roi=expected_roi,
            effective_edge=effective_edge,
            regime=regime,
            expert_votes=expert_votes or {},
            model_version=model_version,
            decision_snapshot=decision_snapshot or {},
            reject_reason=reject_reason,
            reject_detail=reject_detail,
            result="REJECTED" if reject_reason else "PENDING",
            settlement_status="REJECTED" if reject_reason else "PENDING",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            tags=[],
        )

        return record

    def save(self, record: TradeRecord) -> bool:
        """追加写入一条记录到账本"""
        with self._lock:
            try:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                return True
            except Exception as e:
                print(f"[TradeLedger] 写入失败: {e}", flush=True)
                return False

    def update_settlement(
        self,
        trade_id: str,
        result: str,
        realized_pnl: float,
        expiry_price: Optional[float] = None,
        settlement_status: str = "CONFIRMED",
        hibt_order_id: Optional[str] = None,
    ) -> bool:
        """
        更新已有记录的结算信息。
        通过重写整个文件实现（.jsonl 文件通常不大）。
        """
        records = self.load_all()
        updated = False

        for i, rec in enumerate(records):
            if rec.trade_id == trade_id or rec.client_order_id == trade_id:
                rec.result = result
                rec.realized_pnl = realized_pnl
                rec.settlement_status = settlement_status
                rec.settled_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                if expiry_price is not None:
                    rec.expiry_price = expiry_price
                if hibt_order_id is not None:
                    rec.hibt_order_id = hibt_order_id
                records[i] = rec
                updated = True
                break

        if updated:
            self._rewrite_all(records)
        return updated

    def update_hibt_order_id(self, trade_id: str, hibt_order_id: str) -> bool:
        """回填 HIBT 返回的 order_id"""
        return self._update_field(trade_id, "hibt_order_id", hibt_order_id)

    def _update_field(self, trade_id: str, field: str, value) -> bool:
        records = self.load_all()
        updated = False
        for i, rec in enumerate(records):
            if rec.trade_id == trade_id or rec.client_order_id == trade_id:
                setattr(rec, field, value)
                records[i] = rec
                updated = True
                break
        if updated:
            self._rewrite_all(records)
        return updated

    def _rewrite_all(self, records: list):
        with self._lock:
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[TradeLedger] 重写失败: {e}", flush=True)

    def load_all(self) -> list[TradeRecord]:
        """读取所有记录"""
        records = []
        if not os.path.exists(self.filepath):
            return records
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        records.append(TradeRecord.from_dict(d))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception as e:
            print(f"[TradeLedger] 读取失败: {e}", flush=True)
        return records

    def query(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        expiry_minutes: Optional[int] = None,
        regime: Optional[str] = None,
        result: Optional[str] = None,
        settlement_status: Optional[str] = None,
        from_time_ms: Optional[int] = None,
        to_time_ms: Optional[int] = None,
        min_probability: Optional[float] = None,
        max_probability: Optional[float] = None,
        min_edge: Optional[float] = None,
        reject_reason: Optional[str] = None,
        limit: int = 0,
    ) -> list[TradeRecord]:
        """查询过滤记录"""
        records = self.load_all()
        results = []

        for rec in records:
            if symbol and rec.symbol != symbol:
                continue
            if direction and rec.direction != direction:
                continue
            if expiry_minutes and rec.expiry_minutes != expiry_minutes:
                continue
            if regime and rec.regime != regime:
                continue
            if result and rec.result != result:
                continue
            if settlement_status and rec.settlement_status != settlement_status:
                continue
            if from_time_ms and rec.entry_time_ms < from_time_ms:
                continue
            if to_time_ms and rec.entry_time_ms > to_time_ms:
                continue
            if min_probability is not None and rec.raw_probability < min_probability:
                continue
            if max_probability is not None and rec.raw_probability > max_probability:
                continue
            if min_edge is not None and rec.effective_edge < min_edge:
                continue
            if reject_reason and rec.reject_reason != reject_reason:
                continue
            results.append(rec)

        if limit > 0:
            results = results[-limit:]
        return results

    def get_stats(self, records: Optional[list[TradeRecord]] = None) -> dict:
        """计算统计信息"""
        if records is None:
            records = self.load_all()

        settled = [r for r in records if r.result in ("WIN", "LOSS", "TIE")]
        wins = [r for r in settled if r.result == "WIN"]
        losses = [r for r in settled if r.result == "LOSS"]
        ties = [r for r in settled if r.result == "TIE"]

        total = len(settled)
        win_count = len(wins)
        loss_count = len(losses)
        tie_count = len(ties)
        win_rate = win_count / total if total > 0 else 0.0

        total_pnl = sum(r.realized_pnl or 0 for r in settled)
        total_staked = sum(r.stake_usd for r in settled)
        roi = total_pnl / total_staked if total_staked > 0 else 0.0

        avg_predicted_prob = sum(r.raw_probability for r in settled) / total if total > 0 else 0.0
        avg_calibrated_prob = sum(r.calibrated_probability for r in settled) / total if total > 0 else 0.0

        # Brier Score
        brier = 0.0
        for r in settled:
            actual = 1.0 if r.result == "WIN" else 0.0
            brier += (r.calibrated_probability - actual) ** 2
        brier /= total if total > 0 else 1

        return {
            "total_trades": len(records),
            "settled": total,
            "wins": win_count,
            "losses": loss_count,
            "ties": tie_count,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "roi": round(roi, 4),
            "avg_predicted_prob": round(avg_predicted_prob, 4),
            "avg_calibrated_prob": round(avg_calibrated_prob, 4),
            "brier_score": round(brier, 4),
            "avg_stake": round(total_staked / total, 2) if total > 0 else 0,
        }

    def get_stats_by_symbol(self) -> dict:
        """按品种统计"""
        records = self.load_all()
        symbols = set(r.symbol for r in records)
        return {s: self.get_stats(self.query(symbol=s)) for s in symbols}

    def get_stats_by_edge_bucket(self) -> dict:
        """按 Edge 区间统计"""
        records = self.load_all()
        settled = [r for r in records if r.result in ("WIN", "LOSS", "TIE")]
        buckets = {
            "negative": [],
            "0-1%": [],
            "1-3%": [],
            "3-5%": [],
            "5-7%": [],
            "7-10%": [],
            "10%+": [],
        }
        for r in settled:
            edge = r.effective_edge
            if edge < 0:
                buckets["negative"].append(r)
            elif edge < 0.01:
                buckets["0-1%"].append(r)
            elif edge < 0.03:
                buckets["1-3%"].append(r)
            elif edge < 0.05:
                buckets["3-5%"].append(r)
            elif edge < 0.07:
                buckets["5-7%"].append(r)
            elif edge < 0.10:
                buckets["7-10%"].append(r)
            else:
                buckets["10%+"].append(r)
        return {k: self.get_stats(v) for k, v in buckets.items() if v}

    def get_pending_settlements(self) -> list[TradeRecord]:
        """获取所有待结算的交易"""
        return self.query(settlement_status="PENDING")

    def get_recent(self, n: int = 50) -> list[TradeRecord]:
        """获取最近 n 条记录"""
        records = self.load_all()
        return records[-n:]

    def count(self) -> int:
        return len(self.load_all())


# ── 全局单例 ──
_ledger_instance: Optional[TradeLedger] = None


def get_trade_ledger() -> TradeLedger:
    global _ledger_instance
    if _ledger_instance is None:
        _ledger_instance = TradeLedger()
    return _ledger_instance