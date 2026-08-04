# 交易闭环可靠性重构 — 实施计划

## 当前状态

| 文件 | 职责 |
|---|---|
| `engine.py` | 主循环，70% 逻辑混杂在里面（扫描+融合+过滤+下单+结算） |
| `edge_engine.py` | Edge 计算 |
| `opportunity_ranker.py` | 候选排序 |
| `portfolio_risk.py` | 凯利仓位 |
| `config.py` | 全局参数 |
| `trade_ledger.py` | JSONL 交易记录 |
| `settlement_ledger.py` | 余额推算结算 |
| `exchange.py` | HIBT 下单 |
| `realtime_feed.py` | Gate.io 数据 |
| `shadow_mode.py` | 运行模式管理 |
| `probability_calibrator.py` | Walk-forward 校准 |
| `signal_validator.py` | L0-L2 信号验证 |
| `risk_manager.py` | L3-L5 风控 |
| `experts/` | Slow model ExpertManager + LGBMExpert v2 |
| `multi_timeframe_features.py` | 31 特征计算 |

## 重构范围

### Phase 1: 数据层（SQLite 替换 JSONL）

**新建**: `lib/engine/data/__init__.py`
**新建**: `lib/engine/data/schema.py` — SQLite 建表 DDL

两张表替换 `trade_ledger.jsonl`：
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    symbol TEXT, direction TEXT, direction_int INTEGER,
    signal_time_ms INTEGER, submit_time_ms INTEGER,
    accepted_time_ms INTEGER, expiry_time_ms INTEGER,
    gate_price_at_signal REAL, hibt_open_price REAL, hibt_close_price REAL,
    raw_probability REAL, calibrated_probability REAL,
    slow_probability REAL, fast_probability REAL,
    payout_call REAL, payout_put REAL, expected_value REAL,
    stake REAL, model_version TEXT, feature_version TEXT,
    order_status TEXT, settlement_status TEXT,
    actual_result TEXT, actual_pnl REAL,
    balance_before REAL, balance_after REAL,
    error_message TEXT, run_mode TEXT, reject_reason TEXT,
    created_at TEXT, settled_at TEXT
);

CREATE TABLE settlement_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT, hibt_order_id TEXT,
    open_price REAL, close_price REAL,
    payout_call REAL, payout_put REAL,
    result TEXT, pnl REAL,
    source TEXT, -- "hibt_api" / "hibt_closed_orders" / "gate_price_proxy"
    raw_response TEXT, created_at TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);
```

**修改**: `lib/engine/trade_ledger.py` → 重写为 `lib/engine/data/trade_ledger.py`，SQLite 实现，保持 Query API 兼容

### Phase 2: 结算层

**新建**: `lib/engine/settlement/reconciler.py`

`SettlementReconciler`:
- `check_hibt_closed_orders(trade_id)` — 查 HIBT 已关闭订单列表，匹配 trade_id
- `reconcile(trade: TradeRecord)` — 逐单对账，settlement_source="hibt_closed_orders"
- `can_settle_via_hibt() -> bool` — 如果 HIBT 不返回逐单数据 → False
- LIVE gate 检查：`if not reconciiler.can_settle_via_hibt(): block_live()`

**新建**: `lib/engine/settlement/__init__.py`

### Phase 3: 运行模式 (覆盖现有 shadow_mode.py)

**新建**: `lib/engine/run_mode.py`
- `class RunMode(Enum)`: PAPER, SHADOW, LIVE
- `class LiveGate`: 集中检查所有 LIVE 前置条件
- `auto_downgrade_on_failure()` — 异常自动降级
- `KILL_SWITCH = threading.Event()` — 全局 kill switch

删除 `shadow_mode.py` 的 RunMode/SymbolMode/CalibrationStatus 旧逻辑，保留统计功能。

### Phase 4: 决策引擎重写 (覆盖 engine.py 核心)

**新建**: `lib/engine/strategy/decision_engine.py`

```python
class DecisionEngine:
    def evaluate(self, symbol, fast_prob, slow_prob, payout_call, payout_put, calibrated) -> Decision:
        """
        输入: p_up (校准后概率), payout_call_net, payout_put_net
        计算:
          ev_call = p_up * payout_call_net - (1 - p_up)
          ev_put  = (1 - p_up) * payout_put_net - p_up
        输出: max(ev_call, ev_put) 的方向，或 ABSTAIN
        """
```

**删除**: engine.py 中 `_fuse_probabilities`、`_run_fast_entry_scan` 中的固定 "≥0.50 → CALL" 逻辑
**删除**: config.py 中 `MIN_DIRECTION_STRENGTH`

### Phase 5: 风险管理

**新建**: `lib/engine/risk/risk_manager.py`
- 日亏损限制
- 连续亏损暂停
- 单日最大交易次数
- 数据断流检查
- 赔率获取失败暂停
- 结算对账失败暂停 → 降级到 SHADOW
- 订单幂等键
- API 超时后先查询订单状态

覆盖旧 `risk_manager.py`。

### Phase 6: Payout 实时获取

**修改**: `lib/engine/exchange.py`
- 新增 `fetch_hibt_payout(symbol, expiry_minutes)` — 调用 HIBT 合约发现 API
- 新增 `fetch_hibt_closed_orders()` — 获取已关闭订单列表

### Phase 7: 回测 (Walk-Forward)

**新建**: `lib/engine/backtest/walk_forward.py`

严格按时间顺序：
1. 起始索引 → 训练窗口 [t0, t1) → 验证窗口 [t1, t1+gap] 跳过 → 测试窗口 [t1+gap, t2]
2. 每个训练窗口内独立 fit StandardScaler + LightGBM + ProbabilityCalibrator
3. 不允许对整个数据先 fit
4. 15 min 标签重叠 → gap >= 15 bars
5. 未收盘 K 线不能用于特征
6. SlowContext = 信号发生前已完成的 slow 模型结果
7. 输出要求的全部指标

### Phase 8: 概率校准增强

**修改**: `lib/engine/probability_calibrator.py`
- 增加 sigmoid (Platt) calibration
- 比较 no_cal / sigmoid / isotonic
- 校准样本不足时 → isotonic not ready → 禁止 LIVE

### Phase 9: 模型注册

**新建**: `lib/engine/models/model_registry.py`
- 注册所有模型版本
- 标记哪些模型通过了样本外验证
- BTC 默认禁用，需 oos 测试通过后手动启用

### Phase 10: 集成 & 测试

**修改**: `lib/engine/engine.py` — 只保留主循环骨架，调用各模块
**修改**: `lib/engine/main.py` — 支持 `--mode paper/shadow/live`
**新建**: `tests/` — 单元测试
**新建**: `docs/CHANGELOG.md`
**新建**: `scripts/init_db.py`
**新建**: `tests/test_live_gate.py`

## 关键设计决策

1. **SQLite 替换 JSONL**: 当前 JSONL 每次 `_update_field` 都要 `rewrite_all`，O(n²)。SQLite 支持 UPDATE WHERE。
2. **默认 SHADOW，LIVE 需显式开启**: `ENABLE_LIVE_TRADING=true` + 管理后台确认 + 程序重启自动回 SHADOW
3. **EV 公式**: 不硬编码方向阈值，用期望值公式决定方向。ABSTAIN 是合法输出。
4. **HIBT 没有逐单结算数据时禁止 LIVE**: 这是最核心的安全性要求
5. **保留现有 Next.js ProcessManager 通信方式**: stdin/stdout JSON 行协议不变

## 文件清单

### 新建 (17 files)
- `lib/engine/data/__init__.py`
- `lib/engine/data/schema.py`
- `lib/engine/data/trade_ledger.py`
- `lib/engine/settlement/__init__.py`
- `lib/engine/settlement/reconciler.py`
- `lib/engine/run_mode.py`
- `lib/engine/strategy/__init__.py`
- `lib/engine/strategy/decision_engine.py`
- `lib/engine/risk/__init__.py`
- `lib/engine/risk/risk_manager.py` (new)
- `lib/engine/models/model_registry.py`
- `lib/engine/backtest/__init__.py`
- `lib/engine/backtest/walk_forward.py`
- `scripts/init_db.py`
- `tests/__init__.py`
- `tests/test_live_gate.py`
- `docs/CHANGELOG.md`

### 修改 (7 files)
- `lib/engine/engine.py` — 重写决策逻辑，只保留主循环骨架
- `lib/engine/config.py` — 删除旧阈值，增加新配置项
- `lib/engine/exchange.py` — 增加 payout API + closed orders API
- `lib/engine/probability_calibrator.py` — 增加 sigmoid
- `lib/engine/main.py` — 支持 --mode paper/shadow/live
- `lib/engine/models.py` — 增加 ABSTAIN/Decision 数据类
- `lib/engine/shadow_mode.py` — 简化，统计功能保留

### 删除 (3 files)
- `lib/engine/risk_manager.py` — 替换为 `risk/` 包
- `lib/engine/signal_validator.py` — 决策逻辑统一到 decision_engine
- `lib/engine/opportunity_ranker.py` — EV 公式替代 rank_score 排序

## 估计工作量
- Phase 1 (数据层): ~1h
- Phase 2 (结算): ~45m
- Phase 3 (运行模式): ~30m
- Phase 4 (决策引擎): ~1h
- Phase 5 (风险管理): ~45m
- Phase 6 (Payout API): ~30m
- Phase 7 (回测): ~1.5h
- Phase 8 (校准): ~30m
- Phase 9 (模型注册): ~30m
- Phase 10 (集成+测试): ~1h

总计: ~8h
