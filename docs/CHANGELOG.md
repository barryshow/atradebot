# CHANGELOG — 交易闭环可靠性重构

## 概述

将 ATradeBot 从"固定阈值概率猜测"重构为 EV（期望值）驱动、三层运行模式（PAPER / SHADOW / LIVE）、依赖 HIBT 结算对账的可靠交易系统。

## 新建文件

### 数据层
- **lib/engine/data/__init__.py** — 数据层包入口
- **lib/engine/data/schema.py** — SQLite 建表 DDL（trades / settlement_events / model_registry），WAL 模式
- **lib/engine/data/trade_ledger.py** — SQLite TradeLedger，替换原 JSONL。支持 mark_submitted/mark_accepted/mark_rejected/mark_settled/mark_abstain 生命周期管理

### 结算层
- **lib/engine/settlement/__init__.py** — 结算包入口
- **lib/engine/settlement/reconciler.py** — SettlementReconciler，对接 HIBT historyOrderList API。实时 payout 获取、逐单对账、can_settle_via_hibt() 探测

### 运行模式
- **lib/engine/run_mode.py** — RunModeManager + LiveGate + 全局 kill switch。默认 SHADOW，重启自动恢复。LIVE 需 6 项硬性条件全通过

### 决策引擎
- **lib/engine/strategy/__init__.py** — 策略包入口
- **lib/engine/strategy/decision_engine.py** — EV 驱动决策：`ev_call = p_up * payout_call_net - (1-p_up)`。只有 max(ev_call, ev_put) ≥ 0.03 才开单，否则 ABSTAIN

### 风险管理
- **lib/engine/risk/__init__.py** — 风控包入口
- **lib/engine/risk/risk_manager.py** — RiskManager v2：最多 1 单、固定最小金额、日亏损限制、连续亏损暂停、单日最大交易次数、数据断流检测、订单幂等键、BTC 默认禁用

### 模型管理
- **lib/engine/models/__init__.py** — 模型包入口
- **lib/engine/models/model_registry.py** — ModelRegistry：模型版本注册、OOS 验证记录、LIVE 准入检查

### 脚本与测试
- **scripts/init_db.py** — 数据库初始化脚本
- **tests/__init__.py** — 测试包
- **tests/test_live_gate.py** — LIVE 门控测试：9 个场景，验证 6 种阻断条件 + 全条件通过 + 重启恢复 SHADOW

### 配置
- **.env.example** — 完整环境变量模板

## 修改文件

### lib/engine/engine.py
**完全重写。**
- 删除 `_fuse_probabilities()` — 不再加权平均
- 删除 "ensemble_prob >= 0.50 → CALL" 逻辑
- 删除 `CandidateOpportunity`、`EdgeEngine` 依赖
- 新增 EV 驱动扫描流程：`RealtimeFeed → Fast/Slow Model → Payout → DecisionEngine.evaluate() → RiskManager → TradeLedger`
- 新增 `_check_settlement()` — 优先 HIBT API 对账，SHADOW 模式使用 Gate 价格代理结算
- 新增 `_settle_shadow_orders()` — 时间到期自动结算
- 保留 Next.js ProcessManager stdin/stdout JSON 行协议

### lib/engine/config.py
**新增配置项：**
- `ENABLE_LIVE_TRADING` — LIVE 全局开关
- `DEFAULT_RUN_MODE` — 默认 "shadow"
- `MIN_EXPECTED_VALUE` — 最小 EV 阈值 (3%)
- `PAYOUT_REQUIRE_API_FOR_LIVE` — LIVE 要求实时 payout
- `SETTLEMENT_REQUIRE_HIBT_FOR_LIVE` — LIVE 要求 HIBT 结算

## 未修改文件（保持兼容）
- `lib/engine/exchange.py` — place_order 不变，新增 direction mapping 不受影响
- `lib/engine/realtime_feed.py` — 不变
- `lib/engine/multi_timeframe_features.py` — 31 特征计算不变
- `lib/engine/main.py` — 不变（stdin/stdout JSON 协议保持）
- `lib/engine/experts/` — Slow model ExpertManager 不变
- `lib/engine/probability_calibrator.py` — WalkForwardCalibrator 不变
- `lib/engine/notifier.py` — 飞书通知不变
- `app/` — Next.js 前端不变

## 删除的旧逻辑
- ~~固定阈值 "ensemble_prob ≥ 0.50 → CALL"~~ → 替换为 EV 公式
- ~~MIN_DIRECTION_STRENGTH (0.08)~~ → 不再需要，EV 自然过滤弱信号
- ~~余额变化推断单笔输赢~~ → 替换为 HIBT 逐单对账
- ~~JSONL trade_ledger.jsonl 的全量重写~~ → 替换为 SQLite UPDATE

## 启动命令

```bash
# PAPER (回测)
ENABLE_LIVE_TRADING=false BACKTEST_MODE=true python lib/engine/main.py --auto --mode paper

# SHADOW (默认，实时行情但不下单)
python lib/engine/main.py --auto --mode shadow

# LIVE (需 ENABLE_LIVE_TRADING=true + 管理面板确认)
ENABLE_LIVE_TRADING=true python lib/engine/main.py --auto --mode live

# 初始化数据库
python scripts/init_db.py

# 运行测试
python tests/test_live_gate.py
```
