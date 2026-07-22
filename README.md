# AtradeBot 🚀

基于 **XGBoost + Ensemble Experts** 的 **15 分钟事件合约** 量化交易系统。

支持 BTC/ETH/SOL，实时信号 → 下单 → 结算 → 回测全链路。

## 架构总览

```
lib/engine/
├── engine.py              # 主引擎：信号循环 + 交易执行 + 生命周期管理
├── edge_engine.py         # Fast Entry 实时推理引擎（EventEdge V2）
├── models.py              # XGBoost 模型加载 + 多时间框架特征
├── config.py              # 全局配置：风控参数、模式切换、符号列表
├── exchange.py            # HIBT 交易所 API 封装（下单/撤单/仓位查询）
├── portfolio_risk.py      # 组合风控：凯利公式、最大仓位、最大回撤限制
├── backtester.py          # 回测引擎：Walk-Forward 验证
├── settlement_ledger.py   # 结算检测 + 盈亏记录
├── trade_ledger.py        # 交易账本
├── shadow_mode.py         # 影子模式：无实盘下单，用于新策略验证
├── regime_detector.py     # 市场状态检测（趋势/震荡/高波动）
├── opportunity_ranker.py  # 多符号机会排序
├── multi_timeframe_features.py  # 15m/5m/1m 多时间框架特征工程
├── probability_calibrator.py    # 概率校准（Isotonic / Platt Scaling）
├── uncertainty.py         # 不确定性量化
├── model_health.py        # 模型健康监测
├── label_builder.py       # 标签构建（事件合约到期涨跌）
├── live_data.py           # 实时行情数据
├── realtime_feed.py       # WebSocket 实时数据推送
└── experts/               # 专家集成分系统
    ├── ensemble_expert_manager.py  # 专家管理器
    ├── trend_expert.py             # 趋势专家
    ├── mean_reversion_expert.py    # 均值回归专家
    └── volatility_breakout_expert.py  # 波动率突破专家
```

## 策略逻辑

- **品种**：BTCUSDT、ETHUSDT、SOLUSDT（15 分钟结算）
- **信号**：Fast Entry（1m 实时推理）+ Slow Model（15m 闭环 K 线）
- **风控**：凯利滚仓、最大 35% 仓位（小账户）、每符号最多 1 张合约、反向持仓拦截
- **模式**：LIVE（实盘）/ SHADOW（影子验证，不下单）/ PAPER（模拟）

## 快速开始

```bash
# 安装依赖
npm install
pip install -r requirements.txt

# 启动前端
npm run dev

# 启动引擎（通过前端 API 控制）
# 访问 http://localhost:3000 → Engine Control → Start
```

## 部署到 VPS

```bash
# 一键同步：本地 → GitHub → VPS
python scripts/git_sync.py
```

## 训练模型

```bash
python scripts/train_fast_entry.py      # 训练 Fast Entry 模型
python scripts/train_ensemble_v3.py     # 训练专家集成模型
python scripts/retrain.py               # 增量重训练
```

## 回测 & 审计

```bash
python scripts/backtest_fast_entry.py   # Fast Entry 回测
python scripts/backtest_trading.py      # 完整交易回测
python scripts/safety_audit_test.py     # 安全审计
python scripts/verify_eventedge_pipeline.py  # 流水线验证
python scripts/audit_fast_entry.py      # Fast Entry 审计
```

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Next.js 15 + React + TypeScript |
| 后端 | Next.js API Routes + SSE 推送 |
| 引擎 | Python + XGBoost + scikit-learn |
| 交易所 | HIBT API v2（BTC/ETH/SOL 事件合约） |

## License

MIT
