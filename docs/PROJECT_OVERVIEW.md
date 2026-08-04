# ATradeBot 项目概览

## 做什么的
HIBT 二元期权自动交易机器人。在 Gate.io 拉 K 线 → 算特征 → 模型预测 15 分钟涨跌 → 通过 HIBT Event Contract 下单（CALL/PUT）。到期后看余额变化判断胜负。

## 技术栈
- **前端/管理器**: Next.js (App Router)，TypeScript — Web 面板控制启停
- **引擎核心**: Python 3，完全独立进程，由 Next.js ProcessManager spawn
- **交易所**: Gate.io API 拉 K 线，HIBT API 下单（币安链 BSC 合约）
- **AI 模型**: LightGBM + Scikit-learn StandardScaler
- **部署**: Windows 本地开发 → git push → VPS (Ubuntu, PM2, Docker)

## 项目结构
```
atradebot/
├── lib/engine/           ← 核心引擎 (Python)
│   ├── main.py           ← 入口，stdin/stdout JSON 行协议
│   ├── engine.py         ← 主循环：tick() → _run_fast_entry_scan() → _execute_opportunity() → _check_settlement()
│   ├── edge_engine.py    ← Edge 计算：保守概率 - 盈亏平衡点 = effective_edge
│   ├── opportunity_ranker.py  ← 候选排序：risk_adjusted_ev 排序，选 top N
│   ├── portfolio_risk.py ← 渐进式凯利仓位计算（confidence ∈ [0.25, 0.50]）
│   ├── config.py         ← 所有可调参数集中在这里
│   ├── multi_timeframe_features.py ← 31 个特征计算（1m + 5m 多周期）
│   ├── experts/ensemble_expert_manager.py ← Slow 15m LightGBM 模型加载
│   ├── realtime_feed.py  ← Gate.io WebSocket/轮询实时数据
│   ├── exchange.py       ← HIBT 下单 API
│   ├── trade_ledger.py   ← 交易记录
│   └── settlement_ledger.py ← 结算推断（HIBT 不返回逐单结果，只能看余额变化反推）
├── scripts/
│   ├── train_fast_entry.py  ← Fast 1m 模型训练
│   ├── train_ensemble_v3.py ← Slow 15m 模型训练
│   ├── backtest_fast_entry.py ← 原始回测
│   ├── backtest_quality.py   ← 新增：新旧参数对比回测
│   └── git_sync.py        ← 一键部署到 VPS
├── models/               ← 所有训练好的模型 .pkl
└── app/api/              ← Next.js API routes（控制面板后端）
```

## 交易决策全链路
```
RealtimeFeed (每 5s)
  ↓
_fast_entry_scan() 对 BTC/ETH/SOL 各扫一遍：
  │
  ├─ Step 1: compute_fast_entry_features() → 31 特征向量
  ├─ Step 2: Fast Model (LightGBM, 1m) → fast_prob ∈ [0,1]
  ├─ Step 3: Slow Model (LightGBM, 15m) → slow_prob ∈ [0,1]
  ├─ Step 4: _fuse_probabilities() → ensemble_prob (85% Fast + 15% Slow)
  │         direction = CALL if ensemble_prob ≥ 0.50 else PUT
  │
  ├─ Step 5: 方向强度检查  ← NEW，概率偏离 0.50 < 8% → WEAK_DIRECTION 拒绝
  ├─ Step 6: EdgeEngine.compute() → 扣除 margin → effective_edge + expected_roi
  ├─ Step 7: CHOP 检测（震荡行情） / 集中度（同币种 3 次后冷却）
  ├─ Step 8: 冷却 / 持仓上限 / 每小时上限 检查
  │
  ├─ Step 9: OpportunityRanker.rank() → 按 risk_adjusted_ev 排序 → 选 1
  └─ Step 10: PortfolioRiskManager.compute_position_size() → 渐进凯利 → 下单
```

## 两层模型

| | Slow Model (15m) | Fast Model (1m) |
|---|---|---|
| 文件名 | `{sym}_15m_ensemble_v3.pkl` | `{sym}_fast_entry.pkl` |
| 训练 | `train_ensemble_v3.py` | `train_fast_entry.py` |
| 频率 | 每 15min 更新 | 每 5s 扫描 |
| 算法 | LightGBM ensemble | LightGBM 500 树 max_depth=5 |
| 特征数 | 31 个（时间特征 + MACD/RSI/BB/ADX 等） | 31 个（1m 动量/RSI/波动率 + 5m ADX/BB + SlowContext） |
| 目的 | 提供中长期方向背景 | 实时捕捉短期方向变化 |

融合权重：Fast 权重基于 Slow 的置信度动态调整（0.50~0.85），默认偏向 Fast。

## 三个模型质量（7 天回测实测）

| 币种 | prob_mean | prob_std | 新参数胜率 | 旧参数胜率 | 评价 |
|---|---|---|---|---|---|
| **BTC** | 0.5000 | 0.064 | 0 笔（全被过滤） | 87%（回测后见之明） | **废了**，概率全挤在 0.50 附近 |
| **ETH** | 0.5108 | 0.1308 | 70.1% | 65.9% | **唯一好模型**，真正有方向判别力 |
| **SOL** | 0.5223 | 0.1011 | 60.0% | 58.0% | 勉强可用，薄利 |

## 最新改动（commit d6a39ea）— 质量管理

改动动机：之前 ensemble_prob ≥ 0.50 就开单，Edge 门槛 2%，几乎来单不拒。昨天全输。

| 参数 | 旧值 | 新值 | 说明 |
|---|---|---|---|
| MIN_DIRECTION_STRENGTH | （无） | 0.08 | 概率必须偏离 0.50 至少 8% |
| MIN_EFFECTIVE_EDGE | 0.02 | 0.05 | 最小有效优势 5% |
| MIN_EXPECTED_ROI | 0.005 | 0.03 | 最小期望 ROI 3% |
| UNCERTAINTY_MARGIN | 0.02 | 0.03 | 不确定性折扣 |
| CALIBRATION_MARGIN | 0.01 | 0.015 | 校准折扣 |
| KELLY_FRACTION | 0.50 | 0.25 | 1/4 凯利 |
| MAX_ACTIVE_EVENT_CONTRACTS | 3 | 1 | 同一时间只持 1 单 |
| MAX_NEW_TRADES_PER_HOUR | 4 | 3 | 每小时最多 3 笔 |
| TRADE_COOLDOWN_SEC | 120 | 300 | 冷却 5 分钟 |
| SIGNA_COOLDOWN_SECONDS | 60 | 60 | 不变 |
| max_selected (ranker) | 3 | 1 | 每周期只选最优 1 个 |

## 核心痛点

1. **BTC 模型随机**：概率 mean=0.50，无法偏离。新参数下 0 笔合格 → 不开单总比送钱好，但丢了 BTC 交易机会
2. **只有 ETH 靠谱**：概率分布有分散度（std=13%），过滤后胜率 70%+。问题是 ETH 的 payout 0.80 < BTC 的 0.818
3. **SLOW 模型把 1 个模型拆成 3 个虚拟专家 + 随机噪声再平均**：ExpertManager 加载 `_ensemble_v3.pkl`（单个模型），然后复制成 3 份各加 ±1% 随机噪声，再加权平均回去 → 等于白走一圈
4. **模型校准缺失**：WalkForwardCalibrator 用的是 isotonic，但从未通过验证（`calibration_ready = false`），所以实际用的概率未校准
5. **HIBT 不返回逐单结算结果**：靠 30 秒余额变化反推胜负，多单同时到期时无法正确归因
6. **7 天回测太短**：样本量小，结果可信度有限
7. **Payout 是硬编码的**：不是从 HIBT API 实时获取，如果交易所调低某个币种的赔付率，模型不知道
