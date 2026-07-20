# HIBT Event Contract Capability Audit

**Date:** 2026-07-19
**Scope:** 审计 atradebot 项目中所有与 HIBT 交易所相关的网络请求、数据流和字段
**Method:** 逐行审计 `lib/engine/exchange.py`, `lib/engine/order_executor.py`, `lib/engine/engine.py`, `lib/engine/config.py` 及所有引用 HIBT 的代码
**Principle:** 不猜测不存在的 endpoint，只记录当前项目已验证可用的接口

---

## 一、HIBT API Endpoint 清单

### 1.1 余额查询 — ✅ AVAILABLE

| 属性 | 值 |
|------|-----|
| **Base URLs** | `https://www.hibt.com`, `https://api-ws.s0e9tu.com`, `https://api.hibt0.com`, `https://api.hibt8.com` |
| **Endpoint** | `GET /rest/c/future/u/user/balance?langCode=zh_CN&v={v}` |
| **HTTP Method** | GET |
| **代码位置** | `exchange.py:107` |
| **认证方式** | Headers: `x-auth-token`, `Authorization`, `bget_key`, `bget_id`, `Request-Id` |
| **稳定可用** | ✅ 已验证可用（4 端点容错） |
| **公开 API** | ❌ 非公开，需要 JWT token |

**Response 字段（已验证）:**
```json
{
  "code": 0,           // 成功: 0, 200, "0", "200"
  "data": {
    "amount": "123.45" // 账户余额 (USDT)
  },
  "msg": "..."
}
```

**错误码（已验证）:**
- `401, 403, 4001, 4003` → token 过期/无效，返回 -2.0

**项目实际使用的返回字段:**
| 字段 | 来源 | 使用方式 |
|------|------|----------|
| `data.amount` | 真实 HIBT API 响应 | 转为 float，作为账户余额 |
| `code` | 真实 HIBT API 响应 | 判断成功/失败/认证过期 |

### 1.2 下单 — ✅ AVAILABLE

| 属性 | 值 |
|------|-----|
| **Base URLs** | 同上 4 个端点 |
| **Endpoint** | `POST /option/option-order/place` |
| **HTTP Method** | POST (form-encoded) |
| **代码位置** | `exchange.py:145-146` |
| **认证方式** | 同上 |
| **稳定可用** | ✅ 已验证可用（实测下单成功） |

**Request 参数（已验证）:**
```python
data = {
    "amount": "3",              // 下注金额 (USDT, 字符串格式)
    "direction": "1",           // 1=CALL, -1=PUT (HIBT 实测)
    "symbol": "btc_usdt",       // 品种 (小写, 下划线分隔)
    "timeUnit": "15",           // 持仓分钟 (字符串格式)
    "langCode": "zh_CN",
}
```

**Request Headers（已验证）:**
```python
{
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Future_source": "1",
    "Client-Type": "web",
    "Hc-Platform": "web",
    "Hc-Language": "zh_CN",
    "Lang": "zh_CN",
    "Platform": "PC",
    "Origin": "https://www.hibt.com",
    "Referer": "https://www.hibt.com/",
    "x-auth-token": "<JWT>",
    "Authorization": "<JWT>",
    "bget_key": "<key>",
    "bget_id": "<id>",
    "Request-Id": "<md5_hash>",
}
```

**Response 解析（当前代码）:**
```python
rj = res.json()
if rj.get("code") in [0, 200, "0", "200"]:
    return OrderResult(ok=True, code=200, msg="下单成功")
return OrderResult(ok=False, code=rj.get("code"), msg=rj.get("msg", res.text[:30]))
```

**项目实际使用的返回字段:**
| 字段 | 来源 | 使用方式 |
|------|------|----------|
| `code` | 真实 HIBT API 响应 | 判断下单成功/失败 |
| `msg` | 真实 HIBT API 响应 | 错误信息展示 |

**⚠️ 未使用的返回字段（可能存在于响应中但项目未解析）:**
| 字段 | 状态 | 备注 |
|------|------|------|
| `order_id` | UNVERIFIED | 未从响应中提取 |
| `contract_id` | UNVERIFIED | 未从响应中提取 |
| `open_price` | UNVERIFIED | 未从响应中提取 |
| `expiry_time` | UNVERIFIED | 未从响应中提取 |
| `payout` | UNVERIFIED | 未从响应中提取 |
| `stake` | UNVERIFIED | 未从响应中提取 |

---

## 二、关键能力矩阵

### 2.1 Order Placement（下单）— ✅ AVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 下 CALL 单 | ✅ AVAILABLE | direction=1 |
| 下 PUT 单 | ✅ AVAILABLE | direction=-1 |
| 指定金额 | ✅ AVAILABLE | amount 参数，字符串格式 |
| 指定期限 | ✅ AVAILABLE | timeUnit 参数，当前固定 "15" |
| 获取 order_id | ❌ NOT IMPLEMENTED | API 响应可能包含但项目未解析 |
| 获取 contract_id | ❌ NOT IMPLEMENTED | 同上 |
| 获取 open_price | ❌ NOT IMPLEMENTED | 同上 |
| 下单去重 | ✅ AVAILABLE | 15 秒内同一品种同方向去重 |

### 2.2 Payout Discovery（赔付率获取）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 下单前获取实时赔付率 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 下单响应中获取赔付率 | ❌ NOT IMPLEMENTED | 响应可能包含但未解析 |
| 硬编码赔付率 | ⚠️ HARDCODED | `PAYOUT_RATES = {"BTCUSDT": 0.818, "ETHUSDT": 0.80, "SOLUSDT": 0.80}` |

**关键问题:**
- 当前赔付率来自 `config.py` 硬编码，**不是动态获取的**
- 无法确认 HIBT 不同品种/不同期限的 payout 是否变化
- 无法确认 HIBT 返回的 payout 含义是"净盈利比例"还是"总返还比例"
- 如果 HIBT 调整赔付率，系统无法感知

**Edge Engine 影响:**
- 无法动态计算 `break_even_probability = 1 / (1 + net_payout_ratio)`
- 无法动态计算 `expected_roi = p * r - (1 - p)`
- **LIVE 模式禁止启用 Dynamic Edge Strategy，直到赔付率可动态获取**

### 2.3 Expiry Discovery（可用期限）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 获取可用期限列表 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 当前使用期限 | ⚠️ HARDCODED | `HOLD_MINUTES=15`，仅此一个值 |
| 多期限支持 | ❌ NOT IMPLEMENTED | 架构不支持 |

**关键问题:**
- 只有 `timeUnit=15` 经过验证
- 不确定 5m/30m/60m 在 HIBT 上是否实际可用
- ContractDiscovery 无法从 API 获取可用期限

**EventEdge V2 影响:**
- 架构设计上支持多期限，但 LIVE 只能启用已验证的 15 分钟
- 其他期限需要真实账户确认后才能加入 LIVE
- 启动时需输出 `EXPIRY_DISCOVERY_UNAVAILABLE` 警告

### 2.4 Order Status（订单状态查询）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 查询单笔订单状态 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 查询订单结算结果 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 获取 entry_price | ❌ UNAVAILABLE | 下单响应未解析 |
| 获取 expiry_price | ❌ UNAVAILABLE | 无结算查询 API |
| 获取订单 result | ❌ UNAVAILABLE | 无结算查询 API |

### 2.5 Settlement Result（结算结果）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| API 查询结算结果 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 当前结算方式 | ⚠️ TIME-BASED | 时间推算: `hold_minutes * 60000 + 30000ms` |
| 当前盈亏计算 | ⚠️ BALANCE-CHANGE | `pnl = current_balance - pre_balance` |
| 获取 settlement price | ❌ UNAVAILABLE | 完全依赖余额变化推断 |

**当前结算逻辑（`engine.py:162-206`）:**
```python
# 1. 时间推算到期
elapsed_ms = time.time() * 1000 - t["start_ts"]
settle_threshold_ms = config.HOLD_MINUTES * 60000 + 30000  # 15分钟+30秒缓冲

# 2. 余额变化推盈亏
pnl = current_balance - t["pre_balance"]
is_win = pnl > 0
```

**问题:**
- 如果同时间有多笔订单到期，余额变化无法区分各笔盈亏
- 如果主循环在结算期间刷新了余额，盈亏计算可能不准
- 无法获取 HIBT 官方的结算价格
- 无法验证 HIBT 结算是否与我们预期一致

### 2.6 Trade History（交易历史）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 获取历史订单列表 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 获取历史盈亏 | ❌ UNAVAILABLE | 无相关 API 调用 |
| 本地交易记录 | ⚠️ PARTIAL | `active_trades` 列表，仅内存，重启丢失 |

### 2.7 Balance（余额）— ✅ AVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| 实时查询余额 | ✅ AVAILABLE | `fetch_balance()` 每 30 秒刷新 |
| 余额精度 | ✅ AVAILABLE | float, USDT 精度 |
| 多端点容错 | ✅ AVAILABLE | 4 个端点轮询 |

### 2.8 Index Price（结算指数价格）— ❌ UNAVAILABLE

| 子能力 | 状态 | 说明 |
|--------|------|------|
| HIBT 结算指数价格 | ❌ UNAVAILABLE | 无 API 获取 |
| 当前价格来源 | ⚠️ GATE.IO | `hibt_ticks.csv` (Gate.io 1m K线) |

**关键问题:**
- HIBT Event Contract 的结算价格可能使用自己的指数价格，而非 Gate.io 价格
- 当前无法获取 HIBT 的结算指数价格
- 训练标签如果使用 Gate 价格代替 HIBT 结算价格，可能存在系统性偏差

---

## 三、认证体系审计

### 3.1 认证方式

| 认证字段 | 来源 | 优先级 | 已验证 |
|----------|------|--------|--------|
| `x-auth-token` | `HIBT_X_AUTH_TOKEN` env | 最高 | ✅ |
| `Authorization` | `HIBT_AUTHORIZATION` env | 次高 | ✅ |
| `HIBT_TOKEN` | `HIBT_TOKEN` env (fallback) | 兜底 | ✅ |
| `bget_key` | `HIBT_BGET_KEY` env | 低层 | ✅ |
| `bget_id` | `HIBT_BGET_ID` env | 低层 | ✅ |

### 3.2 动态签名

| 参数 | 生成方式 | 状态 |
|------|----------|------|
| `v` 参数 | `MD5(bget_id + timestamp + bget_key)[:16]` | ✅ 动态生成 |
| `Request-Id` | `MD5(bget_id + timestamp + random + bget_key)[:16]` | ✅ 动态生成 |
| `HIBT_V` | 静态兜底 | ⚠️ 备选 |

### 3.3 浏览器指纹

使用 `curl_cffi` 库模拟 Chrome 110 浏览器指纹 (`impersonate="chrome110"`)，绕过 Cloudflare 等反爬机制。

---

## 四、数据流审计

### 4.1 实时价格数据

```
Gate.io 1m K线 → hibt_ticks.csv → engine.py tick() → 品种循环
```

**数据字段:**
```
ts, symbol, open, high, low, close, volume
```

**问题:**
- 数据源是 Gate.io，不是 HIBT
- 可能被 Gate.io API 限流
- CSV 文件可能包含脏数据（header 行、空行）

### 4.2 交易决策数据

```
hibt_ticks.csv → 特征计算 → LightGBM 预测 → 6层过滤 → HIBT API 下单
```

**所有决策输入来自 Gate.io 数据，不是 HIBT 数据。**

### 4.3 结算数据

```
HIBT API 余额 → 时间推算 → 余额变化 → 推断盈亏
```

**结算价格来源: 无。完全依赖余额变化推断。**

---

## 五、能力总结矩阵

| 能力 | 状态 | 数据来源 | 可用于 LIVE |
|------|------|----------|-------------|
| **Order Placement** | ✅ AVAILABLE | HIBT API | ✅ 是 |
| **Balance Query** | ✅ AVAILABLE | HIBT API | ✅ 是 |
| **Payout Discovery** | ❌ UNAVAILABLE | 硬编码 config.py | ❌ 否 |
| **Expiry Discovery** | ❌ UNAVAILABLE | 硬编码 HOLD_MINUTES=15 | ⚠️ 仅15m |
| **Order Status** | ❌ UNAVAILABLE | 无 | ❌ 否 |
| **Settlement Result** | ❌ UNAVAILABLE | 时间推算+余额变化 | ❌ 否 |
| **Trade History** | ❌ UNAVAILABLE | 内存 active_trades | ❌ 否 |
| **Index Price** | ❌ UNAVAILABLE | Gate.io CSV | ❌ 否 |
| **Authentication** | ✅ AVAILABLE | JWT + bget 签名 | ✅ 是 |
| **Browser Fingerprint** | ✅ AVAILABLE | curl_cffi chrome110 | ✅ 是 |

---

## 六、EventEdge V2 架构影响

### 6.1 可以继续的功能

- ✅ 下单（Place Order）
- ✅ 余额查询
- ✅ 15 分钟期限交易
- ✅ 飞书通知
- ✅ SSE 实时事件流
- ✅ 前端控制面板

### 6.2 受限的功能

| 功能 | 限制 | 缓解措施 |
|------|------|----------|
| Dynamic Edge Engine | 无法获取实时 payout | 使用硬编码 payout，标注 `PAYOUT_STATIC` |
| Multi-Expiry | 仅 15m 已验证 | 启动时输出 `EXPIRY_DISCOVERY_UNAVAILABLE` |
| Settlement Ledger | 无真实结算数据 | 使用 `SETTLEMENT_PENDING` 状态 |
| Model Training Labels | 无 HIBT 结算价格 | 使用 Gate 价格作为代理，标注偏差风险 |
| Contract Discovery | 无 API 支持 | 使用配置 `AVAILABLE_EXPIRIES` |

### 6.3 禁止的功能

| 功能 | 原因 |
|------|------|
| LIVE Dynamic Payout | 无法实时获取赔付率 |
| 多期限 LIVE | 仅 15m 验证过 |
| 基于 HIBT 结算价格的回测 | 数据不可用 |
| 基于真实 settlement 的标签 | 数据不可用 |

---

## 七、建议的下一步行动

### 7.1 立即执行（Phase 1.5 完成）

- [x] 完整审计所有 HIBT 网络请求
- [x] 输出能力矩阵
- [x] 识别数据缺口

### 7.2 推荐探索（Phase 2 前）

1. **抓取 HIBT 下单响应完整 JSON** — 确认 `order_id`, `open_price`, `payout` 等字段是否存在于响应中
2. **探索 HIBT 前端页面** — 从浏览器 Network 面板发现更多 API（订单历史、结算结果等）
3. **验证 HIBT payout 是否动态变化** — 在不同时间点观察同一品种的 payout 是否一致
4. **验证 5m/30m/60m 是否可用** — 用最小金额（3U）测试不同 timeUnit 参数

### 7.3 暂不执行

- 不猜测 HIBT API endpoint
- 不虚构不存在的 API 响应字段
- 不在 LIVE 模式启用依赖未验证数据的功能

---

## 八、方向映射验证

**当前代码** (`exchange.py:130-133`):
```python
# HIBT API direction（实测）:
#   direction=1 (CALL/做多) → API传1 (HIBT 多)
#   direction=2 (PUT/做空) → API传-1 (HIBT 空)
hibt_dir = 1 if direction == 1 else -1
```

**Memory 中的记录已过时**（31 天前，已被 commit `ae48088` 修复）。当前映射已纠正。

**方向映射表:**
| 内部 direction | 含义 | HIBT API direction |
|---------------|------|-------------------|
| 1 | CALL (做多) | 1 |
| 2 | PUT (做空) | -1 |

---

## 九、结论

**HIBT 当前实际可用的能力仅限于:**
1. 查询余额
2. 下单（CALL/PUT，指定金额和期限）

**所有其他能力（payout 获取、期限发现、订单状态、结算结果、历史记录、指数价格）均不可用。**

**对 EventEdge V2 的影响:**
- Phase 6 (Edge Engine) 的 Dynamic Payout 功能在 LIVE 模式下不可用
- Phase 3 (真实标签) 无法使用 HIBT 官方结算价格，只能使用 Gate 价格作为代理
- Phase 2 (Trade Ledger) 的 Settlement 必须使用 `SETTLEMENT_PENDING` 状态
- 多期限支持仅限于架构设计，LIVE 只启用 15m

**系统整体可以继续开发 BACKTEST 和 SHADOW 模式，但 LIVE 自动交易的功能范围受限于当前可用的 HIBT API 能力。**