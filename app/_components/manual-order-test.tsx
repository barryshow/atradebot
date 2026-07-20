"use client";
import { useState } from "react";

interface ManualOrderResult {
  ok: boolean;
  code?: number;
  msg?: string;
  symbol?: string;
  direction?: number;
  amount?: number;
  order_id?: string;
  open_price?: number;
  lifecycle?: string;
  raw_response?: Record<string, unknown>;
}

export function ManualOrderTest() {
  const [show, setShow] = useState(false);
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [direction, setDirection] = useState(1); // 1=CALL, 2=PUT
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ManualOrderResult | null>(null);
  const [error, setError] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [orderCount, setOrderCount] = useState(0);

  const handleSubmit = async () => {
    if (!confirmed) {
      setError("请先勾选确认");
      return;
    }
    if (orderCount >= 1) {
      setError("已完成一笔测试订单，禁止重复提交");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch("/api/engine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "manual_order_test",
          symbol,
          direction,
          amount: 3,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        setOrderCount(1);
        // 轮询等待结果（SSE 事件）
        setTimeout(() => {
          // 从最近的 trade_executed 事件获取结果
          fetch("/api/engine")
            .then((r) => r.json())
            .then((state) => {
              setResult({
                ok: true,
                msg: "订单已提交，等待 HIBT 响应...",
                symbol,
                direction,
                amount: 3,
                lifecycle: "ORDER_REQUESTED",
              });
            });
        }, 2000);
      } else {
        setError(data.error || "请求失败");
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError("");
    setConfirmed(false);
    // orderCount stays at 1 after first order
  };

  if (!show) {
    return (
      <button
        onClick={() => setShow(true)}
        className="text-xs text-orange-500 hover:text-orange-300 underline"
      >
        🔧 手动下单测试
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-gray-900 border border-orange-700 rounded-lg p-6 w-full max-w-md">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-bold text-orange-400">
            🔧 MANUAL_ORDER_TEST
          </h2>
          <button onClick={() => setShow(false)} className="text-gray-400 hover:text-white text-xl">
            &times;
          </button>
        </div>

        <div className="bg-orange-900/30 border border-orange-800 rounded p-3 mb-4">
          <p className="text-xs text-orange-300">
            ⚠️ 这是接口测试，不是策略交易。<br />
            本订单不计入 Win Rate / Model Health / PnL 统计。<br />
            最多允许 1 笔，固定 3U。
          </p>
        </div>

        {orderCount >= 1 ? (
          <div className="space-y-3">
            <div className="bg-green-900/30 border border-green-800 rounded p-3">
              <p className="text-sm text-green-400 font-medium">✅ 测试订单已提交</p>
              <p className="text-xs text-green-300 mt-1">已禁止第二笔订单</p>
            </div>
            {result && (
              <div className="bg-gray-800 rounded p-3 space-y-1 font-mono text-xs">
                <div className="text-gray-400">下单结果:</div>
                <div className="text-gray-200">Symbol: {result.symbol}</div>
                <div className="text-gray-200">Direction: {result.direction === 1 ? "CALL" : "PUT"}</div>
                <div className="text-gray-200">Amount: 3 USDT</div>
                <div className="text-gray-200">Status: {result.lifecycle || "PENDING"}</div>
                {result.order_id && (
                  <div className="text-green-400">Order ID: {result.order_id}</div>
                )}
                {result.open_price && (
                  <div className="text-gray-200">Open Price: {result.open_price}</div>
                )}
                {result.code && (
                  <div className="text-gray-200">Code: {result.code}</div>
                )}
                {result.msg && (
                  <div className="text-gray-200">Msg: {result.msg}</div>
                )}
              </div>
            )}
            <div className="text-xs text-gray-500">
              等待引擎 SSE 推送 trade_executed 和 trade_result 事件。
              <br />
              15 分钟后到期，前端将显示结算结果 (SETTLED_UNVERIFIED)。
            </div>
            <button
              onClick={() => setShow(false)}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm"
            >
              关闭（保持后台运行）
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1">品种</label>
              <select
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-white"
              >
                <option value="BTCUSDT">BTCUSDT</option>
                <option value="ETHUSDT">ETHUSDT</option>
                <option value="SOLUSDT">SOLUSDT</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-gray-400 block mb-1">方向</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setDirection(1)}
                  className={`flex-1 px-4 py-2 rounded text-sm font-medium ${
                    direction === 1
                      ? "bg-green-600 text-white"
                      : "bg-gray-800 text-gray-400"
                  }`}
                >
                  CALL (做多)
                </button>
                <button
                  onClick={() => setDirection(2)}
                  className={`flex-1 px-4 py-2 rounded text-sm font-medium ${
                    direction === 2
                      ? "bg-red-600 text-white"
                      : "bg-gray-800 text-gray-400"
                  }`}
                >
                  PUT (做空)
                </button>
              </div>
            </div>

            <div className="bg-gray-800 rounded p-3">
              <div className="text-xs text-gray-400">下注金额</div>
              <div className="text-lg font-bold text-white font-mono">3 USDT</div>
              <div className="text-xs text-gray-500">固定最低金额，不可修改</div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-xs text-gray-300">
                我确认这是接口测试，不是策略交易。不计入 PnL 统计。
              </span>
            </label>

            {error && (
              <div className="bg-red-900/30 border border-red-800 rounded p-2 text-xs text-red-400">
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={loading || !confirmed}
              className="w-full px-4 py-3 bg-orange-600 hover:bg-orange-500 disabled:bg-gray-700 text-white rounded font-bold"
            >
              {loading ? "提交中..." : "确认下单 3U"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}