"use client";
import type { TradeRecord } from "@/lib/types/engine";

function formatTime(ts: number) {
  return new Date(ts).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function TradeTable({ trades }: { trades: TradeRecord[] }) {
  if (trades.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8 text-sm">
        暂无交易记录
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-800 text-gray-400 text-left">
            <th className="py-2 px-2 font-medium">时间</th>
            <th className="py-2 px-2 font-medium">品种</th>
            <th className="py-2 px-2 font-medium">方向</th>
            <th className="py-2 px-2 font-medium">入场价</th>
            <th className="py-2 px-2 font-medium">投入</th>
            <th className="py-2 px-2 font-medium">ML胜率</th>
            <th className="py-2 px-2 font-medium">AI理由</th>
            <th className="py-2 px-2 font-medium">结果</th>
            <th className="py-2 px-2 font-medium">盈亏</th>
          </tr>
        </thead>
        <tbody>
          {trades
            .slice()
            .reverse()
            .map((t, i) => (
              <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                <td className="py-2 px-2 font-mono text-xs text-gray-400">{formatTime(t.ts)}</td>
                <td className="py-2 px-2 font-medium">{t.symbol.replace("USDT", "")}</td>
                <td className="py-2 px-2">
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      t.direction === "CALL" ? "bg-green-900/50 text-green-400" : "bg-red-900/50 text-red-400"
                    }`}
                  >
                    {t.direction === "CALL" ? "做多" : "做空"}
                    {t.flipped ? " 🔄" : ""}
                  </span>
                </td>
                <td className="py-2 px-2 font-mono">${t.entryPrice.toFixed(2)}</td>
                <td className="py-2 px-2">{t.amount}U</td>
                <td className="py-2 px-2 font-mono">{(t.mlProb * 100).toFixed(1)}%</td>
                <td className="py-2 px-2 text-xs text-gray-400 max-w-[120px] truncate">{t.aiReason}</td>
                <td className="py-2 px-2">
                  {t.result === "win" && <span className="text-green-400">✅</span>}
                  {t.result === "loss" && <span className="text-red-400">❌</span>}
                  {t.result === "pending" && <span className="text-yellow-400 animate-pulse">⏳</span>}
                </td>
                <td className="py-2 px-2 font-mono">
                  {typeof t.pnl === "number" ? (
                    <span className={t.pnl >= 0 ? "text-green-400" : "text-red-400"}>
                      {t.pnl >= 0 ? "+" : ""}
                      {t.pnl.toFixed(2)}U
                    </span>
                  ) : (
                    <span className="text-gray-600">--</span>
                  )}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}
