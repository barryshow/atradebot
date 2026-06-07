"use client";
import type { TradeRecord } from "@/lib/types/engine";

export function PnlChart({ trades }: { trades: TradeRecord[] }) {
  const completed = trades.filter((t) => t.result === "win" || t.result === "loss");
  if (completed.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-500 text-sm">
        等待交易数据...
      </div>
    );
  }

  // Build cumulative PnL
  let cum = 0;
  const points = completed.map((t, i) => {
    cum += t.pnl || 0;
    return { x: i, y: cum, trade: t };
  });

  const minY = Math.min(0, ...points.map((p) => p.y));
  const maxY = Math.max(0, ...points.map((p) => p.y));
  const rangeY = maxY - minY || 1;
  const h = 120;
  const w = 400;
  const padY = 10;

  const toSvg = (p: { x: number; y: number }) => ({
    x: (p.x / Math.max(points.length - 1, 1)) * w,
    y: h - padY - ((p.y - minY) / rangeY) * (h - 2 * padY),
  });

  const svgPoints = points.map(toSvg);
  const pathD = svgPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const zeroY = toSvg({ x: 0, y: 0 }).y;

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-gray-300">累计盈亏</span>
        <span className={`text-sm font-bold ${cum >= 0 ? "text-green-400" : "text-red-400"}`}>
          {cum >= 0 ? "+" : ""}
          {cum.toFixed(2)}U
        </span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-32" preserveAspectRatio="none">
        {/* Zero line */}
        <line x1="0" y1={zeroY} x2={w} y2={zeroY} stroke="#374151" strokeWidth="1" strokeDasharray="4" />
        {/* PnL line */}
        <path d={pathD} fill="none" stroke={cum >= 0 ? "#22c55e" : "#ef4444"} strokeWidth="2" />
        {/* Area fill */}
        <path
          d={`${pathD} L ${w} ${h} L 0 ${h} Z`}
          fill={cum >= 0 ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)"}
        />
        {/* End dot */}
        {svgPoints.length > 0 && (
          <circle
            cx={svgPoints[svgPoints.length - 1].x}
            cy={svgPoints[svgPoints.length - 1].y}
            r="3"
            fill={cum >= 0 ? "#22c55e" : "#ef4444"}
          />
        )}
      </svg>
    </div>
  );
}
