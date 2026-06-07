"use client";
import type { SymbolSnapshot } from "@/lib/types/engine";

const symbolIcons: Record<string, string> = {
  BTCUSDT: "BTC",
  ETHUSDT: "ETH",
  SOLUSDT: "SOL",
};

function Gauge({ value, low, high, label }: { value: number; low: number; high: number; label: string }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  const inZone = value >= low && value <= high;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={inZone ? "text-red-400" : "text-green-400"}>{value.toFixed(3)}</span>
      </div>
      <div className="h-1.5 bg-gray-700 rounded-full relative">
        <div
          className={`h-full rounded-full ${inZone ? "bg-red-500" : "bg-green-500"}`}
          style={{ width: `${pct}%` }}
        />
        {/* Dead zone markers */}
        <div
          className="absolute top-0 h-full bg-red-900/30"
          style={{ left: `${low * 100}%`, width: `${(high - low) * 100}%` }}
        />
      </div>
    </div>
  );
}

function Indicator({ label, value, unit = "" }: { label: string; value: number; unit?: string }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-gray-400">{label}</span>
      <span className="font-mono text-gray-200">
        {value.toFixed(2)}
        {unit}
      </span>
    </div>
  );
}

export function SymbolCard({ snapshot }: { snapshot: Partial<SymbolSnapshot> }) {
  const sym = snapshot.symbol || "???";
  const short = symbolIcons[sym] || sym.slice(0, 3);
  const dir = snapshot.direction;
  const prob = snapshot.mlProb;
  const gates = snapshot.riskGates || [];

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold text-white">{short}</span>
          {dir && (
            <span
              className={`text-xs font-medium px-2 py-0.5 rounded ${
                dir === "CALL" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
              }`}
            >
              {dir}
            </span>
          )}
        </div>
        {typeof prob === "number" && (
          <div className="text-right">
            <div className="text-xs text-gray-400">ML胜率</div>
            <div className={`text-lg font-bold ${prob >= 0.556 ? "text-green-400" : "text-gray-500"}`}>
              {(prob * 100).toFixed(1)}%
            </div>
          </div>
        )}
      </div>

      {typeof snapshot.price === "number" && (
        <div className="text-sm font-mono text-gray-300">
          ${snapshot.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
        </div>
      )}

      <div className="space-y-2">
        {typeof snapshot.bbPos === "number" && (
          <Gauge value={snapshot.bbPos} low={0.4} high={0.6} label="BB Position" />
        )}

        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {typeof snapshot.adx === "number" && <Indicator label="ADX" value={snapshot.adx} />}
          {typeof snapshot.rsi === "number" && <Indicator label="RSI" value={snapshot.rsi} />}
          {typeof snapshot.macd === "number" && <Indicator label="MACD" value={snapshot.macd} />}
          {typeof snapshot.bsp5 === "number" && (
            <Indicator label="BSP_5" value={snapshot.bsp5} />
          )}
        </div>
      </div>

      {/* Risk gate indicators */}
      {gates.length > 0 && (
        <div className="flex gap-1.5 flex-wrap">
          {gates.map((g, i) => (
            <span
              key={i}
              className={`text-[10px] px-1.5 py-0.5 rounded ${
                g.passed ? "bg-green-900/50 text-green-400" : "bg-red-900/50 text-red-400"
              }`}
              title={g.reason}
            >
              L{g.level} {g.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
