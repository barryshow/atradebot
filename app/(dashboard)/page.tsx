"use client";
import { useEngineSSE } from "@/lib/hooks/use-engine-sse";
import { EngineControl } from "@/app/_components/engine-control";
import { StatsBar } from "@/app/_components/stats-bar";
import { SymbolCard } from "@/app/_components/symbol-card";
import { CandleChart } from "@/app/_components/candle-chart";
import { TradeTable } from "@/app/_components/trade-table";
import { LogStream } from "@/app/_components/log-stream";
import { PnlChart } from "@/app/_components/pnl-chart";
import { RiskGateVisualizer } from "@/app/_components/risk-gate-visualizer";
import { HibtConfigPanel } from "@/app/_components/hibt-config";
import { RegimeIndicator, EdgePanel, ExpertVotes, ModelHealthPanel } from "@/app/_components/eventedge-panels";
import { SYMBOLS } from "@/lib/types/engine";
import type { TradeRecordFlat } from "@/lib/types/candles";
import type { EdgeResult, MarketRegimeType } from "@/lib/types/engine";

export default function DashboardPage() {
  const { connected, status, symbols, recentTrades, logs, regime, regimeConfidence, expertVotes: sseExpertVotes, edge, modelHealth, calibrationStatus, shadowTrades } = useEngineSSE();

  // Flatten trades for chart markers
  const tradeMarkers: TradeRecordFlat[] = recentTrades.map((t) => ({
    ts: t.ts,
    symbol: t.symbol,
    direction: t.direction,
    entryPrice: t.entryPrice,
    amount: t.amount,
    result: t.result,
    pnl: t.pnl,
  }));

  // Use real SSE data — no Math.random()
  const activeSymbol = "BTCUSDT";
  const currentRegime = regime[activeSymbol] as MarketRegimeType;
  const currentConfidence = regimeConfidence[activeSymbol];
  const currentEdge = edge[activeSymbol];
  const currentVotes = sseExpertVotes[activeSymbol] || {};

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Header with engine control */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <EngineControl state={status.state} />
          <RegimeIndicator regime={currentRegime} confidence={currentConfidence} />
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400">
            {connected ? "已连接" : "断开连接"}
          </span>
          {/* Calibration status badge */}
          <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
            calibrationStatus === "READY" ? "bg-green-900/60 text-green-400" : "bg-yellow-900/60 text-yellow-400"
          }`}>
            {calibrationStatus === "READY" ? "CALIBRATED" : "UNCALIBRATED"}
          </span>
          <HibtConfigPanel />
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar status={status} />

      {/* K-line chart — takes center stage */}
      <CandleChart
        activeSymbol={activeSymbol}
        trades={tradeMarkers}
      />

      {/* EventEdge V2: Edge + Expert + Health row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Edge Analysis — real SSE data */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Edge 分析</h3>
          <EdgePanel edge={currentEdge} />
        </div>

        {/* Expert Votes — real SSE data */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">专家投票</h3>
          <ExpertVotes votes={currentVotes} regime={currentRegime} />
        </div>

        {/* Model Health — real SSE data */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">模型健康</h3>
          <ModelHealthPanel health={{
            isDegraded: (modelHealth.isDegraded as boolean) || false,
            actualWinRate: (modelHealth.actualWinRate as number) || 0,
            predictedWinRate: (modelHealth.predictedWinRate as number) || 0,
            winRateDelta: (modelHealth.winRateDelta as number) || 0,
            brierScore: (modelHealth.brierScore as number) || 0,
            expectedCalibrationError: (modelHealth.ece as number) || 0,
            window: (modelHealth.window as number) || 0,
          }} />
        </div>

        {/* PnL Chart */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <PnlChart trades={recentTrades} />
        </div>
      </div>

      {/* Symbol cards — compact row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {SYMBOLS.map((sym) => (
          <SymbolCard key={sym} snapshot={symbols[sym] || { symbol: sym }} />
        ))}
      </div>

      {/* Risk gate visualizer */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-2">风控流程</h3>
        <RiskGateVisualizer gates={recentTrades.length > 0 ? recentTrades[recentTrades.length - 1].riskGates : []} />
      </div>

      {/* Trade table */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">交易记录</h3>
        <TradeTable trades={recentTrades} />
      </div>

      {/* Log stream */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">实时日志</h3>
        <LogStream logs={logs} />
      </div>
    </div>
  );
}
