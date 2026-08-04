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
import { ManualOrderTest } from "@/app/_components/manual-order-test";
import { RegimeIndicator, EdgePanel, ExpertVotes, ModelHealthPanel } from "@/app/_components/eventedge-panels";
import { SYMBOLS } from "@/lib/types/engine";
import type { TradeRecordFlat } from "@/lib/types/candles";
import type { MarketRegimeType } from "@/lib/types/engine";
import type { RunModeParam } from "@/lib/hooks/use-engine-control";
import { useState } from "react";

export default function DashboardPage() {
  const {
    connected, status, symbols, recentTrades, logs,
    regime, regimeConfidence, expertVotes: sseExpertVotes,
    edge, modelHealth, calibrationStatus, shadowTrades,
    fastScans, settlementAvailable, payoutAvailable,
  } = useEngineSSE();
  const [activeMode, setActiveMode] = useState<RunModeParam>("shadow");

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

  const activeSymbol = "BTCUSDT";
  const currentRegime = regime[activeSymbol] as MarketRegimeType | undefined;
  const currentConfidence = regimeConfidence[activeSymbol];
  const currentEdge = edge[activeSymbol];
  const currentVotes = sseExpertVotes[activeSymbol] || {};

  // Latest fast scan for display
  const latestScan: Record<string, unknown> | null = fastScans.length > 0 ? (fastScans[fastScans.length - 1] as Record<string, unknown>) : null;

  return (
    <div className="space-y-4 max-w-7xl mx-auto">
      {/* Header with engine control */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-4">
          <EngineControl
            state={status.state}
            runMode={status.runMode}
            liveGate={status.liveGate}
            onModeChange={setActiveMode}
          />
          <RegimeIndicator regime={currentRegime} confidence={currentConfidence} />
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              connected ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-xs text-gray-400">
            {connected ? "已连接" : "断开连接"}
          </span>

          {/* v6 health indicators */}
          <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
            calibrationStatus === "READY" ? "bg-green-900/60 text-green-400" : "bg-yellow-900/60 text-yellow-400"
          }`}>
            {calibrationStatus === "READY" ? "CALIBRATED" : "UNCALIBRATED"}
          </span>

          {settlementAvailable !== undefined && (
            <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
              settlementAvailable ? "bg-green-900/60 text-green-400" : "bg-red-900/60 text-red-400"
            }`}>
              {settlementAvailable ? "HIBT结算✓" : "HIBT结算✗"}
            </span>
          )}

          {payoutAvailable !== undefined && (
            <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
              payoutAvailable ? "bg-green-900/60 text-green-400" : "bg-red-900/60 text-red-400"
            }`}>
              {payoutAvailable ? "Payout API✓" : "Payout NO API"}
            </span>
          )}

          <ManualOrderTest />
          <HibtConfigPanel />
        </div>
      </div>

      {/* Stats bar */}
      <StatsBar status={status} />

      {/* K-line chart */}
      <CandleChart
        activeSymbol={activeSymbol}
        trades={tradeMarkers}
      />

      {/* EventEdge V2: Edge + Expert + Health + PnL row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Edge Analysis */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Edge 分析</h3>
          <EdgePanel edge={currentEdge} />
        </div>

        {/* Fast Scan EV */}
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">最近扫描 (EV)</h3>
          {latestScan ? (
            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">{String(latestScan.symbol ?? "")}</span>
                <span className={latestScan.direction === "CALL" ? "text-green-400" : latestScan.direction === "PUT" ? "text-red-400" : "text-gray-500"}>
                  {String(latestScan.direction ?? "--")}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Fast Prob</span>
                <span className="font-mono">{((latestScan.fastProb as number || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Calibrated</span>
                <span className="font-mono">{((latestScan.calibratedProb as number || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">EV CALL</span>
                <span className={`font-mono ${(latestScan.evCall as number || 0) > 0.03 ? "text-green-400" : "text-gray-500"}`}>
                  {(latestScan.evCall as number || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">EV PUT</span>
                <span className={`font-mono ${(latestScan.evPut as number || 0) > 0.03 ? "text-green-400" : "text-gray-500"}`}>
                  {(latestScan.evPut as number || 0).toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between border-t border-gray-800 pt-1 mt-1">
                <span className="text-gray-500">Status</span>
                <span className={latestScan.status === "PASSED" ? "text-green-400" : "text-yellow-400"}>
                  {latestScan.status as string}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-xs text-gray-500">等待扫描...</div>
          )}
        </div>

        {/* Model Health */}
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

      {/* Symbol cards */}
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

      {/* Trade table + shadow trades */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">交易记录</h3>
          <TradeTable trades={recentTrades} />
        </div>
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Shadow 订单</h3>
          {shadowTrades.length === 0 ? (
            <div className="text-center text-gray-500 py-8 text-sm">暂无 Shadow 订单</div>
          ) : (
            <div className="space-y-1.5 max-h-80 overflow-y-auto">
              {shadowTrades.slice().reverse().map((raw, i) => {
                const t = raw as Record<string, unknown>;
                const symbol = String(t.symbol ?? "");
                const direction = String(t.direction ?? "");
                const ev = Number(t.ev ?? 0);
                const amount = String(t.amount ?? "0");
                const settled = Boolean(t.settled);
                const pnl = Number(t.pnl ?? 0);
                return (
                <div key={i} className={`text-xs p-2 rounded ${settled ? "bg-gray-800/50" : "bg-gray-800"}`}>
                  <div className="flex justify-between">
                    <span className="font-medium">{symbol}</span>
                    <span className={direction === "CALL" ? "text-green-400" : "text-red-400"}>{direction}</span>
                  </div>
                  <div className="flex justify-between text-gray-500 mt-0.5">
                    <span>EV: {ev.toFixed(4)}</span>
                    <span>{amount}U</span>
                    {settled && (
                      <span className={pnl >= 0 ? "text-green-400" : "text-red-400"}>
                        {pnl >= 0 ? "+" : ""}{pnl.toFixed(2)}U
                      </span>
                    )}
                  </div>
                </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Log stream */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">实时日志</h3>
        <LogStream logs={logs} />
      </div>
    </div>
  );
}
