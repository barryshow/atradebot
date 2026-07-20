"use client";
import type { MarketRegimeType, RejectReasonCode, EdgeResult } from "@/lib/types/engine";

const regimeLabels: Record<MarketRegimeType, string> = {
  TREND_UP: "上涨趋势",
  TREND_DOWN: "下跌趋势",
  RANGE: "震荡",
  HIGH_VOLATILITY: "高波动",
  LOW_LIQUIDITY: "低流动性",
  EVENT_RISK: "事件风险",
};

const regimeColors: Record<MarketRegimeType, string> = {
  TREND_UP: "text-green-400 bg-green-900/40",
  TREND_DOWN: "text-red-400 bg-red-900/40",
  RANGE: "text-yellow-400 bg-yellow-900/40",
  HIGH_VOLATILITY: "text-orange-400 bg-orange-900/40",
  LOW_LIQUIDITY: "text-gray-400 bg-gray-700/40",
  EVENT_RISK: "text-red-500 bg-red-900/60",
};

const rejectLabels: Record<RejectReasonCode, string> = {
  NO_EDGE: "无优势",
  LOW_EDGE: "优势不足",
  HIGH_UNCERTAINTY: "不确定性高",
  MODEL_DEGRADED: "模型退化",
  CORRELATION_LIMIT: "相关性限制",
  DAILY_STOP: "日止损",
  WEEKLY_DRAWDOWN: "周回撤",
  LOW_LIQUIDITY: "低流动性",
  EVENT_RISK: "事件风险",
  ACCOUNT_TOO_SMALL_FOR_RISK_RULE: "账户太小",
  PORTFOLIO_LIMIT: "组合限制",
  CONSECUTIVE_LOSS: "连续亏损",
  SIGNAL_VALIDATION: "信号验证",
  ORDER_FAILED: "下单失败",
  CONFIG_DISABLED: "配置禁用",
};

export function RegimeIndicator({ regime, confidence }: { regime?: MarketRegimeType; confidence?: number }) {
  if (!regime) return null;
  return (
    <div className="flex items-center gap-2">
      <span className={`text-xs font-medium px-2 py-0.5 rounded ${regimeColors[regime] || "bg-gray-800 text-gray-400"}`}>
        {regimeLabels[regime] || regime}
      </span>
      {typeof confidence === "number" && (
        <span className="text-xs text-gray-500">{(confidence * 100).toFixed(0)}%</span>
      )}
    </div>
  );
}

export function EdgePanel({ edge }: { edge?: Partial<EdgeResult> }) {
  if (!edge) return null;
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">校准概率</span>
        <span className="font-mono text-gray-200">{(edge.calibratedProbability ?? 0) * 100}%</span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">保守概率</span>
        <span className="font-mono text-gray-200">{(edge.conservativeProbability ?? 0) * 100}%</span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">盈亏平衡</span>
        <span className="font-mono text-gray-200">{(edge.breakEvenProbability ?? 0) * 100}%</span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">有效优势</span>
        <span className={`font-mono font-bold ${(edge.effectiveEdge ?? 0) >= 0.02 ? "text-green-400" : "text-red-400"}`}>
          {((edge.effectiveEdge ?? 0) * 100).toFixed(2)}%
        </span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">期望ROI</span>
        <span className={`font-mono ${(edge.expectedRoi ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
          {((edge.expectedRoi ?? 0) * 100).toFixed(2)}%
        </span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">赔付率</span>
        <span className="font-mono text-gray-300">{edge.payoutSource === "hardcoded" ? "⚙️" : "📡"} {(edge.netPayoutRatio ?? 0) * 100}%</span>
      </div>
      {edge.passed === false && edge.rejectReason && (
        <div className="text-xs text-red-400 mt-1">
          ❌ {rejectLabels[edge.rejectReason as RejectReasonCode] || edge.rejectReason}
        </div>
      )}
    </div>
  );
}

export function ExpertVotes({ votes, regime }: { votes?: Record<string, number>; regime?: MarketRegimeType }) {
  if (!votes || Object.keys(votes).length === 0) return null;
  const expertNames: Record<string, string> = {
    trend: "趋势专家",
    mean_reversion: "均值回归",
    volatility_breakout: "波动突破",
  };
  const sorted = Object.entries(votes).sort(([, a], [, b]) => b - a);
  return (
    <div className="space-y-1.5">
      {sorted.map(([name, prob]) => (
        <div key={name} className="flex items-center justify-between text-xs">
          <span className="text-gray-400">{expertNames[name] || name}</span>
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-gray-700 rounded-full">
              <div
                className="h-full rounded-full bg-blue-500"
                style={{ width: `${Math.min(prob * 100, 100)}%` }}
              />
            </div>
            <span className="font-mono text-gray-200 w-10 text-right">{(prob * 100).toFixed(0)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export function ModelHealthPanel({ health }: { health?: { isDegraded: boolean; actualWinRate: number; predictedWinRate: number; winRateDelta: number; brierScore: number; expectedCalibrationError: number; window: number } }) {
  if (!health) return null;
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-400">模型健康 ({health.window}笔)</span>
        <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${health.isDegraded ? "bg-red-900/60 text-red-400" : "bg-green-900/60 text-green-400"}`}>
          {health.isDegraded ? "⚠️ 退化" : "✅ 正常"}
        </span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">实际胜率</span>
        <span className="font-mono text-gray-200">{(health.actualWinRate * 100).toFixed(1)}%</span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">预测胜率</span>
        <span className="font-mono text-gray-200">{(health.predictedWinRate * 100).toFixed(1)}%</span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">偏差</span>
        <span className={`font-mono ${Math.abs(health.winRateDelta) > 0.08 ? "text-red-400" : "text-gray-200"}`}>
          {health.winRateDelta >= 0 ? "+" : ""}{(health.winRateDelta * 100).toFixed(1)}%
        </span>
      </div>
      <div className="flex justify-between text-xs">
        <span className="text-gray-500">Brier</span>
        <span className="font-mono text-gray-200">{health.brierScore.toFixed(3)}</span>
      </div>
    </div>
  );
}