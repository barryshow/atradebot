export type EngineState = "stopped" | "starting" | "running" | "paused" | "error";

export type RunMode = "BACKTEST" | "SHADOW" | "LIVE";

export interface EngineStatus {
  state: EngineState;
  pid: number | null;
  uptime: number;
  tradeCountToday: number;
  wins: number;
  losses: number;
  activeTrades: number;
  maxConcurrentTrades: number;
  balance: number;
  lastTick: string | null;
  betMode?: string;
  profit?: number;
  runMode?: RunMode;
  calibrationReady?: boolean;
  healthTradeCount?: number;
  liveGate?: {
    passed: boolean;
    reasons: string[];
    checks: Record<string, boolean>;
  };
  symbol_modes?: Record<string, string>;
}

export type EngineEventType =
  | "status"
  | "tick"
  | "features"
  | "prediction"
  | "risk_gate"
  | "trade_executed"
  | "trade_result"
  | "signal_flip"
  | "log"
  | "error"
  | "balance_update"
  | "candle_update"
  | "pong"
  // EventEdge V2
  | "regime_update"
  | "expert_votes"
  | "edge_calculation"
  | "opportunity_ranked"
  | "model_health"
  | "trade_rejected"
  | "shadow_trade"
  | "calibration_status"
  | "decision_cycle"
  | "funnel"
  | "emergency_stop";

export interface EngineEvent {
  type: EngineEventType;
  ts: number;
  payload: Record<string, unknown>;
}

export interface RiskGateResult {
  level: number;
  name: string;
  passed: boolean;
  reason: string;
  details?: Record<string, unknown>;
}

export interface TradeRecord {
  ts: number;
  symbol: string;
  direction: "CALL" | "PUT";
  amount: number;
  entryPrice: number;
  exitPrice?: number;
  mlProb: number;
  aiApproval: number;
  aiReason: string;
  riskGates: RiskGateResult[];
  result?: "win" | "loss" | "pending";
  pnl?: number;
  flipped: boolean;
}

export interface SymbolSnapshot {
  symbol: string;
  price: number;
  mlProb: number;
  direction: "CALL" | "PUT" | null;
  adx: number;
  bbPos: number;
  rsi: number;
  macd: number;
  bsp5: number;
  regime?: MarketRegimeType;
  riskGates: RiskGateResult[];
  lastTrade: TradeRecord | null;
}

export interface PayoutRates {
  [symbol: string]: number;
}

export const PAYOUT_RATES: PayoutRates = {
  BTCUSDT: 0.818,
  ETHUSDT: 0.80,
  SOLUSDT: 0.80,
};

export const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] as const;
export type Symbol = (typeof SYMBOLS)[number];

export interface DashboardData {
  status: EngineStatus;
  symbols: Record<string, SymbolSnapshot>;
  recentTrades: TradeRecord[];
  recentLogs: string[];
}

// ═══════════════════════════════════════════════════════════
// EventEdge V2 Types
// ═══════════════════════════════════════════════════════════

export type MarketRegimeType =
  | "TREND_UP"
  | "TREND_DOWN"
  | "RANGE"
  | "HIGH_VOLATILITY"
  | "LOW_LIQUIDITY"
  | "EVENT_RISK";

export type RejectReasonCode =
  | "NO_EDGE"
  | "LOW_EDGE"
  | "HIGH_UNCERTAINTY"
  | "MODEL_DEGRADED"
  | "CORRELATION_LIMIT"
  | "DAILY_STOP"
  | "WEEKLY_DRAWDOWN"
  | "LOW_LIQUIDITY"
  | "EVENT_RISK"
  | "ACCOUNT_TOO_SMALL_FOR_RISK_RULE"
  | "PORTFOLIO_LIMIT"
  | "CONSECUTIVE_LOSS"
  | "SIGNAL_VALIDATION"
  | "ORDER_FAILED"
  | "CONFIG_DISABLED";

export type SettlementStatus =
  | "PENDING"
  | "CONFIRMED"
  | "ESTIMATED"
  | "DISPUTED"
  | "REJECTED"
  | "EXPIRED";

export interface MarketRegime {
  regime: MarketRegimeType;
  confidence: number;
  atr: number;
  adx: number;
  volatility: number;
  emaSlope: number;
  bbWidth: number;
  volumeZscore: number;
  details: Record<string, unknown>;
}

export interface ExpertVote {
  expertName: string;
  direction: "CALL" | "PUT";
  rawProbability: number;
  calibratedProbability: number;
  confidence: number;
}

export interface EdgeResult {
  symbol: string;
  expiryMinutes: number;
  direction: "CALL" | "PUT";
  entryPrice: number;
  calibratedProbability: number;
  conservativeProbability: number;
  payoutRatio: number;
  netPayoutRatio: number;
  payoutSource: "api" | "hardcoded" | "estimated";
  breakEvenProbability: number;
  probabilityEdge: number;
  effectiveEdge: number;
  expectedRoi: number;
  uncertaintyMargin: number;
  calibrationMargin: number;
  modelDegradationMargin: number;
  passed: boolean;
  rejectReason: RejectReasonCode | "";
  regime: MarketRegimeType;
  expertVotes: Record<string, number>;
}

export interface TradeRecordV2 {
  tradeId: string;
  hibtOrderId?: string;
  clientOrderId: string;
  symbol: string;
  direction: "CALL" | "PUT";
  directionInt: number;
  entryTimeMs: number;
  expiryTimeMs: number;
  expiryMinutes: number;
  entryPrice: number;
  expiryPrice?: number;
  stakeUsd: number;
  betFraction: number;
  payoutRatio: number;
  netPayoutRatio: number;
  payoutSource: string;
  rawProbability: number;
  calibratedProbability: number;
  conservativeProbability: number;
  breakEvenProbability: number;
  probabilityEdge: number;
  expectedRoi: number;
  effectiveEdge: number;
  uncertaintyMargin: number;
  calibrationMargin: number;
  modelDegradationMargin: number;
  modelVersion: string;
  regime: MarketRegimeType;
  expertVotes: Record<string, number>;
  result: "WIN" | "LOSS" | "TIE" | "PENDING" | "REJECTED";
  realizedPnl?: number;
  settlementStatus: SettlementStatus;
  rejectReason: RejectReasonCode | "";
  rejectDetail: string;
  createdAt: string;
  settledAt?: string;
}

export interface ModelHealthReport {
  window: number;
  tradeCount: number;
  actualWinRate: number;
  predictedWinRate: number;
  winRateDelta: number;
  brierScore: number;
  expectedCalibrationError: number;
  ev: number;
  actualPnl: number;
  roi: number;
  maxDrawdown: number;
  isDegraded: boolean;
  degradationReason: string;
}

export interface ProbabilityBucket {
  bucket: string;
  count: number;
  predictedProb: number;
  actualWinRate: number;
  calibrationError: number;
}

export interface ReliabilityDiagram {
  buckets: ProbabilityBucket[];
  overallBrierScore: number;
  overallECE: number;
}

export interface EdgeBucketStats {
  bucket: string;
  tradeCount: number;
  winRate: number;
  avgEffectiveEdge: number;
  actualRoi: number;
  expectedRoi: number;
}

export interface SettlementSummary {
  periodDays: number;
  settled: number;
  pending: number;
  wins: number;
  losses: number;
  ties: number;
  winRate: number;
  totalPnl: number;
  totalStaked: number;
  roi: number;
  estimatedCount: number;
  confirmedCount: number;
}