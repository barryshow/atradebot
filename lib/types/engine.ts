export type EngineState = "stopped" | "starting" | "running" | "paused" | "error";

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
  bootstrapProfitTarget?: number;
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
  | "pong";

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
