/** A single OHLCV candle from the CSV data */
export interface Candle {
  ts: number; // Unix ms
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** Formatted for TradingView lightweight-charts */
export interface CandlePoint {
  time: number; // Unix seconds (Epoch)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/** A trade marker to overlay on the chart */
export interface TradeMarker {
  time: number; // Unix seconds (Epoch)
  symbol: string;
  direction: "CALL" | "PUT";
  entryPrice: number;
  result?: "win" | "loss" | "pending";
  pnl?: number;
  amount: number;
}

/** API response shape */
export interface CandlesResponse {
  symbol: string;
  candles: CandlePoint[];
  tradeMarkers: TradeMarker[];
}

/** Per-symbol candle store (keyed by symbol) */
export interface CandleStore {
  [symbol: string]: CandlePoint[];
}

/** Subset of TradeRecord for marker use */
export interface TradeRecordFlat {
  ts: number;
  symbol: string;
  direction: "CALL" | "PUT";
  entryPrice: number;
  amount: number;
  result?: "win" | "loss" | "pending";
  pnl?: number;
}
