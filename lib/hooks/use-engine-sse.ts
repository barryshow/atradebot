"use client";
import { useEffect, useReducer, useRef, useCallback } from "react";
import type {
  EngineEvent,
  EngineStatus,
  SymbolSnapshot,
  TradeRecord,
  RiskGateResult,
  MarketRegimeType,
  EdgeResult,
  RejectReasonCode,
} from "@/lib/types/engine";

interface State {
  connected: boolean;
  status: EngineStatus;
  symbols: Record<string, Partial<SymbolSnapshot>>;
  recentTrades: TradeRecord[];
  recentEvents: EngineEvent[];
  logs: string[];
  // EventEdge V2
  regime: Record<string, MarketRegimeType>;
  regimeConfidence: Record<string, number>;
  expertVotes: Record<string, Record<string, number>>;
  edge: Record<string, Partial<EdgeResult>>;
  modelHealth: Record<string, unknown>;
  calibrationStatus: string;
  lastRejected: Array<{ symbol: string; reason: string; detail?: string }>;
  shadowTrades: Array<Record<string, unknown>>;
}

type Action =
  | { type: "connected" }
  | { type: "disconnected" }
  | { type: "event"; event: EngineEvent };

const initialStatus: EngineStatus = {
  state: "stopped",
  pid: null,
  uptime: 0,
  tradeCountToday: 0,
  wins: 0,
  losses: 0,
  activeTrades: 0,
  maxConcurrentTrades: 3,
  balance: 0,
  lastTick: null,
};

const initialState: State = {
  connected: false,
  status: initialStatus,
  symbols: {},
  recentTrades: [],
  recentEvents: [],
  logs: [],
  regime: {},
  regimeConfidence: {},
  expertVotes: {},
  edge: {},
  modelHealth: {},
  calibrationStatus: "NOT_READY",
  lastRejected: [],
  shadowTrades: [],
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "connected":
      return { ...state, connected: true };
    case "disconnected":
      return { ...state, connected: false };
    case "event": {
      const evt = action.event;
      const p = evt.payload as Record<string, unknown>;
      const recentEvents = [...state.recentEvents, evt].slice(-200);

      switch (evt.type) {
        case "status": {
          const newStatus = { ...state.status, ...(p as Partial<EngineStatus>) };
          return { ...state, status: newStatus, recentEvents };
        }
        case "features": {
          const sym = p.symbol as string;
          const ind = p.indicators as Record<string, number>;
          return {
            ...state,
            symbols: {
              ...state.symbols,
              [sym]: { ...state.symbols[sym], symbol: sym, ...ind },
            },
            recentEvents,
          };
        }
        case "prediction": {
          const sym = p.symbol as string;
          return {
            ...state,
            symbols: {
              ...state.symbols,
              [sym]: {
                ...state.symbols[sym],
                symbol: sym,
                mlProb: p.prob_win as number,
                direction: p.direction === 1 ? "CALL" : "PUT",
              },
            },
            recentEvents,
          };
        }
        // ── EventEdge V2 Events ──
        case "regime_update": {
          const sym = p.symbol as string;
          return {
            ...state,
            regime: { ...state.regime, [sym]: p.regime as MarketRegimeType },
            regimeConfidence: { ...state.regimeConfidence, [sym]: p.confidence as number },
            symbols: {
              ...state.symbols,
              [sym]: { ...state.symbols[sym], symbol: sym, regime: p.regime as MarketRegimeType },
            },
            recentEvents,
          };
        }
        case "expert_votes": {
          const sym = p.symbol as string;
          const votes = p.votes as Record<string, { prob: number; dir: string }>;
          const voteMap: Record<string, number> = {};
          for (const [name, v] of Object.entries(votes)) {
            voteMap[name] = v.prob;
          }
          return {
            ...state,
            expertVotes: { ...state.expertVotes, [sym]: voteMap },
            recentEvents,
          };
        }
        case "edge_calculation": {
          const sym = p.symbol as string;
          return {
            ...state,
            edge: {
              ...state.edge,
              [sym]: {
                calibratedProbability: p.calibrated_probability as number,
                conservativeProbability: p.conservative_probability as number,
                breakEvenProbability: p.break_even_probability as number,
                effectiveEdge: p.effective_edge as number,
                expectedRoi: p.expected_roi as number,
                passed: p.passed as boolean,
                rejectReason: (p.reject_reason as RejectReasonCode) || "",
              },
            },
            recentEvents,
          };
        }
        case "model_health": {
          return {
            ...state,
            modelHealth: {
              ...state.modelHealth,
              window: p.window as number,
              isDegraded: p.is_degraded as boolean,
              actualWinRate: p.actual_win_rate as number,
              predictedWinRate: p.predicted_win_rate as number,
              winRateDelta: p.win_rate_delta as number,
              brierScore: p.brier_score as number,
              ece: (p as Record<string, unknown>).ece as number,
              tradeCount: (p as Record<string, unknown>).trade_count as number,
            },
            recentEvents,
          };
        }
        case "calibration_status": {
          return {
            ...state,
            calibrationStatus: (p.status as string) || "NOT_READY",
            recentEvents,
          };
        }
        case "trade_rejected": {
          return {
            ...state,
            lastRejected: [
              ...state.lastRejected,
              { symbol: p.symbol as string, reason: p.reason as string, detail: (p as Record<string, unknown>).detail as string },
            ].slice(-20),
            recentEvents,
          };
        }
        case "shadow_trade": {
          return {
            ...state,
            shadowTrades: [...state.shadowTrades, p as Record<string, unknown>].slice(-100),
            recentEvents,
          };
        }
        case "trade_executed": {
          const trade: TradeRecord = {
            ts: evt.ts,
            symbol: p.symbol as string,
            direction: p.direction as "CALL" | "PUT",
            amount: p.amount as number,
            entryPrice: p.entryPrice as number,
            mlProb: p.mlProb as number,
            aiApproval: 0,
            aiReason: p.aiReason as string,
            riskGates: [],
            result: "pending",
            flipped: p.flipped as boolean,
          };
          return {
            ...state,
            recentTrades: [...state.recentTrades, trade].slice(-100),
            recentEvents,
          };
        }
        case "trade_result": {
          const sym = p.symbol as string;
          const trades = state.recentTrades.map((t) =>
            t.symbol === sym && t.result === "pending"
              ? { ...t, result: p.result as "win" | "loss", pnl: p.pnl as number, exitPrice: p.exitPrice as number }
              : t
          );
          return { ...state, recentTrades: trades, recentEvents };
        }
        case "signal_flip": {
          const sym = p.symbol as string;
          const dir = p.newDirection === 1 ? "CALL" : "PUT";
          return {
            ...state,
            symbols: {
              ...state.symbols,
              [sym]: {
                ...state.symbols[sym],
                symbol: sym,
                direction: dir as "CALL" | "PUT",
                mlProb: p.newProb as number,
              },
            },
            recentEvents,
          };
        }
        case "balance_update":
          return {
            ...state,
            status: { ...state.status, balance: p.balance as number },
            recentEvents,
          };
        case "candle_update": {
          const sym = p.symbol as string;
          return {
            ...state,
            symbols: {
              ...state.symbols,
              [sym]: { ...state.symbols[sym], symbol: sym, price: p.close as number },
            },
            recentEvents,
          };
        }
        case "log":
          return {
            ...state,
            logs: [...state.logs, p.msg as string].slice(-300),
            recentEvents,
          };
        default:
          return { ...state, recentEvents };
      }
    }
    default:
      return state;
  }
}

export function useEngineSSE() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoff = useRef(1000);

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
    }

    const es = new EventSource("/api/engine/stream");
    esRef.current = es;

    es.onopen = () => {
      dispatch({ type: "connected" });
      backoff.current = 1000;
    };

    es.onmessage = (msg) => {
      try {
        const event: EngineEvent = JSON.parse(msg.data);
        dispatch({ type: "event", event });
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      dispatch({ type: "disconnected" });
      es.close();
      esRef.current = null;
      // Reconnect with backoff
      reconnectTimer.current = setTimeout(() => {
        backoff.current = Math.min(backoff.current * 2, 30000);
        connect();
      }, backoff.current);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      esRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  return state;
}
