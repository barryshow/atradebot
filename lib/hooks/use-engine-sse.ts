"use client";
import { useEffect, useReducer, useRef, useCallback } from "react";
import type {
  EngineEvent,
  EngineStatus,
  SymbolSnapshot,
  TradeRecord,
  RiskGateResult,
} from "@/lib/types/engine";

interface State {
  connected: boolean;
  status: EngineStatus;
  symbols: Record<string, Partial<SymbolSnapshot>>;
  recentTrades: TradeRecord[];
  recentEvents: EngineEvent[];
  logs: string[];
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
        case "risk_gate": {
          const sym = p.symbol as string;
          const existing = state.symbols[sym] || { symbol: sym };
          const gates = [...(existing.riskGates || []), p as unknown as RiskGateResult].slice(-5);
          return {
            ...state,
            symbols: {
              ...state.symbols,
              [sym]: { ...existing, riskGates: gates },
            },
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
