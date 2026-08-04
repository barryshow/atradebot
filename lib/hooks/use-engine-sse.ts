"use client";
import { useEffect, useReducer, useRef, useCallback } from "react";
import type {
  EngineEvent,
  EngineStatus,
  SymbolSnapshot,
  TradeRecord,
  MarketRegimeType,
} from "@/lib/types/engine";

interface State {
  connected: boolean;
  status: EngineStatus;
  symbols: Record<string, Partial<SymbolSnapshot>>;
  recentTrades: TradeRecord[];
  recentEvents: EngineEvent[];
  logs: string[];
  regime: Record<string, MarketRegimeType>;
  regimeConfidence: Record<string, number>;
  expertVotes: Record<string, Record<string, number>>;
  edge: Record<string, Record<string, unknown>>;
  modelHealth: Record<string, unknown>;
  calibrationStatus: string;
  lastRejected: Array<{ symbol: string; reason: string; detail?: string }>;
  shadowTrades: Array<Record<string, unknown>>;
  // v6 new fields
  fastScans: Array<Record<string, unknown>>;
  settlementAvailable: boolean;
  payoutAvailable: boolean;
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
  maxConcurrentTrades: 1,
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
  fastScans: [],
  settlementAvailable: false,
  payoutAvailable: false,
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
          return {
            ...state,
            status: newStatus,
            settlementAvailable: (p as Record<string, unknown>).settlement_available as boolean ?? state.settlementAvailable,
            payoutAvailable: (p as Record<string, unknown>).payout_available as boolean ?? state.payoutAvailable,
            recentEvents,
          };
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
        // ── v6: Fast scan with EV ──
        case "fast_scan": {
          const sym = p.symbol as string;
          const scan = {
            symbol: sym,
            direction: p.direction as string,
            fastProb: p.fast_prob as number,
            slowProb: p.slow_prob as number,
            calibratedProb: p.calibrated_prob as number,
            evCall: p.ev_call as number,
            evPut: p.ev_put as number,
            selectedEv: p.selected_ev as number,
            status: p.status as string,
            payoutSource: p.payout_source as string,
            activeContracts: p.active_contracts as number,
            ts: evt.ts,
          };
          return {
            ...state,
            fastScans: [...state.fastScans, scan].slice(-100),
            symbols: {
              ...state.symbols,
              [sym]: {
                ...state.symbols[sym],
                symbol: sym,
                mlProb: scan.fastProb,
                direction: scan.direction as "CALL" | "PUT" | null,
              },
            },
            recentEvents,
          };
        }
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
          for (const [name, v] of Object.entries(votes || {})) {
            voteMap[name] = v.prob;
          }
          return {
            ...state,
            expertVotes: { ...state.expertVotes, [sym]: voteMap },
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
        // ── v6 shadow events ──
        case "shadow_order_created":
        case "shadow_trade": {
          return {
            ...state,
            shadowTrades: [...state.shadowTrades, p as Record<string, unknown>].slice(-100),
            recentEvents,
          };
        }
        case "shadow_order_settled": {
          return {
            ...state,
            shadowTrades: [...state.shadowTrades, { ...p, ts: evt.ts, settled: true }].slice(-100),
            recentEvents,
          };
        }
        case "trade_executed": {
          const trade: TradeRecord = {
            ts: evt.ts,
            symbol: p.symbol as string,
            direction: p.direction as "CALL" | "PUT",
            amount: p.amount as number,
            entryPrice: (p.entryPrice as number) ?? (p.entry_price as number) ?? 0,
            mlProb: (p.ev as number) ?? (p.calibratedProbability as number) ?? 0.5,
            aiApproval: 0,
            aiReason: `EV=${((p.ev as number) ?? 0).toFixed(4)}`,
            riskGates: [],
            result: "pending",
            flipped: false,
          };
          return {
            ...state,
            recentTrades: [...state.recentTrades, trade].slice(-100),
            recentEvents,
          };
        }
        case "trade_result": {
          const tid = p.trade_id as string;
          const trades = state.recentTrades.map((t, idx) =>
            t.result === "pending" && state.recentTrades.length - idx <= 20
              ? { ...t, result: (p.result as "win" | "loss") || "pending", pnl: p.pnl as number }
              : t
          );
          return { ...state, recentTrades: trades, recentEvents };
        }
        case "balance_update":
          return {
            ...state,
            status: { ...state.status, balance: p.balance as number },
            recentEvents,
          };
        case "log":
          return {
            ...state,
            logs: [...state.logs, p.msg as string].slice(-300),
            recentEvents,
          };
        case "funnel": {
          // Funnel data is informational, log it
          const ft = p.type as string;
          if (ft === "strategy") {
            return {
              ...state,
              logs: [
                ...state.logs,
                `[FUNNEL ${p.symbol}] passed=${p.decision_passed || 0} reject_ev=${p.reject_insufficient_ev || 0} shadow=${p.shadow_trade || 0}`,
              ].slice(-300),
              recentEvents,
            };
          }
          return { ...state, recentEvents };
        }
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
