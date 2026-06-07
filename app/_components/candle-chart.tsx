"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type LineData,
  type HistogramData,
  type Time,
  CrosshairMode,
  ColorType,
} from "lightweight-charts";
import type { CandlePoint, TradeRecordFlat } from "@/lib/types/candles";
import { SYMBOLS } from "@/lib/types/engine";

interface Props {
  activeSymbol?: string;
  trades: TradeRecordFlat[];
}

type ChartSeries = {
  candleSeries: ISeriesApi<"Candlestick">;
  volumeSeries: ISeriesApi<"Histogram">;
};

const symbolColors: Record<string, string> = {
  BTCUSDT: "#f7931a",
  ETHUSDT: "#627eea",
  SOLUSDT: "#9945ff",
};

export function CandleChart({ activeSymbol = "BTCUSDT", trades }: Props) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ChartSeries | null>(null);
  const [symbol, setSymbol] = useState(activeSymbol);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---------- Fetch candles from API ----------
  const fetchCandles = useCallback(async () => {
    try {
      const res = await fetch(`/api/candles?symbol=${symbol}&limit=300`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setLoading(false);
      setError(null);
      return data.candles as CandlePoint[];
    } catch (err) {
      setError(String(err));
      setLoading(false);
      return null;
    }
  }, [symbol]);

  // ---------- Build chart ----------
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111" },
        textColor: "#9ca3af",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#6366f1", width: 1, style: 2, labelBackgroundColor: "#6366f1" },
        horzLine: { color: "#6366f1", width: 1, style: 2, labelBackgroundColor: "#6366f1" },
      },
      rightPriceScale: {
        borderColor: "#374151",
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      timeScale: {
        borderColor: "#374151",
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        fixRightEdge: true,
      },
      width: chartContainerRef.current.clientWidth,
      height: 420,
    });

    // Candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#22c55e",
      wickDownColor: "#ef4444",
      wickUpColor: "#22c55e",
    });

    // Volume histogram
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    seriesRef.current = { candleSeries, volumeSeries };

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // ---------- Load data ----------
  const [candleCache, setCandleCache] = useState<CandlePoint[]>([]);

  useEffect(() => {
    if (!seriesRef.current) return;

    let cancelled = false;

    async function load(retries = 5) {
      setLoading(true);
      // Wait for chart to be ready
      for (let r = 0; r < retries; r++) {
        if (seriesRef.current) break;
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      if (cancelled || !seriesRef.current) return;

      const candles = await fetchCandles();
      if (cancelled || !candles || !seriesRef.current) return;

      const { candleSeries, volumeSeries } = seriesRef.current;

      const cdData: CandlestickData[] = candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }));

      const volData: HistogramData[] = candles.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
      }));

      try {
        candleSeries.setData(cdData);
        volumeSeries.setData(volData);
        chartRef.current?.timeScale().fitContent();
        setCandleCache(candles);
      } catch (e) {
        console.error('Chart load error:', e);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [symbol, fetchCandles]);

  // ---------- Poll every 5 seconds for new candles ----------
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      if (!seriesRef.current) return;
      const candles = await fetchCandles();
      if (!candles || candles.length === 0 || !seriesRef.current) return;

      const { candleSeries, volumeSeries } = seriesRef.current;
      const last = candles[candles.length - 1];
      const lastVol = candles[candles.length - 1];

      candleSeries.update({
        time: last.time as Time,
        open: last.open,
        high: last.high,
        low: last.low,
        close: last.close,
      });
      volumeSeries.update({
        time: lastVol.time as Time,
        value: lastVol.volume,
        color: lastVol.close >= lastVol.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
      });
    }, 5000);

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [symbol, fetchCandles]);

  // ---------- Overlay trade markers ----------
  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;

    const { candleSeries } = seriesRef.current;
    const symbolTrades = trades
      .filter((t) => t.symbol === symbol && t.entryPrice > 0)
      .slice(-20);

    candleSeries.setMarkers(
      symbolTrades.map((t) => ({
        time: Math.floor(t.ts / 1000) as Time,
        position: t.direction === "CALL" ? "belowBar" : "aboveBar",
        color: t.result === "win" ? "#22c55e" : t.result === "loss" ? "#ef4444" : "#eab308",
        shape: t.direction === "CALL" ? "arrowUp" : "arrowDown",
        text: `${t.direction === "CALL" ? "▲" : "▼"} ${t.entryPrice.toFixed(0)}`,
        size: 1,
      }))
    );
  }, [trades, symbol]);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      {/* Header with symbol selector */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <div className="flex items-center gap-1">
          {SYMBOLS.map((s) => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                symbol === s
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              {s.replace("USDT", "")}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          {loading && (
            <span className="text-xs text-gray-500 animate-pulse">加载中...</span>
          )}
          {error && (
            <span className="text-xs text-red-400" title={error}>
              数据异常
            </span>
          )}
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: symbolColors[symbol] || "#6b7280" }}
          />
        </div>
      </div>

      {/* Chart area */}
      {loading && candleCache.length === 0 && (
        <div className="flex items-center justify-center h-[420px] text-gray-500 text-sm">
          <div className="text-center space-y-2">
            <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full mx-auto" />
            <span>等待K线数据...</span>
          </div>
        </div>
      )}

      {error && !loading && (
        <div className="flex items-center justify-center h-[420px] text-gray-500 text-sm">
          <div className="text-center space-y-2">
            <div className="text-2xl">⚠️</div>
            <span>无法获取行情数据</span>
            <div className="text-xs text-gray-600">检查引擎是否运行且CSV有数据</div>
          </div>
        </div>
      )}

      <div ref={chartContainerRef} className={loading && candleCache.length === 0 && !error ? "hidden" : ""} />
    </div>
  );
}
