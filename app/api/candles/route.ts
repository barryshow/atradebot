import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { getServerConfig } from "@/lib/server/config";
import type { CandlePoint, TradeRecordFlat } from "@/lib/types/candles";

interface CsvRow {
  ts: string;
  symbol: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
}

/**
 * Parse the shared hibt_ticks.csv into structured candle data.
 * CSV columns: ts,symbol,open,high,low,close,volume (first 7 cols).
 */
function parseCsvFile(filePath: string, symbol?: string): CandlePoint[] {
  if (!fs.existsSync(filePath)) return [];

  const raw = fs.readFileSync(filePath, "utf-8").trim();
  if (!raw) return [];

  const lines = raw.split("\n");
  const rows: CsvRow[] = [];
  let headerSkipped = false;

  for (const line of lines) {
    const parts = line.split(",");
    if (parts.length < 7) continue;
    const ts = parts[0].trim();
    const sym = parts[1].trim();
    // Skip CSV header row
    if (!headerSkipped && (ts === "ts" || sym === "symbol")) {
      headerSkipped = true;
      continue;
    }
    // Filter by symbol if specified
    if (symbol && sym !== symbol) continue;

    rows.push({
      ts,
      symbol: sym,
      open: parts[2].trim(),
      high: parts[3].trim(),
      low: parts[4].trim(),
      close: parts[5].trim(),
      volume: parts[6].trim(),
    });
  }

  return rows
    .map((r) => ({
      time: Math.floor(Number(r.ts) / 1000), // Epoch seconds
      open: Number(r.open),
      high: Number(r.high),
      low: Number(r.low),
      close: Number(r.close),
      volume: Number(r.volume),
    }))
    .filter((c) => c.time > 0 && c.open > 0 && c.high > 0 && c.low > 0 && c.close > 0)
    .sort((a, b) => a.time - b.time); // 按时间升序排列
}

/**
 * Merge trade records into markers that can be overlaid on the chart.
 */
function buildTradeMarkers(symbol: string, trades: TradeRecordFlat[]): TradeRecordFlat[] {
  return trades.filter((t) => t.symbol === symbol);
}

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const symbol = searchParams.get("symbol") || "BTCUSDT";
  const limit = Number(searchParams.get("limit")) || 500;

  const config = getServerConfig();
  const candles = parseCsvFile(config.radarCsvPath, symbol);

  // If there are more candles than the limit, we sample at a regular interval
  // to keep the chart rendering smooth while showing full range
  let sampled: CandlePoint[];
  if (candles.length <= limit) {
    sampled = candles;
  } else {
    const step = candles.length / limit;
    sampled = [];
    for (let i = 0; i < candles.length; i += step) {
      sampled.push(candles[Math.floor(i)]);
    }
    // Always include the last candle
    if (sampled[sampled.length - 1] !== candles[candles.length - 1]) {
      sampled.push(candles[candles.length - 1]);
    }
  }

  return NextResponse.json({
    symbol,
    candles: sampled,
  });
}
