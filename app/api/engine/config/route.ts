import { NextRequest, NextResponse } from "next/server";

export async function GET() {
  // Return non-sensitive config
  return NextResponse.json({
    symbols: (process.env.TRADE_SYMBOLS || "BTCUSDT,ETHUSDT,SOLUSDT").split(","),
    holdMinutes: Number(process.env.HOLD_MINUTES) || 5,
    maxConcurrentTrades: Number(process.env.MAX_CONCURRENT_TRADES) || 3,
    tradeCooldownSec: Number(process.env.TRADE_COOLDOWN_SEC) || 180,
    rejectCooldownSec: Number(process.env.REJECT_COOLDOWN_SEC) || 45,
    minProbability: Number(process.env.MIN_PROBABILITY) || 0.556,
    bbDeadZone: [Number(process.env.BB_DEAD_ZONE_LOW) || 0.4, Number(process.env.BB_DEAD_ZONE_HIGH) || 0.6],
    adxOscillating: Number(process.env.ADX_OSCILLATING_THRESHOLD) || 35,
    adxExtreme: Number(process.env.ADX_EXTREME_THRESHOLD) || 44,
    bbExtremeHigh: Number(process.env.BB_EXTREME_HIGH) || 0.7,
    bbExtremeLow: Number(process.env.BB_EXTREME_LOW) || 0.3,
  });
}

export async function PUT(req: NextRequest) {
  const body = await req.json();
  // In a full implementation, this would send config_update to the Python process
  return NextResponse.json({ ok: true, msg: "Config update received", config: body });
}
