import { NextRequest, NextResponse } from "next/server";
import { eventBus } from "@/lib/server/event-bus";

export async function GET(req: NextRequest) {
  const searchParams = req.nextUrl.searchParams;
  const limit = Number(searchParams.get("limit")) || 50;

  // Get trade events from the event bus
  const recent = eventBus.getRecent(500);
  const trades = recent
    .filter((e) => e.type === "trade_executed" || e.type === "trade_result")
    .slice(-limit);

  return NextResponse.json({ trades, total: trades.length });
}
