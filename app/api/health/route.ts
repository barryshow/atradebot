import { NextResponse } from "next/server";
import { processManager } from "@/lib/server/process-manager";

export async function GET() {
  const state = processManager.getState();
  return NextResponse.json({
    ok: true,
    engine: state.state,
    pid: state.pid,
    uptime: state.uptime,
    balance: state.balance,
    wins: state.wins,
    losses: state.losses,
    timestamp: new Date().toISOString(),
  });
}
