import { NextRequest, NextResponse } from "next/server";
import { processManager } from "@/lib/server/process-manager";

export async function GET() {
  return NextResponse.json(processManager.getState());
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const action = body.action as string;
  const mode = (body.mode as string) || "shadow";

  switch (action) {
    case "start": {
      const result = await processManager.start(mode);
      return NextResponse.json({ ok: result.ok, state: processManager.getState(), error: result.error });
    }
    case "stop": {
      const result = await processManager.stop();
      return NextResponse.json({ ok: result.ok, state: processManager.getState() });
    }
    case "pause": {
      processManager.pause();
      return NextResponse.json({ ok: true, state: processManager.getState() });
    }
    case "resume": {
      processManager.resume();
      return NextResponse.json({ ok: true, state: processManager.getState() });
    }
    case "manual_order_test": {
      // 手动下单测试 — 仅验证接口，不影响策略统计
      const symbol = body.symbol as string;
      const direction = body.direction as number; // 1=CALL, 2=PUT
      const amount = body.amount as number || 3;
      if (!symbol || !direction) {
        return NextResponse.json({ ok: false, error: "Missing symbol or direction" }, { status: 400 });
      }
      if (amount < 3 || amount !== Math.floor(amount)) {
        return NextResponse.json({ ok: false, error: "Amount must be integer >= 3 USDT" }, { status: 400 });
      }
      processManager.sendCommand("manual_order_test", { symbol, direction, amount });
      return NextResponse.json({ ok: true, msg: "Manual order test command sent" });
    }
    default:
      return NextResponse.json({ ok: false, error: `Unknown action: ${action}` }, { status: 400 });
  }
}
