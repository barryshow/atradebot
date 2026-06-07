import { NextRequest, NextResponse } from "next/server";
import { processManager } from "@/lib/server/process-manager";

export async function GET() {
  return NextResponse.json(processManager.getState());
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  const action = body.action as string;

  switch (action) {
    case "start": {
      const result = await processManager.start();
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
    default:
      return NextResponse.json({ ok: false, error: `Unknown action: ${action}` }, { status: 400 });
  }
}
