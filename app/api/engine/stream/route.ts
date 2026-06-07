import { eventBus } from "@/lib/server/event-bus";
import { processManager } from "@/lib/server/process-manager";

export const dynamic = "force-dynamic";

export async function GET() {
  const encoder = new TextEncoder();
  let streamCleanup: (() => void) | undefined;

  const stream = new ReadableStream({
    start(controller) {
      // Send initial state snapshot
      const snapshot = processManager.getState();
      controller.enqueue(
        encoder.encode(`data: ${JSON.stringify({ type: "status", ts: Date.now(), payload: snapshot })}\n\n`)
      );

      // Send recent events for context
      const recent = eventBus.getRecent(20);
      for (const evt of recent) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(evt)}\n\n`));
      }

      // Subscribe to live events
      const unsubscribe = eventBus.subscribe((event) => {
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        } catch {
          // Client disconnected
        }
      });

      // Heartbeat every 15s
      const heartbeat = setInterval(() => {
        try {
          controller.enqueue(encoder.encode(": heartbeat\n\n"));
        } catch {
          clearInterval(heartbeat);
        }
      }, 15000);

      // Store refs for cleanup in cancel()
      streamCleanup = () => {
        clearInterval(heartbeat);
        unsubscribe();
      };
    },
    cancel() {
      streamCleanup?.();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
