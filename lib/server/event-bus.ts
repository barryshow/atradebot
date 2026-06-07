import { EventEmitter } from "events";
import type { EngineEvent } from "@/lib/types/engine";

const MAX_RECENT = 500;

class EventBus {
  private emitter = new EventEmitter();
  private recent: EngineEvent[] = [];

  emit(event: EngineEvent) {
    this.recent.push(event);
    if (this.recent.length > MAX_RECENT) {
      this.recent = this.recent.slice(-MAX_RECENT);
    }
    this.emitter.emit("event", event);
  }

  subscribe(handler: (event: EngineEvent) => void): () => void {
    this.emitter.on("event", handler);
    return () => this.emitter.off("event", handler);
  }

  getRecent(count = 50): EngineEvent[] {
    return this.recent.slice(-count);
  }

  getLatestByType(type: string): EngineEvent | undefined {
    for (let i = this.recent.length - 1; i >= 0; i--) {
      if (this.recent[i].type === type) return this.recent[i];
    }
    return undefined;
  }

  clear() {
    this.recent = [];
  }
}

// Singleton via globalThis to survive Next.js HMR in dev
const GLOBAL_KEY = "__atradebot_event_bus__";

function getEventBus(): EventBus {
  if (!(globalThis as Record<string, unknown>)[GLOBAL_KEY]) {
    (globalThis as Record<string, unknown>)[GLOBAL_KEY] = new EventBus();
  }
  return (globalThis as Record<string, unknown>)[GLOBAL_KEY] as EventBus;
}

export const eventBus = getEventBus();
