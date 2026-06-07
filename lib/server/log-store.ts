const MAX_LOGS = 500;

class LogStore {
  private logs: string[] = [];

  add(line: string) {
    this.logs.push(line);
    if (this.logs.length > MAX_LOGS) {
      this.logs = this.logs.slice(-MAX_LOGS);
    }
  }

  getRecent(count = 100): string[] {
    return this.logs.slice(-count);
  }

  clear() {
    this.logs = [];
  }
}

const GLOBAL_KEY = "__atradebot_log_store__";

function getLogStore(): LogStore {
  if (!(globalThis as Record<string, unknown>)[GLOBAL_KEY]) {
    (globalThis as Record<string, unknown>)[GLOBAL_KEY] = new LogStore();
  }
  return (globalThis as Record<string, unknown>)[GLOBAL_KEY] as LogStore;
}

export const logStore = getLogStore();
