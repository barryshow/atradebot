import { spawn, type ChildProcess } from "child_process";
import { createInterface } from "readline";
import path from "path";
import { EventEmitter } from "events";
import { eventBus } from "./event-bus";
import { logStore } from "./log-store";
import type { EngineState, EngineEvent, EngineStatus } from "@/lib/types/engine";

export class ProcessManager extends EventEmitter {
  private process: ChildProcess | null = null;
  private state: EngineState = "stopped";
  private startTime: number | null = null;
  private wins = 0;
  private losses = 0;
  private balance = 0;
  private activeTrades = 0;
  private restartCount = 0;
  private maxRestarts = 3;

  getState(): EngineStatus {
    return {
      state: this.state,
      pid: this.process?.pid ?? null,
      uptime: this.startTime ? Math.floor((Date.now() - this.startTime) / 1000) : 0,
      tradeCountToday: this.wins + this.losses,
      wins: this.wins,
      losses: this.losses,
      activeTrades: this.activeTrades,
      maxConcurrentTrades: 3,
      balance: this.balance,
      lastTick: null,
    };
  }

  async start(): Promise<{ ok: boolean; error?: string }> {
    if (this.process && this.state !== "stopped") {
      return { ok: false, error: `Already ${this.state}` };
    }

    let pythonPath = process.env.PYTHON_PATH;
    if (!pythonPath) {
      pythonPath = process.platform === "win32" ? "python" : "python3";
    }
    const engineDir = path.resolve(/* turbopackIgnore: true */ process.cwd(), "lib", "engine");
    const mainPy = path.join(engineDir, "main.py");

    try {
      this.process = spawn(pythonPath, [mainPy], {
        cwd: path.resolve(process.cwd()),
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
      });

      this.state = "starting";
      this.startTime = Date.now();
      this.emitState();

      // Parse stdout JSON lines
      const rl = createInterface({ input: this.process.stdout! });
      rl.on("line", (line) => {
        try {
          const event: EngineEvent = JSON.parse(line);
          this.handleEvent(event);
          eventBus.emit(event);
        } catch {
          logStore.add(`[stdout] ${line}`);
        }
      });

      // Capture stderr as logs
      const errRl = createInterface({ input: this.process.stderr! });
      errRl.on("line", (line) => {
        logStore.add(`[python] ${line}`);
      });

      this.process.on("exit", (code) => {
        logStore.add(`[process] Python exited with code ${code}`);
        this.state = "stopped";
        this.process = null;
        this.emitState();

        // Auto-restart on unexpected exit
        if (code !== 0 && this.restartCount < this.maxRestarts) {
          this.restartCount++;
          const delay = Math.min(2000 * this.restartCount, 10000);
          logStore.add(`[process] Auto-restart in ${delay}ms (attempt ${this.restartCount})`);
          setTimeout(() => this.start(), delay);
        }
      });

      this.process.on("error", (err) => {
        logStore.add(`[process] Error: ${err.message}`);
        this.state = "error";
        this.emitState();
      });

      // Send start command
      setTimeout(() => this.sendCommand("start"), 500);
      this.restartCount = 0;
      return { ok: true };
    } catch (err) {
      this.state = "error";
      this.emitState();
      return { ok: false, error: String(err) };
    }
  }

  async stop(): Promise<{ ok: boolean }> {
    if (!this.process) {
      return { ok: true };
    }
    this.sendCommand("stop");
    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        this.process?.kill("SIGKILL");
        resolve({ ok: true });
      }, 5000);

      this.process!.on("exit", () => {
        clearTimeout(timeout);
        this.state = "stopped";
        this.process = null;
        this.emitState();
        resolve({ ok: true });
      });
    });
  }

  pause() {
    this.sendCommand("pause");
    this.state = "paused";
    this.emitState();
  }

  resume() {
    this.sendCommand("resume");
    this.state = "running";
    this.emitState();
  }

  private sendCommand(command: string, args?: Record<string, unknown>) {
    if (!this.process?.stdin?.writable) return;
    const cmd = JSON.stringify({ command, ...args });
    this.process.stdin.write(cmd + "\n");
  }

  private handleEvent(event: EngineEvent) {
    const p = event.payload as Record<string, unknown>;
    switch (event.type) {
      case "status":
        if (p.state === "running" || p.state === "paused" || p.state === "stopped") {
          this.state = p.state as EngineState;
          this.emitState();
        }
        if (typeof p.balance === "number") this.balance = p.balance;
        if (typeof p.wins === "number") this.wins = p.wins as number;
        if (typeof p.losses === "number") this.losses = p.losses as number;
        if (typeof p.activeTrades === "number") this.activeTrades = p.activeTrades as number;
        break;
      case "balance_update":
        if (typeof p.balance === "number") this.balance = p.balance as number;
        break;
      case "trade_executed":
        this.activeTrades++;
        break;
      case "trade_result":
        if (p.result === "win") this.wins++;
        else this.losses++;
        this.activeTrades = Math.max(0, this.activeTrades - 1);
        break;
    }
  }

  private emitState() {
    const status = this.getState();
    this.emit("state", status);
    eventBus.emit({ type: "status", ts: Date.now(), payload: status as unknown as Record<string, unknown> });
  }
}

// Singleton
const GLOBAL_KEY = "__atradebot_process_manager__";

function getProcessManager(): ProcessManager {
  if (!(globalThis as Record<string, unknown>)[GLOBAL_KEY]) {
    (globalThis as Record<string, unknown>)[GLOBAL_KEY] = new ProcessManager();
  }
  return (globalThis as Record<string, unknown>)[GLOBAL_KEY] as ProcessManager;
}

export const processManager = getProcessManager();
