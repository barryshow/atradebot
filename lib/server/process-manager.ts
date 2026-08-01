import { spawn, exec, type ChildProcess } from "child_process";
import { createInterface } from "readline";
import path from "path";
import { EventEmitter } from "events";
import { eventBus } from "./event-bus";
import { logStore } from "./log-store";
import type { EngineState, EngineEvent, EngineStatus } from "@/lib/types/engine";

const isWindows = process.platform === "win32";

export class ProcessManager extends EventEmitter {
  private process: ChildProcess | null = null;
  private state: EngineState = "stopped";
  private startTime: number | null = null;
  private wins = 0;
  private losses = 0;
  private balance = 0;
  private activeTrades = 0;
  private _stopping = false;

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

  async start(mode?: string): Promise<{ ok: boolean; error?: string }> {
    if (this.process) {
      return { ok: false, error: `Process already exists (state=${this.state})` };
    }
    if (this._stopping) {
      return { ok: false, error: "Process is being stopped, wait for it to fully exit" };
    }

    // ── Windows: kill stale PID lock before starting ──
    if (isWindows) {
      try {
        exec("del /f /q %TEMP%\\atradebot_engine.pid 2>nul");
      } catch { /* ok */ }
    }

    let pythonPath = process.env.PYTHON_PATH;
    if (!pythonPath) {
      pythonPath = isWindows ? "python" : "python3";
    }
    const engineDir = path.resolve(/* turbopackIgnore: true */ process.cwd(), "lib", "engine");
    const mainPy = path.join(engineDir, "main.py");

    return new Promise((resolve) => {
      try {
        this.process = spawn(pythonPath, [mainPy, "--auto", "--mode", mode || "shadow"], {
          cwd: path.resolve(process.cwd()),
          stdio: ["pipe", "pipe", "pipe"],
          env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
        });

        this.state = "starting";
        this.startTime = Date.now();
        this.emitState();

        // Parse stdout JSON lines — force UTF-8 on Windows (avoids GBK decode error)
        this.process.stdout!.setEncoding("utf-8");
        const rl = createInterface({ input: this.process.stdout! });
        let started = false;
        rl.on("line", (line) => {
          try {
            const event: EngineEvent = JSON.parse(line);
            this.handleEvent(event);
            eventBus.emit(event);
            // Resolve when engine reports "running" state for the first time
            if (!started && event.type === "status" && (event.payload as any).state === "running") {
              started = true;
              resolve({ ok: true });
            }
          } catch {
            logStore.add(`[stdout] ${line}`);
          }
        });

        // Capture stderr as logs
        this.process.stderr!.setEncoding("utf-8");
        const errRl = createInterface({ input: this.process.stderr! });
      errRl.on("line", (line) => {
        logStore.add(`[python] ${line}`);
      });

      this.process.on("exit", (code) => {
        logStore.add(`[process] Python exited with code ${code}`);
        this.state = "stopped";
        this.process = null;
        this._stopping = false;
        this.emitState();
      });

      this.process.on("error", (err) => {
        logStore.add(`[process] Error: ${err.message}`);
        this.state = "error";
        this.emitState();
        if (!started) resolve({ ok: false, error: err.message });
      });

      // Failsafe: resolve after 15s even if "running" event never arrives
      setTimeout(() => {
        if (!started) {
          started = true;
          logStore.add("[process] Start timeout — engine may still be starting");
          resolve({ ok: true });
        }
      }, 15000);
    } catch (err) {
      this.state = "error";
      this.emitState();
      resolve({ ok: false, error: String(err) });
    }
    });
  }

  async stop(): Promise<{ ok: boolean }> {
    if (!this.process) {
      this.state = "stopped";
      this._stopping = false;
      this.emitState();
      return { ok: true };
    }
    if (this._stopping) {
      logStore.add("[process] Already stopping, waiting...");
      return { ok: true };
    }
    this._stopping = true;
    this.state = "stopped";
    this.emitState();
    this.sendCommand("stop");

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        try {
          if (isWindows) {
            exec(`taskkill /f /pid ${this.process!.pid!} 2>nul`);
          } else {
            this.process?.kill("SIGKILL");
          }
        } catch { /* process already dead */ }
        this.process = null;
        this._stopping = false;
        this.emitState();
        logStore.add(`[process] Stopped via ${isWindows ? "taskkill" : "SIGKILL"} (timeout)`);
        resolve({ ok: true });
      }, 5000);

      const proc = this.process!;
      const cleanup = () => {
        clearTimeout(timeout);
        this.state = "stopped";
        this.process = null;
        this._stopping = false;
        this.emitState();
        logStore.add("[process] Stopped cleanly");
        resolve({ ok: true });
      };

      if (!proc.killed && proc.exitCode === null) {
        proc.once("exit", cleanup);
      } else {
        cleanup();
      }
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

  sendCommand(command: string, args?: Record<string, unknown>) {
    if (!this.process?.stdin?.writable) {
      logStore.add(`[process] Cannot send command "${command}": stdin not writable`);
      return;
    }
    const cmd = JSON.stringify({ command, ...(args || {}) });
    try {
      this.process.stdin.write(cmd + "\n", "utf-8");
      logStore.add(`[process] Sent command: ${command}`);
    } catch (err) {
      logStore.add(`[process] Failed to send command "${command}": ${err}`);
    }
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
