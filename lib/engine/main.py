# -*- coding: utf-8 -*-
"""
Entry point: spawned by Next.js ProcessManager.
Communicates via stdout JSON lines, reads commands from stdin.

PID lock: ensures only one instance runs at a time regardless of how it's started.
"""
import sys
import io
import json
import time
import threading
import os
import warnings
import signal
import atexit
import tempfile

# ── PID lock ──────────────────────────────────────────────────────────
PID_FILE = os.path.join(tempfile.gettempdir(), "atradebot_engine.pid")

def _acquire_pid_lock() -> bool:
    """Write PID file with flock semantics. Returns False if another instance is running."""
    try:
        # Check existing PID
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            try:
                os.kill(old_pid, 0)  # Signal 0 = test if alive
                print(f"[PID Lock] Another engine already running (pid={old_pid}), exiting.", flush=True)
                return False
            except (OSError, ProcessLookupError):
                pass  # Stale PID file
            os.remove(PID_FILE)

        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception as e:
        print(f"[PID Lock] Warning: could not write PID file: {e}", flush=True)
    return True

def _release_pid_lock():
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(PID_FILE)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────

# Force UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

warnings.filterwarnings("ignore")
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(k, None)

# Add parent dir to path so imports work when spawned as subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.engine.engine import TradingEngine, emit
from lib.engine.shadow_mode import RunMode, set_run_mode, get_shadow_mode


def stdin_reader(engine: TradingEngine):
    """Read commands from stdin (sent by Node.js ProcessManager).
    Gracefully handles nohup/PM2 mode where stdin is closed."""
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = cmd.get("command", "")
            if action == "start":
                run_mode = cmd.get("mode", "live")
                engine.set_run_mode(run_mode)
                engine.start()
            elif action == "stop":
                engine.stop()
                break
            elif action == "pause":
                engine.pause()
            elif action == "resume":
                engine.resume()
            elif action == "ping":
                emit("pong", {"ts": int(time.time() * 1000)})
            elif action == "status":
                emit("status", engine.get_status())
    except (OSError, EOFError, ValueError):
        # stdin closed (nohup/PM2/systemd mode) — silent exit
        pass


def main():
    # ── PID lock ────────────────────────────────────────────
    if not _acquire_pid_lock():
        sys.exit(1)
    atexit.register(_release_pid_lock)
    # ────────────────────────────────────────────────────────

    # ── Parse args for auto-start ───────────────────────────
    import argparse
    parser_auto = argparse.ArgumentParser()
    parser_auto.add_argument("--auto", action="store_true", help="Auto-start engine without waiting for frontend command")
    parser_auto.add_argument("--mode", type=str, default="shadow", choices=["live", "shadow", "backtest"],
                        help="Run mode (default: shadow)")
    auto_args, _ = parser_auto.parse_known_args()
    # ────────────────────────────────────────────────────────

    engine = TradingEngine()

    # Start stdin command reader in background (waits for frontend commands)
    reader = threading.Thread(target=stdin_reader, args=(engine,), daemon=True)
    reader.start()

    # Emit initial idle state — engine is alive but waiting for frontend "start"
    emit("status", {"state": "stopped", "msg": "Engine ready, waiting for frontend start command"})

    # ── Auto-start if requested, or if running interactively ──
    if auto_args.auto or sys.stdin.isatty():
        if not auto_args.auto:
            emit("log", {"msg": "Interactive mode detected, auto-starting"})
        engine.set_run_mode(auto_args.mode)
        engine.start()
        emit("log", {"msg": f"Auto-start mode: {auto_args.mode}"})

    # Main loop — ticks when running, waits when stopped
    last_balance_check = time.time()
    idle_count = 0
    while True:
        if engine.running and not engine.paused:
            try:
                engine.tick()
            except Exception as e:
                emit("error", {"msg": f"Main loop error: {str(e)[:200]}"})

            # Periodic balance refresh (skip if shadow simulated balance)
            if time.time() - last_balance_check > 30 and not getattr(engine, '_shadow_simulated_balance', False):
                try:
                    from lib.engine.exchange import fetch_balance
                    rb = fetch_balance()
                    if rb >= 0:
                        engine.balance = rb
                        emit("balance_update", {"balance": rb})
                except Exception:
                    pass
                last_balance_check = time.time()

        elif engine.paused:
            pass  # wait for resume

        elif not engine.running:
            # Engine stopped — exit if idle too long (no start command received)
            idle_count += 1
            if idle_count > 30:  # 30 * 2s = 60s timeout
                emit("status", {"state": "stopped", "msg": "No start command received within 60s, exiting"})
                break

        # Emit status heartbeat
        emit("status", engine.get_status())
        time.sleep(2)

    # Clean exit
    emit("status", {"state": "stopped", "msg": "Engine process exiting"})


if __name__ == "__main__":
    main()
