# -*- coding: utf-8 -*-
"""
Entry point: spawned by Next.js ProcessManager.
Communicates via stdout JSON lines, reads commands from stdin.
"""
import sys
import io
import json
import time
import threading
import os
import warnings

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


def stdin_reader(engine: TradingEngine):
    """Read commands from stdin (sent by Node.js ProcessManager)."""
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


def main():
    engine = TradingEngine()

    # Start stdin command reader in background
    reader = threading.Thread(target=stdin_reader, args=(engine,), daemon=True)
    reader.start()

    # Auto-start immediately
    engine.start()
    emit("status", {"state": "running", "msg": "Engine auto-started"})

    # Main loop
    last_balance_check = time.time()
    while True:
        if engine.running and not engine.paused:
            try:
                engine.tick()
            except Exception as e:
                emit("error", {"msg": f"Main loop error: {str(e)[:200]}"})

            # Periodic balance refresh
            if time.time() - last_balance_check > 30:
                try:
                    from lib.engine.exchange import fetch_balance
                    rb = fetch_balance()
                    if rb >= 0:
                        engine.balance = rb
                        emit("balance_update", {"balance": rb})
                except Exception:
                    pass
                last_balance_check = time.time()

        # Emit status heartbeat
        emit("status", engine.get_status())
        time.sleep(2)

        # Check if stdin thread is still alive
        if not reader.is_alive():
            break


if __name__ == "__main__":
    main()
