#!/usr/bin/env python3
"""Deploy atradebot to VPS via git clone + npm build + pm2."""
import io, sys, time, paramiko

# Force UTF-8 for stdout/stderr (handles GBK issues on Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

HOST, PASS, USER, PORT = "134.195.211.142", "22eP5w7TTA", "root", 22
REPO = "https://github.com/barryshow/atradebot.git"
REMOTE_DIR = "/root/atradebot"


def run(ssh, cmd, timeout=120):
    """Run a command and return (stdout, stderr, exit_code)."""
    print(f"\n$ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        print(out.encode('utf-8', errors='replace').decode('utf-8'))
    if err and exit_status != 0:
        print(f"STDERR: {err.encode('utf-8', errors='replace').decode('utf-8')}", file=sys.stderr)
    if exit_status != 0:
        print(f"  -> exit code: {exit_status}")
    return out, err, exit_status


def main():
    print(f"Connecting to {HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, look_for_keys=False, allow_agent=False, timeout=15)
    print("Connected!\n")

    # 1. Clean up old deployment
    print("=" * 50)
    print("STEP 1/6: Cleaning up old deployment")
    print("=" * 50)
    run(ssh, "pm2 delete atradebot 2>/dev/null; pm2 save --force 2>/dev/null", timeout=10)
    run(ssh, "rm -rf /root/atradebot && rm -rf /root/atradebot_bak && sync", timeout=15)
    # Verify
    out, _, _ = run(ssh, "test -d /root/atradebot && echo EXISTS || echo CLEAN", timeout=5)

    # 2. Clone repository
    print("\n" + "=" * 50)
    print("STEP 2/6: Cloning repository from GitHub")
    print("=" * 50)
    run(ssh, f"cd /root && git clone {REPO}", timeout=60)

    # 3. Install npm dependencies
    print("\n" + "=" * 50)
    print("STEP 3/6: Installing npm dependencies")
    print("=" * 50)
    run(ssh, "cd /root/atradebot && npm install", timeout=180)

    # 4. Build Next.js app
    print("\n" + "=" * 50)
    print("STEP 4/6: Building Next.js app")
    print("=" * 50)
    out, err, ec = run(ssh, "cd /root/atradebot && npm run build 2>&1", timeout=300)
    if ec != 0:
        print("\n❌ BUILD FAILED!")
        lines = out.split('\n') if out else []
        for l in lines[-50:]:
            print(l)
        ssh.close()
        sys.exit(1)
    print("\nBuild successful!")

    # 5. Start with pm2
    print("\n" + "=" * 50)
    print("STEP 5/6: Starting with pm2")
    print("=" * 50)
    run(ssh, "cd /root/atradebot && npm install -g pm2 2>/dev/null", timeout=30)
    run(ssh, "cd /root/atradebot && pm2 start npm --name atradebot -- start", timeout=15)
    run(ssh, "pm2 save", timeout=10)
    run(ssh, "pm2 set pm2:autodump true 2>/dev/null", timeout=5)
    time.sleep(3)

    # 6. Verify
    print("\n" + "=" * 50)
    print("STEP 6/6: Verification")
    print("=" * 50)
    run(ssh, "pm2 status 2>&1", timeout=10)
    run(ssh, "ss -tlnp | grep 3000 || echo 'Port 3000 not found'", timeout=10)
    out, _, _ = run(ssh, "curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/api/health 2>&1 || echo 'not_ok'", timeout=10)

    print("\n" + "=" * 50)
    if "200" in out:
        print(f"✅ Deployment successful! http://{HOST}:3000")
    else:
        print(f"⚠️  App started but health endpoint returned: {out}")
        print(f"   Check manually: http://{HOST}:3000")
    print("=" * 50)

    ssh.close()


if __name__ == "__main__":
    main()