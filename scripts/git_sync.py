#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键同步：git push → VPS git pull → rebuild → restart
用法:  python scripts/git_sync.py

流程:
  1. git add / commit / push (自动检测是否有未提交修改)
  2. SSH 到 VPS → git pull → npm install (增量) → npm run build → pm2 restart

安全: VPS 密码从 scripts/vps_config.py 集中读取
"""
import io, sys, os, subprocess, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 从集中配置文件读取 VPS 凭据
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vps_config import HOST, USER, PASS, PORT, REMOTE_DIR


def local_run(cmd, capture=False):
    print(f"\n[本地] $ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if r.returncode != 0 and not capture:
        print(f"  -> exit {r.returncode}")
        if r.stderr:
            print(f"  STDERR: {r.stderr[:500]}")
    return r


def vps_run(ssh, cmd, timeout=120):
    print(f"\n[VPS] $ {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    ec = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if out:
        print(out)
    if err and ec != 0:
        print(f"  STDERR: {err[:500]}")
    if ec != 0:
        print(f"  -> exit {ec}")
    return out, err, ec


def main():
    # ── Step 1: 本地 git commit + push ──
    print("=" * 60)
    print("STEP 1/2: 本地提交并推送到 GitHub")
    print("=" * 60)

    # 检查是否有未提交修改
    r = local_run("git status --porcelain", capture=True)
    if r.stdout.strip():
        print("检测到未提交的修改：")
        print(r.stdout)
        # add all
        local_run("git add -A")
        # commit with timestamp
        ts = time.strftime("%Y-%m-%d %H:%M")
        local_run(f'git commit -m "sync: auto commit {ts}"')
    else:
        print("没有新的本地修改")
        # 检查是否有未推送的 commit
        r = local_run("git log origin/main..HEAD --oneline", capture=True)
        if not r.stdout.strip():
            print("本地和远程一致，无需推送")

    # push
    local_run("git push origin main")

    # ── Step 2: SSH 到 VPS 同步 ──
    print("\n" + "=" * 60)
    print("STEP 2/2: 连接 VPS 并部署")
    print("=" * 60)

    try:
        import paramiko
    except ImportError:
        print("\n⚠ paramiko 未安装，请安装：pip install paramiko")
        print("或手动在 VPS 上执行：")
        print(f"  cd {REMOTE_DIR} && git pull && npm install && npm run build")
        print("然后 pm2 restart atradebot")
        sys.exit(1)

    print(f"连接 {HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, look_for_keys=False, allow_agent=False, timeout=15)
    print("已连接！")

    # 2a. git pull
    out, _, ec = vps_run(ssh, f"cd {REMOTE_DIR} && git fetch origin && git reset --hard origin/main", timeout=60)
    vps_run(ssh, f"cd {REMOTE_DIR} && git log -1 --oneline", timeout=10)

    # 2b. npm install (增量，安全)
    vps_run(ssh, f"cd {REMOTE_DIR} && npm install --no-audit --no-fund 2>&1 | tail -10", timeout=180)

    # 2c. npm run build
    print("\n--- npm run build ---")
    out, _, ec = vps_run(ssh, f"cd {REMOTE_DIR} && npm run build 2>&1", timeout=300)
    if ec != 0:
        print("\n❌ 构建失败！最后 30 行：")
        for l in (out or "").split("\n")[-30:]:
            print(l)
        ssh.close()
        sys.exit(1)
    print("✅ 构建成功！")

    # 2d. pm2 restart
    vps_run(ssh, "pm2 restart atradebot --update-env", timeout=15)
    vps_run(ssh, "pm2 save", timeout=10)
    time.sleep(2)
    vps_run(ssh, "pm2 status atradebot", timeout=10)

    # 2e. 重启 Python 引擎
    print("\n--- 重启 Python 引擎 ---")
    vps_run(ssh, "pkill -f 'lib/engine/main.py' 2>/dev/null; sleep 2", timeout=10)
    vps_run(ssh,
        f"cd {REMOTE_DIR} && mkdir -p logs && nohup python3 -u lib/engine/main.py > logs/engine.log 2>&1 &",
        timeout=5)
    time.sleep(5)
    out, _, _ = vps_run(ssh, "pgrep -af 'lib/engine/main.py' || echo '⚠ 引擎未运行'", timeout=5)
    if "main.py" in out:
        print("✅ Python 引擎已启动")
    else:
        print("⚠ Python 引擎可能未启动，请检查 VPS 日志")

    # 2f. 健康检查
    out, _, _ = vps_run(ssh, "curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/api/health 2>&1", timeout=10)
    print(f"\nHealth endpoint: {out}")
    if "200" in out:
        print(f"\n{'=' * 60}")
        print(f"✅ 部署成功！http://{HOST}:3000")
        print(f"{'=' * 60}")
    else:
        print(f"⚠ 健康检查返回: {out}")
        print(f"  手动检查: http://{HOST}:3000")

    ssh.close()


if __name__ == "__main__":
    main()
