#!/usr/bin/env python3
"""一键部署ATradeBot到VPS v3"""
import io, os, sys, time, tarfile, paramiko
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HOST, PASS, USER, PORT = "134.195.211.142", "22eP5w7TTA", "root", 22
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_DIR = "/root/atradebot"

exclude = {"node_modules", ".next", ".git", "__pycache__", ".claude", ".vscode"}

def main():
    print(f"连接 {HOST}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, PORT, USER, PASS, look_for_keys=False, allow_agent=False, timeout=15)
    print("✅ 连接成功\n")

    # 1. 打包 & 上传
    print("[1] 打包上传项目...")
    tar_path = os.path.join(LOCAL_DIR, "atradebot_deploy.tar.gz")
    os.chdir(LOCAL_DIR)
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in os.listdir(LOCAL_DIR):
            if item in exclude: continue
            tar.add(item)

    sftp = ssh.open_sftp()
    sftp.put(tar_path, REMOTE_DIR + "/project.tar.gz")
    sftp.close()
    print(f"  -> {os.path.getsize(tar_path)//1024}KB 上传完成")

    # 2. 解压
    print("\n[2] VPS安装...")
    _, stdout, _ = ssh.exec_command(
        f"cd {REMOTE_DIR} && tar xzf project.tar.gz --no-overwrite-dir 2>/dev/null; "
        f"rm -f project.tar.gz; "
        f"python3 -m venv .venv 2>/dev/null; "
        f"source .venv/bin/activate; "
        f"pip install --quiet -r lib/engine/requirements.txt; "
        f"pip install --quiet xgboost scikit-learn; "
        f"npm install --omit=dev; "
        f"npm run build; "
        f"npm install -g pm2; "
        f"mkdir -p logs; "
        f"pm2 delete atradebot 2>/dev/null; "
        f"cat > ecosystem.config.cjs << 'EOF'\n"
        f"module.exports={{apps:[{{name:'atradebot',script:'node_modules/next/dist/bin/next',args:'start -p 3000',cwd:'{REMOTE_DIR}',env:{{NODE_ENV:'production',PORT:'3000'}},instances:1,exec_mode:'fork',max_restarts:10}}]}};\n"
        f"EOF\n"
        f"pm2 start ecosystem.config.cjs && "
        f"nohup .venv/bin/python scripts/simulate_data.py --csv ./hibt_ticks.csv > logs/simulator.log 2>&1 &",
        timeout=600
    )
    code = stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    print(f"  exit: {code}")
    if out: print(f"  {out[:500]}")

    print("\n[3] 验证...")
    time.sleep(10)
    _, stdout, _ = ssh.exec_command("curl -s http://localhost:3000/api/health", timeout=10)
    out = stdout.read().decode().strip()
    if "balance" in out:
        print(f"\n✅ 部署成功！http://{HOST}:3000")
    else:
        print(f"\n⚠️ {out[:200]}")

    ssh.close()


if __name__ == "__main__":
    main()