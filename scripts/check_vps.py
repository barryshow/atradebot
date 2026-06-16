# -*- coding: utf-8 -*-
"""SSH到VPS查引擎日志"""
import paramiko, traceback

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('134.195.211.142', 22, 'root', '22eP5w7TTA')

# 查引擎日志最后200行
stdin, stdout, stderr = client.exec_command('cat /root/atradebot/atradebot-engine.log 2>/dev/null | tail -200')
print('=== ENGINE LOG (last 200 lines) ===')
for line in stdout:
    print(line, end='')

err = stderr.read().decode().strip()
if err:
    print('STDERR:', err)

# 查PM2进程
stdin2, stdout2, stderr2 = client.exec_command('pm2 list')
print()
print('=== PM2 LIST ===')
for line in stdout2:
    print(line, end='')

# 查最近30分钟日志
stdin3, stdout3, stderr3 = client.exec_command('pm2 logs --lines 100 --nostream 2>&1 | tail -100')
print()
print('=== PM2 LOGS (last 100 lines) ===')
for line in stdout3:
    print(line, end='')

# 查数据文件是否有效
stdin4, stdout4, stderr4 = client.exec_command('wc -l /root/atradebot/hibt_ticks.csv 2>/dev/null; echo "---"; tail -5 /root/atradebot/hibt_ticks.csv 2>/dev/null')
print()
print('=== CSV FILE ===')
for line in stdout4:
    print(line, end='')

# 查 .env.local 是否存在
stdin5, stdout5, stderr5 = client.exec_command('ls -la /root/atradebot/.env.local 2>/dev/null && echo "EXISTS" || echo "NOT EXISTS"')
print()
for line in stdout5:
    print(line, end='')

client.close()
