#!/usr/bin/env bash
set -euo pipefail

# ======================================================
#  ATradeBot VPS 部署脚本
#  Ubuntu 22.04+ / Debian 11+ 一键部署
# ======================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "${GREEN}✅ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ---------- Configuration ----------
APP_DIR="${HOME}/atradebot"
NODE_VERSION="20"
PYTHON_VERSION="python3"

# Check for environment variables file
ENV_FILE="${APP_DIR}/.env.local"

# ---------- Step 1: System dependencies ----------
log "📦 安装系统依赖..."
sudo apt-get update -qq
sudo apt-get install -y -qq curl git build-essential ${PYTHON_VERSION} ${PYTHON_VERSION}-pip ${PYTHON_VERSION}-venv nginx 2>/dev/null
ok "系统依赖安装完成"

# ---------- Step 2: Node.js ----------
log "📦 安装 Node.js ${NODE_VERSION}..."
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | sudo -E bash - >/dev/null 2>&1
    sudo apt-get install -y -qq nodejs 2>/dev/null
fi
ok "Node.js $(node --version) / npm $(npm --version)"

# ---------- Step 3: Clone/Update project ----------
log "📂 部署项目代码..."
if [ -d "$APP_DIR" ]; then
    warn "项目目录已存在 ($APP_DIR)，跳过 git clone"
    warn "请手动执行 git pull 更新代码"
else
    # If you have a git repo, use it. Otherwise, the user needs to scp/rsync the project.
    warn "请将本地项目文件上传到 ${APP_DIR}"
    warn "例如: rsync -avz --exclude node_modules --exclude .next ./atradebot/ user@vps:${APP_DIR}/"
    mkdir -p "$APP_DIR"
fi

# ---------- Step 4: Python virtualenv & dependencies ----------
log "🐍 配置 Python 环境..."
VENV_DIR="${APP_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    ${PYTHON_VERSION} -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"
pip install --quiet --upgrade pip
if [ -f "${APP_DIR}/lib/engine/requirements.txt" ]; then
    pip install --quiet -r "${APP_DIR}/lib/engine/requirements.txt"
fi
pip install --quiet lightgbm pandas numpy curl_cffi joblib

ok "Python 环境就绪: $(${PYTHON_VERSION} --version)"

# ---------- Step 5: Node dependencies ----------
log "📦 安装 Node 依赖..."
cd "$APP_DIR"
npm install --omit=dev 2>/dev/null
ok "npm 依赖安装完成"

# ---------- Step 6: Build Next.js ----------
log "🔨 构建项目..."
if [ -f "package.json" ]; then
    npm run build 2>&1 | tail -5
    ok "Next.js 构建完成"
fi

# ---------- Step 7: Environment setup ----------
log "🔐 配置环境变量..."
if [ ! -f ".env.local" ]; then
    warn "请创建 .env.local 文件并填入以下内容:"
    echo
    echo "───────────────────────────────────────────────"
    echo "cat > ${APP_DIR}/.env.local << 'EOF'"
    echo "# HIBT API Token (必填)"
    echo "HIBT_TOKEN=你的HIBT_TOKEN"
    echo ""
    echo "# AI API (必填)"
    echo "AI_API_KEY=你的API_KEY"
    echo "AI_URL=https://api.siliconflow.cn/v1/chat/completions"
    echo "AI_MODEL=deepseek-ai/DeepSeek-V3"
    echo ""
    echo "# 飞书通知 (可选)"
    echo "FEISHU_WEBHOOK=你的飞书Webhook"
    echo ""
    echo "# 引擎配置 (可选)"
    echo "PYTHON_PATH=${APP_DIR}/.venv/bin/python3"
    echo "RADAR_CSV_PATH=${APP_DIR}/hibt_ticks.csv"
    echo "MODEL_DIR=${APP_DIR}/models"
    echo "FIXED_BET=3"
    echo "HOLD_MINUTES=5"
    echo "MAX_CONCURRENT_TRADES=3"
    echo "EOF"
    echo "───────────────────────────────────────────────"
    echo
else
    ok ".env.local 已存在"
fi

# ---------- Step 8: PM2 for process management ----------
log "⚙️  配置进程管理..."
if ! command -v pm2 &>/dev/null; then
    sudo npm install -g pm2 >/dev/null 2>&1
fi

# Create PM2 ecosystem file
cat > "${APP_DIR}/ecosystem.config.cjs" << 'PM2EOF'
module.exports = {
  apps: [{
    name: "atradebot",
    script: "node_modules/next/dist/bin/next",
    args: "start",
    cwd: __dirname,
    env: {
      NODE_ENV: "production",
      PORT: 3000,
    },
    env_file: ".env.local",
    instances: 1,
    exec_mode: "fork",
    max_restarts: 10,
    restart_delay: 5000,
    exp_backoff_restart_delay: 100,
    error_file: "logs/error.log",
    out_file: "logs/output.log",
    merge_logs: true,
    time: true,
  }]
};
PM2EOF

mkdir -p "${APP_DIR}/logs"

# Start with PM2
cd "$APP_DIR"
pm2 start ecosystem.config.cjs 2>/dev/null || pm2 restart atradebot 2>/dev/null || true
pm2 save

ok "PM2 进程管理配置完成"

# ---------- Step 9: Nginx reverse proxy (optional) ----------
log "🌐 配置 Nginx 反向代理 (可选)..."
NGINX_CONF="/etc/nginx/sites-available/atradebot"
if [ ! -f "$NGINX_CONF" ]; then
    # Get server IP
    SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "0.0.0.0")

    sudo tee "$NGINX_CONF" > /dev/null << NGINXEOF
server {
    listen 80;
    server_name _;

    client_max_body_size 10m;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    # Optional: stream Python engine logs
    location /api/engine/stream {
        proxy_pass http://127.0.0.1:3000/api/engine/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }
}
NGINXEOF

    sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t 2>/dev/null && sudo systemctl restart nginx && ok "Nginx 配置完成" || warn "Nginx 配置有误，请手动检查"
fi

# ---------- Step 10: Health check ----------
log "🏥 健康检查..."
sleep 3
if curl -s http://127.0.0.1:3000/api/health > /dev/null 2>&1; then
    ok "ATradeBot 运行正常!"
    SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
    echo
    echo "================================================"
    echo -e "  ${GREEN}🎉 部署完成!${NC}"
    echo
    echo "  访问地址: http://${SERVER_IP:-VPS_IP}"
    echo "  PM2 管理: pm2 status"
    echo "  查看日志: pm2 logs atradebot"
    echo "  重启:     pm2 restart atradebot"
    echo ""
    echo "  ⚠️  不要忘记:"
    echo "  1. 配置 .env.local (如果还没配置)"
    echo "  2. 上传模型文件到 models/ 目录"
    echo "================================================"
else
    warn "服务未正常启动，请检查日志: pm2 logs atradebot"
    echo "尝试手动启动: cd ${APP_DIR} && npm start"
fi
