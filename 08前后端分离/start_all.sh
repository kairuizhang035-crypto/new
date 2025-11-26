#!/usr/bin/env bash
set -euo pipefail

# 一键启动前后端服务（Flask + Vite）
# - 后端端口：5000（占用则跳过启动）
# - 前端端口：自动在 8080-8090 或 8000-8010 中择一可用端口
# - 日志：backend/server.log、frontend/server.log

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

log() { echo -e "${YELLOW}[dev]${NC} $*"; }
ok() { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${RED}[warn]${NC} $*"; }

is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -i :"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    ss -ltn | awk '{print $4}' | grep -q ":$port$" || false
  fi
}

pick_free_port() {
  local start="${1:-8080}"
  local end="${2:-8090}"
  for p in $(seq "$start" "$end"); do
    if ! is_port_in_use "$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

start_backend() {
  cd "$BASE_DIR/backend"
  log "后端：检查虚拟环境与依赖..."
  if [ ! -d "venv" ]; then
    log "后端：创建Python虚拟环境..."
    python3 -m venv venv
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
  log "后端：安装依赖（requirements.txt）..."
  pip install -r requirements.txt

  export FLASK_APP=app.py
  export FLASK_ENV=development
  export FLASK_DEBUG=1

  if is_port_in_use 5000; then
    local pid
    pid="$(lsof -ti :5000 || true)"
    ok "后端已在运行：http://localhost:5000 (PID: ${pid:-unknown})"
  else
    log "后端：启动 Flask (http://localhost:5000)..."
    nohup python app.py > "$BASE_DIR/backend/server.log" 2>&1 &
    ok "后端启动成功；日志：backend/server.log"
  fi
}

start_frontend() {
  cd "$BASE_DIR/frontend"
  log "前端：检查依赖..."
  if [ ! -d "node_modules" ]; then
    log "前端：安装Node.js依赖..."
    npm install --no-audit --no-fund
  fi

  # 选择前端端口
  local port
  if port="$(pick_free_port 8080 8090)"; then
    :
  else
    warn "8080-8090端口均占用，尝试8000-8010..."
    port="$(pick_free_port 8000 8010 || echo 8000)"
  fi

  if is_port_in_use "$port"; then
    warn "选定端口 ${port} 已占用，改用 3000。"
    port="3000"
  fi

  log "前端：启动 Vite 开发服务器 (http://localhost:${port})..."
  nohup npm run dev -- --port "${port}" > "$BASE_DIR/frontend/server.log" 2>&1 &
  ok "前端启动成功；地址：http://localhost:${port}；日志：frontend/server.log"
}

log "一键启动：后端+前端"
start_backend
start_frontend
ok "完成。前端通过 /api 代理到后端 5000。按 Ctrl+C 退出当前脚本（服务继续运行）。"