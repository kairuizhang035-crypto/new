#!/bin/bash

# 启动Flask后端服务
echo "正在启动Flask后端服务..."

# 进入后端目录
cd "$(dirname "$0")/backend"

# 优先使用当前已激活的 Conda 环境
PY_BIN="${CONDA_PREFIX}/bin/python"
PIP_BIN="${CONDA_PREFIX}/bin/pip"

if [ -x "$PY_BIN" ] && [ -x "$PIP_BIN" ]; then
    echo "使用 Conda 环境: $CONDA_PREFIX"
else
    # 回退到项目内 venv
    if [ ! -d "venv" ]; then
        echo "创建Python虚拟环境..."
        python3 -m venv venv
    fi
    echo "激活虚拟环境..."
    source venv/bin/activate
    PY_BIN="python"
    PIP_BIN="pip"
fi

echo "安装Python依赖..."
"$PIP_BIN" install -r requirements.txt

# 设置环境变量
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# 启动Flask应用
echo "启动Flask应用 (http://localhost:5000)..."
"$PY_BIN" app.py
