#!/bin/bash

# 启动Flask后端服务
echo "正在启动Flask后端服务..."

# 进入后端目录
cd "$(dirname "$0")/backend"

# 检查是否存在虚拟环境
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 设置环境变量
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# 启动Flask应用
echo "启动Flask应用 (http://localhost:5000)..."
python app.py