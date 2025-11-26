#!/bin/bash

# 启动Vue前端服务
echo "正在启动Vue前端服务..."

# 进入前端目录
cd "$(dirname "$0")/frontend"

# 检查是否已安装依赖
if [ ! -d "node_modules" ]; then
    echo "安装Node.js依赖..."
    npm install
fi

# 启动开发服务器
echo "启动Vue开发服务器 (http://localhost:8000)..."
npm run dev