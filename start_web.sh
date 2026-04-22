#!/bin/bash

echo "====================================="
echo " 多智能体数据查询系统 - Web界面启动"
echo "====================================="
echo

# 优先使用项目虚拟环境 Python
PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# 检查环境变量
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "[错误] 未设置 OPENAI_API_KEY（可写入 .env）"
    echo
    echo "请先在项目根目录创建 .env，示例："
    echo "OPENAI_BASE_URL=https://www.v2code.cc"
    echo "OPENAI_API_KEY=your_openai_key"
    echo
    exit 1
fi

echo "环境变量已设置"
echo

# 检查数据库是否存在
if [ ! -f "data/company.db" ]; then
    echo "[警告] 业务数据库不存在，正在初始化..."
    cd data
    "$PYTHON_BIN" init_db.py
    cd ..
    echo "业务数据库初始化完成"
    echo
fi

if [ ! -f "data/long_term_memory.db" ]; then
    echo "[警告] 记忆数据库不存在，正在初始化..."
    cd data
    "$PYTHON_BIN" init_memory_db.py
    cd ..
    echo "记忆数据库初始化完成"
    echo
fi

echo "数据库检查完成"
echo

echo "正在启动Web服务器..."
echo
echo "访问地址: http://localhost:5000"
echo
echo "按 Ctrl+C 停止服务器"
echo "====================================="
echo

"$PYTHON_BIN" app.py

