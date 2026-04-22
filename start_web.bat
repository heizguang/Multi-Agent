@echo off
chcp 65001
setlocal EnableDelayedExpansion
echo =====================================
echo  多智能体数据查询系统 - Web界面启动
echo =====================================
echo.

REM 优先使用项目虚拟环境 Python
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

REM 从 .env 加载环境变量（如果存在）
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /R /V /C:"^[ ]*#" /C:"^[ ]*$" ".env"`) do (
        if /I "%%A"=="OPENAI_API_KEY" set "OPENAI_API_KEY=%%B"
        if /I "%%A"=="OPENAI_BASE_URL" set "OPENAI_BASE_URL=%%B"
        if /I "%%A"=="TAVILY_API_KEY" set "TAVILY_API_KEY=%%B"
    )
)

REM 检查环境变量
if "%OPENAI_API_KEY%"=="" (
    echo [错误] 未设置 OPENAI_API_KEY（可写入 .env）
    echo.
    echo 请先在项目根目录创建 .env，示例：
    echo OPENAI_BASE_URL=https://www.v2code.cc
    echo OPENAI_API_KEY=your_openai_key
    echo.
    pause
    exit /b 1
)

echo 环境变量已准备
echo.

REM 检查数据库是否存在
if not exist "data\company.db" (
    echo [警告] 业务数据库不存在，正在初始化...
    cd data
    "%PYTHON_EXE%" init_db.py
    cd ..
    echo 业务数据库初始化完成
    echo.
)

if not exist "data\long_term_memory.db" (
    echo [警告] 记忆数据库不存在，正在初始化...
    cd data
    "%PYTHON_EXE%" init_memory_db.py
    cd ..
    echo 记忆数据库初始化完成
    echo.
)

echo 数据库检查完成
echo.

echo 正在启动Web服务器...
echo.
echo 访问地址: http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务器
echo =====================================
echo.

"%PYTHON_EXE%" app.py

pause

