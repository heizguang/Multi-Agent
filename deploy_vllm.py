"""
使用 vLLM 部署 NL2SQL 模型
"""

import os
import subprocess
import time
import requests
import json

# 配置
MODEL_PATH = "/root/autodl-tmp/models/qwen3.5-4b"  # 模型路径
VLLM_PORT = 8000
VLLM_HOST = "0.0.0.0"
VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "600"))  # 秒
VLLM_HEALTH_CHECK_INTERVAL = float(os.environ.get("VLLM_HEALTH_CHECK_INTERVAL", "2"))  # 秒

# vLLM 服务进程
vllm_process = None


def install_vllm():
    """安装 vLLM"""
    print("[信息] 安装 vLLM...")
    subprocess.run([
        "pip", "install", "vllm>=0.4.0"
    ], check=True)
    print("[成功] vLLM 安装完成")


def start_vllm_server():
    """启动 vLLM 服务"""
    global vllm_process
    
    print(f"[信息] 启动 vLLM 服务: {MODEL_PATH}")
    print(f"[信息] 监听地址: {VLLM_HOST}:{VLLM_PORT}")
    
    # 构建命令
    cmd = [
        "vllm", "serve",
        MODEL_PATH,
        "--dtype", "half",
        "--max-model-len", "512",
        "--tensor-parallel-size", "1",
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--gpu-memory-utilization", "0.9"
    ]
    
    # 启动进程
    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print(f"[信息] vLLM 进程已启动 PID: {vllm_process.pid}")
    print(f"[信息] 等待服务就绪（超时: {VLLM_STARTUP_TIMEOUT}s）...")
    
    # 等待服务就绪
    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        # 子进程如果提前退出，直接失败返回，避免“假超时”
        if vllm_process.poll() is not None:
            print(f"[错误] vLLM 进程已退出，退出码: {vllm_process.returncode}")
            return False
        try:
            response = requests.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2)
            if response.status_code == 200:
                print("[成功] vLLM 服务已就绪!")
                return True
        except Exception:
            pass
        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)
    
    print("[错误] vLLM 服务启动超时")
    return False


def stop_vllm_server():
    """停止 vLLM 服务"""
    global vllm_process
    
    if vllm_process:
        print("[信息] 停止 vLLM 服务...")
        vllm_process.terminate()
        vllm_process.wait()
        print("[成功] vLLM 服务已停止")


def generate_sql(question: str, schema: str = "") -> str:
    """使用 vLLM 生成 SQL
    
    Args:
        question: 用户问题
        schema: 数据库结构
    
    Returns:
        生成的 SQL
    """
    # 构建 prompt
    if schema:
        prompt = f"根据用户问题生成SQL查询。只返回SQL语句。\n\n数据库结构：\n{schema}\n\n问题：{question}\nSQL："
    else:
        prompt = f"根据用户问题生成SQL查询。只返回SQL语句。\n\n问题：{question}\nSQL："
    
    url = f"http://localhost:{VLLM_PORT}/v1/completions"
    
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.1,
        "stop": ["<|im_end|>", "\n\n"]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["text"]
            
            # 提取 SQL
            sql = extract_sql(text)
            return sql
        else:
            print(f"[错误] vLLM 返回: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"[错误] 请求失败: {e}")
        return ""


def extract_sql(text: str) -> str:
    """从输出中提取 SQL"""
    import re
    
    # 查找 SQL 部分
    match = re.search(r'SQL[：:]\s*(.+?)(?:\n\n|$)', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        sql = text.strip()
    
    # 清理
    sql = sql.strip().rstrip(';').strip()
    return sql


def test_vllm():
    """测试 vLLM 服务"""
    print("\n" + "=" * 50)
    print("测试 vLLM SQL 生成")
    print("=" * 50)
    
    # 测试问题
    test_questions = [
        "公司一共有多少名员工？",
        "各部门分别有多少人？",
        "研发部平均工资是多少？",
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        sql = generate_sql(question)
        print(f"SQL: {sql}")
    
    print("\n" + "=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='vLLM 部署 NL2SQL')
    parser.add_argument('--mode', choices=['start', 'stop', 'test'], default='start',
                       help='模式: start=启动服务, stop=停止服务, test=测试')
    args = parser.parse_args()
    
    if args.mode == 'start':
        start_vllm_server()
        
    elif args.mode == 'stop':
        stop_vllm_server()
        
    elif args.mode == 'test':
        test_vllm()


if __name__ == "__main__":
    main()
