"""
使用 vLLM 部署 NL2SQL 模型（带进程管理）

改进点：
- 跨会话可靠 start/stop/status（PID 文件）
- 启动就绪检测 + 进程提前退出检测
- 可配置 max-model-len，默认 4096
"""

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

# 配置（可通过环境变量覆盖）
MODEL_PATH = os.environ.get("VLLM_MODEL_PATH", "/root/autodl-tmp/models/qwen3.5-4b")
VLLM_BASE_MODEL_PATH = os.environ.get("VLLM_BASE_MODEL_PATH", "/root/autodl-tmp/models/qwen3.5-4b")
VLLM_LORA_PATH = os.environ.get("VLLM_LORA_PATH", "/root/autodl-tmp/models/nl2sql-qwen3.5-4b/final")
VLLM_LORA_NAME = os.environ.get("VLLM_LORA_NAME", "nl2sql")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "6006"))
VLLM_HOST = os.environ.get("VLLM_HOST", "0.0.0.0")
VLLM_DTYPE = os.environ.get("VLLM_DTYPE", "half")
VLLM_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
VLLM_GPU_MEMORY_UTILIZATION = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.8")
VLLM_TENSOR_PARALLEL_SIZE = os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")
VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "600"))
VLLM_HEALTH_CHECK_INTERVAL = float(os.environ.get("VLLM_HEALTH_CHECK_INTERVAL", "2"))
VLLM_TRUST_REMOTE_CODE = os.environ.get("VLLM_TRUST_REMOTE_CODE", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

STATE_DIR = Path(os.environ.get("VLLM_STATE_DIR", "./data"))
PID_FILE = STATE_DIR / "vllm.pid"
LOG_FILE = STATE_DIR / "vllm.log"


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _health_url() -> str:
    return f"http://127.0.0.1:{VLLM_PORT}/v1/models"


def _using_lora() -> bool:
    return bool(VLLM_BASE_MODEL_PATH and VLLM_LORA_PATH)


def _serve_model_path() -> str:
    if _using_lora():
        return VLLM_BASE_MODEL_PATH
    return MODEL_PATH


def _request_model_name() -> str:
    # 启用 LoRA 时，请求侧 model 使用 LoRA 名称；
    # 否则保持与 serve 的模型路径一致。
    if _using_lora():
        return VLLM_LORA_NAME
    return MODEL_PATH


def _read_pid() -> Optional[int]:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    _ensure_state_dir()
    PID_FILE.write_text(str(pid), encoding="utf-8")


def _clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink(missing_ok=True)


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_service_ready() -> bool:
    try:
        response = requests.get(_health_url(), timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def _build_cmd() -> list[str]:
    cmd = [
        "vllm",
        "serve",
        _serve_model_path(),
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--dtype",
        VLLM_DTYPE,
        "--max-model-len",
        str(VLLM_MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(VLLM_TENSOR_PARALLEL_SIZE),
        "--gpu-memory-utilization",
        str(VLLM_GPU_MEMORY_UTILIZATION),
    ]
    if VLLM_TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")
    if _using_lora():
        cmd.extend(
            [
                "--enable-lora",
                "--lora-modules",
                f"{VLLM_LORA_NAME}={VLLM_LORA_PATH}",
            ]
        )
    return cmd


def start_vllm_server(force_restart: bool = False) -> bool:
    """启动 vLLM 服务。"""
    existing_pid = _read_pid()
    if existing_pid and _is_process_alive(existing_pid):
        if _is_service_ready():
            print(f"[信息] vLLM 已在运行，PID: {existing_pid}")
            print(f"[信息] 服务地址: http://{VLLM_HOST}:{VLLM_PORT}")
            if not force_restart:
                return True
            print("[信息] 将执行重启...")
            stop_vllm_server()
        elif not force_restart:
            print(f"[警告] 检测到旧 PID 存活但服务不可用: {existing_pid}")
            print("[提示] 可执行 --mode stop 或 --force-restart")
            return False

    if _is_service_ready() and not force_restart:
        print("[警告] 端口服务已可用，但无 PID 记录。")
        print("[提示] 可能是外部启动的 vLLM，请先手动停止后再用本脚本管理。")
        return True

    print(f"[信息] 启动 vLLM 服务: {_serve_model_path()}")
    if _using_lora():
        print(f"[信息] LoRA 适配器: {VLLM_LORA_NAME} -> {VLLM_LORA_PATH}")
    else:
        print("[信息] LoRA 适配器: 未启用")
    print(f"[信息] 监听地址: {VLLM_HOST}:{VLLM_PORT}")
    print(f"[信息] max-model-len: {VLLM_MAX_MODEL_LEN}")
    print(f"[信息] 日志文件: {LOG_FILE}")
    cmd = _build_cmd()

    _ensure_state_dir()
    with LOG_FILE.open("a", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    _write_pid(process.pid)
    print(f"[信息] vLLM 进程已启动 PID: {process.pid}")
    print(f"[信息] 等待服务就绪（超时: {VLLM_STARTUP_TIMEOUT}s）...")

    deadline = time.time() + VLLM_STARTUP_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            print(f"[错误] vLLM 进程已退出，退出码: {process.returncode}")
            print(f"[提示] 查看日志: {LOG_FILE}")
            _clear_pid()
            return False
        if _is_service_ready():
            print("[成功] vLLM 服务已就绪!")
            return True
        time.sleep(VLLM_HEALTH_CHECK_INTERVAL)

    print("[错误] vLLM 服务启动超时")
    print(f"[提示] 查看日志: {LOG_FILE}")
    return False


def stop_vllm_server() -> bool:
    """停止 vLLM 服务。"""
    pid = _read_pid()
    if not pid:
        print("[信息] 未找到 PID 记录，无需停止。")
        return True
    if not _is_process_alive(pid):
        print(f"[信息] PID 文件存在但进程已退出: {pid}")
        _clear_pid()
        return True

    print(f"[信息] 停止 vLLM 服务，PID: {pid} ...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(f"[错误] 停止失败: {e}")
        return False

    deadline = time.time() + 20
    while time.time() < deadline:
        if not _is_process_alive(pid):
            _clear_pid()
            print("[成功] vLLM 服务已停止")
            return True
        time.sleep(0.5)

    print("[警告] SIGTERM 超时，尝试 SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError as e:
        print(f"[错误] 强制停止失败: {e}")
        return False

    _clear_pid()
    print("[成功] vLLM 服务已强制停止")
    return True


def status_vllm_server() -> None:
    """查看服务状态。"""
    pid = _read_pid()
    alive = bool(pid and _is_process_alive(pid))
    ready = _is_service_ready()

    print("=" * 50)
    print("vLLM 服务状态")
    print("=" * 50)
    print(f"PID 文件: {PID_FILE}")
    print(f"日志文件: {LOG_FILE}")
    print(f"记录 PID: {pid if pid else '无'}")
    print(f"进程存活: {'是' if alive else '否'}")
    print(f"接口可用: {'是' if ready else '否'}")
    print(f"健康检查: {_health_url()}")
    print("=" * 50)


def test_vllm():
    """测试 vLLM 服务。"""
    if not _is_service_ready():
        print("[错误] vLLM 服务不可用，请先启动服务")
        return
    print("[信息] 健康检查通过，开始测试 SQL 生成")

    prompt = (
        "你是SQL专家。只输出一条SQLite SQL，不要解释，不要Markdown。\n"
        "问题：各部门分别有多少人？\nSQL:"
    )
    payload = {
        "model": _request_model_name(),
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0.0,
    }
    url = f"http://127.0.0.1:{VLLM_PORT}/v1/completions"
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"[错误] 测试请求失败: {response.status_code} {response.text}")
        return
    text = response.json().get("choices", [{}])[0].get("text", "")
    print(f"[成功] 模型返回: {text.strip()}")


def main():
    parser = argparse.ArgumentParser(description="vLLM 部署 NL2SQL（进程管理版）")
    parser.add_argument(
        "--mode",
        choices=["start", "stop", "status", "test"],
        default="start",
        help="模式: start/stop/status/test",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="start 模式下先停止再启动",
    )
    args = parser.parse_args()

    if args.mode == "start":
        ok = start_vllm_server(force_restart=args.force_restart)
        raise SystemExit(0 if ok else 1)
    if args.mode == "stop":
        ok = stop_vllm_server()
        raise SystemExit(0 if ok else 1)
    if args.mode == "status":
        status_vllm_server()
        raise SystemExit(0)
    if args.mode == "test":
        test_vllm()
        raise SystemExit(0)


if __name__ == "__main__":
    main()
