"""
Flask Web API for Multi-Agent Data Query System
提供RESTful API接口供前端调用，支持普通查询和流式SSE查询。
"""

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from urllib.parse import unquote

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# 将当前目录添加到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from env_loader import load_env_file

load_env_file()

from agent import MultiAgentSystem

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # 允许跨域请求

# 全局系统实例（用于存储不同用户的会话）
user_systems: Dict[str, MultiAgentSystem] = {}

# 会话日志（JSONL）：每行一条完整对话记录，便于后续检索和分析
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "conversation_logs.jsonl"
_log_lock = threading.Lock()


class RequestLogMiddleware:
    """记录所有 WSGI 请求，包括静态文件和流式响应。"""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        start_time = time.perf_counter()
        method = environ.get("REQUEST_METHOD", "-")
        raw_path = environ.get("PATH_INFO") or "/"
        path = unquote(raw_path)
        query = environ.get("QUERY_STRING", "")
        display_path = f"{path}?{query}" if query else path
        remote_addr = environ.get("HTTP_X_FORWARDED_FOR") or environ.get("REMOTE_ADDR", "-")
        status_holder = {"status": "000"}
        logged = {"done": False}

        logger.info(f"{method} {display_path} - 请求开始 - 客户端: {remote_addr}")

        def logging_start_response(status, headers, exc_info=None):
            status_holder["status"] = status.split(" ", 1)[0]
            return start_response(status, headers, exc_info)

        def write_log(error=None):
            if logged["done"]:
                return
            logged["done"] = True
            duration_ms = (time.perf_counter() - start_time) * 1000
            status_code = status_holder["status"]
            message = (
                f"{method} {display_path} - 状态码: {status_code} - "
                f"耗时: {duration_ms:.2f}ms - 客户端: {remote_addr}"
            )
            if error is not None:
                logger.exception(f"{message} - 请求异常")
            elif str(status_code).startswith(("4", "5")):
                logger.warning(message)
            else:
                logger.info(message)

        try:
            iterable = self.wsgi_app(environ, logging_start_response)
        except Exception as e:
            write_log(e)
            raise

        try:
            for chunk in iterable:
                yield chunk
        except Exception as e:
            write_log(e)
            raise
        finally:
            close = getattr(iterable, "close", None)
            if close is not None:
                close()
            write_log()


app.wsgi_app = RequestLogMiddleware(app.wsgi_app)


def _append_chat_log(record: Dict[str, Any]) -> None:
    """追加一条会话日志到 JSONL 文件。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _parse_sse_event(event_text: str) -> Dict[str, Any] | None:
    """从 SSE 文本中解析 data JSON。"""
    if not isinstance(event_text, str):
        return None

    for line in event_text.splitlines():
        if line.startswith("data: "):
            payload = line[6:].strip()
            try:
                return json.loads(payload)
            except Exception:
                return None
    return None


def get_or_create_system(user_id: str) -> MultiAgentSystem:
    """获取或创建用户的系统实例"""
    if user_id not in user_systems:
        system = MultiAgentSystem()
        system.login(user_id)
        user_systems[user_id] = system
    return user_systems[user_id]


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('static', 'index.html')


@app.route('/api/login', methods=['POST'])
def login():
    """用户登录接口"""
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        
        # 创建或获取用户系统
        system = get_or_create_system(user_id)

        # 直接从长期记忆数据库加载用户信息（无需等待对话总结）
        ltm = system.master_agent.long_term_memory
        profile = ltm.get_user_profile(user_id)
        preferences = ltm.get_all_preferences(user_id)
        knowledge = ltm.get_all_knowledge(user_id, limit=50)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'session_id': system.session_id,
            'message': f'欢迎 {user_id}！',
            'user_info': {
                'logged_in': True,
                'user_id': user_id,
                'session_id': system.session_id,
                'profile': profile,
                'preferences': preferences,
                'knowledge': knowledge
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/query', methods=['POST'])
def query():
    """查询接口"""
    start_time = time.perf_counter()
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            }), 400
        
        # 获取用户系统
        system = get_or_create_system(user_id)
        
        # 执行查询
        answer = system.query(question)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        _append_chat_log({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "endpoint": "/api/query",
            "success": True,
            "user_id": user_id,
            "session_id": system.session_id,
            "question": question,
            "answer": answer,
            "duration_ms": duration_ms,
        })
        
        return jsonify({
            'success': True,
            'answer': answer,
            'user_id': user_id,
            'session_id': system.session_id
        })
    except Exception as e:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        try:
            _append_chat_log({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "endpoint": "/api/query",
                "success": False,
                "error": str(e),
                "duration_ms": duration_ms,
            })
        except Exception:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/new_session', methods=['POST'])
def new_session():
    """创建新会话"""
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        
        system = get_or_create_system(user_id)
        system.new_session()
        
        return jsonify({
            'success': True,
            'session_id': system.session_id,
            'message': '已开始新会话'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/user_info', methods=['POST'])
def user_info():
    """获取用户信息"""
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        
        system = get_or_create_system(user_id)
        # 直接读取长期记忆，包括知识列表
        ltm = system.master_agent.long_term_memory
        info = system.get_user_info()
        info['knowledge'] = ltm.get_all_knowledge(user_id, limit=50)

        return jsonify({
            'success': True,
            'user_info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/query_stream', methods=['POST'])
def query_stream():
    """流式查询接口（Server-Sent Events）
    
    前端使用 fetch + ReadableStream 接收，实现逐字打字效果。
    事件类型：
      - status: 处理状态更新（如"正在查询数据库..."）
      - intent: 识别到的意图类型
      - sql: 生成的SQL语句（含重试次数）
      - sources: 联网搜索来源URL列表
      - chart: ECharts图表配置JSON
      - chunk: LLM输出的文字片段（流式）
      - error: 错误信息（非致命，继续处理）
      - done: 流结束标志（含完整answer）
    """
    start_time = time.perf_counter()
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'success': False, 'error': '问题不能为空'}), 400
        
        system = get_or_create_system(user_id)
        final_answer = ""
        stream_summary: Dict[str, Any] = {
            "intent": None,
            "status_updates": [],
            "sql": [],
            "errors": [],
            "sources": [],
            "has_chart": False,
        }
        
        def generate():
            nonlocal final_answer
            try:
                for event in system.stream_query(question):
                    parsed = _parse_sse_event(event)
                    if parsed:
                        evt_type = parsed.get("type")
                        if evt_type == "intent":
                            stream_summary["intent"] = parsed.get("intent")
                        elif evt_type == "status":
                            if len(stream_summary["status_updates"]) < 30:
                                stream_summary["status_updates"].append(parsed.get("message", ""))
                        elif evt_type == "sql":
                            if len(stream_summary["sql"]) < 5:
                                stream_summary["sql"].append({
                                    "statement": parsed.get("sql", ""),
                                    "retry_count": parsed.get("retry_count", 0),
                                })
                        elif evt_type == "sources":
                            src = parsed.get("sources") or []
                            if isinstance(src, list):
                                stream_summary["sources"].extend([s for s in src if isinstance(s, str)])
                        elif evt_type == "chart":
                            stream_summary["has_chart"] = True
                        elif evt_type == "chunk":
                            final_answer += str(parsed.get("content", ""))
                        elif evt_type == "done":
                            if not final_answer:
                                final_answer = str(parsed.get("answer", ""))
                        elif evt_type == "error":
                            stream_summary["errors"].append(str(parsed.get("message", "")))
                    yield event
            except Exception as e:
                stream_summary["errors"].append(str(e))
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'answer': f'系统错误: {str(e)}'})}\n\n"
            finally:
                # 去重 sources，保留顺序
                unique_sources = []
                seen = set()
                for src in stream_summary["sources"]:
                    if src not in seen:
                        seen.add(src)
                        unique_sources.append(src)
                stream_summary["sources"] = unique_sources

                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                _append_chat_log({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "endpoint": "/api/query_stream",
                    "success": len(stream_summary["errors"]) == 0,
                    "user_id": user_id,
                    "session_id": system.session_id,
                    "question": question,
                    "answer": final_answer,
                    "duration_ms": duration_ms,
                    "stream": stream_summary,
                })
        
        return Response(
            stream_with_context(generate()),
            content_type='text/event-stream; charset=utf-8',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """动态切换模型接口"""
    try:
        data = request.json
        user_id = data.get('user_id', 'guest')
        model = data.get('model')
        base_url = data.get('base_url')
        api_key = data.get('api_key')
        
        system = get_or_create_system(user_id)
        result = system.switch_model(model=model, base_url=base_url, api_key=api_key)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': '模型切换失败'
        }), 500


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """获取当前模型信息"""
    try:
        user_id = request.args.get('user_id', 'guest')
        
        system = get_or_create_system(user_id)
        model_info = system.get_current_model()
        
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查接口"""
    search_available = False
    try:
        if user_systems:
            first_system = next(iter(user_systems.values()))
            search_available = first_system.master_agent.search_agent.available
    except Exception:
        pass
    
    return jsonify({
        'status': 'healthy',
        'active_users': len(user_systems),
        'features': {
            'sql_self_correction': True,
            'streaming': True,
            'web_search': search_available,
            'data_visualization': True
        }
    })


if __name__ == '__main__':
    import subprocess
    import socket

    def check_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def check_milvus_running():
        return check_port_in_use(19530)

    def start_milvus_server():
        milvus_data_dir = Path(os.getenv("MILVUS_DATA_DIR", Path(__file__).parent / "milvus_data"))
        milvus_data_dir.mkdir(parents=True, exist_ok=True)
        
        if check_milvus_running():
            logger.info("Milvus 服务已在运行")
            return True
        
        logger.info("正在启动 Milvus 服务...")
        try:
            import shutil
            milvus_cmd = shutil.which("milvus-server")
            if not milvus_cmd:
                milvus_cmd = "milvus-server"
            
            logger.info(f"使用命令: {milvus_cmd}")
            
            proc = subprocess.Popen(
                [milvus_cmd, "--data", str(milvus_data_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            import time
            for i in range(30):
                if check_milvus_running():
                    logger.info("Milvus 服务启动成功")
                    return True
                time.sleep(1)
                if i % 5 == 0:
                    logger.info(f"等待 Milvus 启动... ({i+1}s)")
            
            logger.warning("Milvus 服务启动超时")
            return False
        except Exception as e:
            logger.warning(f"启动 Milvus 服务失败: {e}")
            return False

    def init_db_and_check():
        db_path = Path(__file__).parent / "data" / "company.db"
        if not db_path.exists():
            logger.info("检测到数据库文件不存在，正在初始化...")
            try:
                from data.init_db import init_database
                init_database()
                logger.info("数据库初始化完成")
            except Exception as e:
                logger.exception(f"数据库初始化失败: {e}")
        else:
            logger.info(f"数据库已存在: {db_path}")
        
        milvus_enabled = os.getenv("MILVUS_ENABLED", "false").lower() == "true"
        if milvus_enabled:
            if check_milvus_running():
                logger.info("Milvus 服务已在运行")
            else:
                logger.warning("Milvus 服务未运行，向量检索功能将不可用")
                logger.warning("如需启用向量检索，请先在另一个终端运行: milvus-server --data milvus_data")

    init_db_and_check()
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("错误：未设置 OPENAI_API_KEY（可放在 .env 中）")
        sys.exit(1)
    
    logger.info("多智能体数据查询系统 Web API 启动中...")
    logger.info("访问地址: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

