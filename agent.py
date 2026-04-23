"""
LangGraph智能问答Agent - 多智能体版本

基于LangGraph工作流的自然语言数据库查询和分析系统。
支持一主两从的多智能体架构：
- 主智能体：MasterAgent（意图识别、路由、汇总）
- 子智能体1：SQLQueryAgent（数据库查询）
- 子智能体2：DataAnalysisAgent（数据分析）

支持长短期记忆：
- 短期记忆：MemorySaver（会话内对话历史）
- 长期记忆：LongTermMemory（跨会话用户偏好和知识）
"""

import os
import json
import threading
import uuid
from typing import Dict, Any
from pathlib import Path

from langchain_core.language_models import BaseLLM
import yaml

from env_loader import load_env_file
from llm_client import OpenAICompatRequestsLLM

load_env_file()

from agents import MasterAgent

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


class MultiAgentSystem:
    """多智能体系统 - 主入口类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化多智能体系统
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.llm = self._init_llm()
        self.db_path = self.config["database"]["path"]
        self.runtime_state_path = Path("./data/user_runtime_state.json")
        self._runtime_lock = threading.Lock()
        self._runtime_state = self._load_runtime_state()
        
        # 记忆配置
        memory_config = self.config.get("memory", {})
        memory_db_path = memory_config.get("long_term_db", "./data/long_term_memory.db")
        short_term_max_tokens = memory_config.get("short_term_max_tokens", 1000)
        
        # 联网搜索配置
        search_config = self.config.get("search", {})
        tavily_api_key = search_config.get("tavily_api_key", "")
        
        # 初始化主智能体（内部会初始化三个子智能体：SQL、Analysis、Search）
        self.master_agent = MasterAgent(
            llm=self.llm,
            db_path=self.db_path,
            num_examples=self.config["nl2sql"]["num_examples"],
            memory_db_path=memory_db_path,
            short_term_max_tokens=short_term_max_tokens,
            tavily_api_key=tavily_api_key
        )
        
        # 用户登录状态
        self.user_id = None  # 当前登录用户
        self.session_id = None  # 当前会话ID

    def _load_runtime_state(self) -> Dict[str, Any]:
        if not self.runtime_state_path.exists():
            return {"users": {}}
        try:
            with self.runtime_state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                users = data.get("users")
                if isinstance(users, dict):
                    return {"users": users}
        except Exception as e:
            console.print(f"[yellow]加载运行时状态失败，将使用空状态: {e}[/yellow]")
        return {"users": {}}

    def _save_runtime_state(self) -> None:
        self.runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.runtime_state_path.with_suffix(self.runtime_state_path.suffix + ".tmp")
        with self._runtime_lock:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(self._runtime_state, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.runtime_state_path)

    def _restore_session_id(self, user_id: str) -> str:
        users = self._runtime_state.setdefault("users", {})
        user_state = users.get(user_id, {})
        session_id = user_state.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return session_id
        return str(uuid.uuid4())

    def _persist_user_runtime_state(self, user_id: str) -> None:
        users = self._runtime_state.setdefault("users", {})
        users[user_id] = {
            "session_id": self.session_id,
        }
        self._save_runtime_state()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 替换环境变量
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                return os.getenv(obj[2:-1], obj)
            return obj
        
        return replace_env_vars(config)
    
    def _init_llm(self):
        """初始化语言模型
        
        使用 OpenAI 兼容接口连接 DashScope，支持所有通义千问模型
        （qwen-turbo-latest / qwen-plus-latest / qwen-max-latest / qwen3.5-plus 等）
        """
        llm_config = self.config["llm"]
        
        if llm_config["provider"] == "dashscope":
            base_url = llm_config.get(
                "base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            # OpenAI-compatible relay endpoints usually require /v1.
            if base_url and not base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            model = llm_config["model"]
            api_key = llm_config["api_key"]
            temperature = llm_config["temperature"]
            max_tokens = llm_config["max_tokens"]
            sdk_enabled = str(
                llm_config.get("sdk_enabled", os.getenv("OPENAI_SDK_ENABLED", "false"))
            ).lower() in {"1", "true", "yes", "on"}

            # 默认优先使用 requests 通道，避免在模块导入阶段卡在 langchain_openai/certifi。
            if not sdk_enabled:
                print("[LLM] 使用 requests 兼容通道启动（未启用 ChatOpenAI SDK）")
                return OpenAICompatRequestsLLM(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60,
                )

            # 只有显式开启时才尝试导入 SDK，避免启动阶段因证书或环境问题卡死。
            try:
                from langchain_openai import ChatOpenAI

                sdk_llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    use_responses_api=llm_config.get("use_responses_api", False),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True,
                )
                print("[LLM] ChatOpenAI 通道初始化完成")
                return sdk_llm
            except Exception as e:
                print(f"[LLM] ChatOpenAI 通道不可用，切换 requests 通道: {e}")
                return OpenAICompatRequestsLLM(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60,
                )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config['provider']}")
    
    def login(self, user_id: str) -> bool:
        """用户登录
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否登录成功
        """
        try:
            self.user_id = user_id
            restored_session_id = self._restore_session_id(user_id)
            self.session_id = str(uuid.uuid4())  # 生成新的会话ID
            
            # 更新长期记忆中的用户活跃时间
            self.session_id = restored_session_id
            self._persist_user_runtime_state(user_id)
            self.master_agent.long_term_memory.update_user_activity(user_id)
            
            return True
        except Exception as e:
            console.print(f"[red]登录失败: {e}[/red]")
            return False
    
    def query(self, question: str) -> str:
        """执行查询（阻塞式，返回完整回答）
        
        Args:
            question: 用户问题
        
        Returns:
            回答结果
        """
        if not self.user_id:
            return "请先登录。您可以输入任意用户ID开始使用。"
        
        thread_id = f"{self.user_id}_{self.session_id}"
        
        return self.master_agent.query(
            question, 
            thread_id=thread_id,
            user_id=self.user_id
        )
    
    def stream_query(self, question: str):
        """流式查询，返回 SSE 事件生成器
        
        Args:
            question: 用户问题
            
        Yields:
            SSE 格式字符串
        """
        if not self.user_id:
            import json
            yield f"data: {json.dumps({'type': 'chunk', 'content': '请先登录。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': '请先登录。'})}\n\n"
            return
        
        thread_id = f"{self.user_id}_{self.session_id}"
        
        yield from self.master_agent.stream_query(
            question,
            thread_id=thread_id,
            user_id=self.user_id
        )
    
    def set_thread_id(self, thread_id: str):
        """设置会话线程ID（保留兼容性）
        
        Args:
            thread_id: 线程ID
        """
        # 已废弃，现在使用 user_id + session_id 自动生成
        console.print("[yellow]提示：thread_id现在由user_id和session_id自动生成[/yellow]")
    
    def new_session(self):
        """开始新会话（保留当前用户）"""
        if self.user_id:
            self.session_id = str(uuid.uuid4())
            self._persist_user_runtime_state(self.user_id)
            console.print(f"[green]已开始新会话: {self.session_id[:8]}...[/green]")
        else:
            console.print("[yellow]请先登录[/yellow]")
    
    def get_user_info(self) -> Dict[str, Any]:
        """获取当前用户信息"""
        if not self.user_id:
            return {"logged_in": False}
        
        profile = self.master_agent.long_term_memory.get_user_profile(self.user_id)
        preferences = self.master_agent.long_term_memory.get_all_preferences(self.user_id)
        
        return {
            "logged_in": True,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "profile": profile,
            "preferences": preferences
        }
    
    def switch_model(self, model: str = None, base_url: str = None, api_key: str = None) -> Dict[str, Any]:
        """动态切换模型配置
        
        Args:
            model: 新模型名称（如 gpt-4、qwen-max 等）
            base_url: 新的API地址
            api_key: 新的API密钥
        
        Returns:
            切换结果信息
        """
        try:
            if model:
                self.config["llm"]["model"] = model
            if base_url:
                self.config["llm"]["base_url"] = base_url
            if api_key:
                self.config["llm"]["api_key"] = api_key
            
            new_llm = self._init_llm()
            old_model = self.llm.model if hasattr(self.llm, 'model') else "unknown"
            self.llm = new_llm
            
            new_model = self.config["llm"]["model"]
            self.master_agent.llm = new_llm
            
            self.master_agent.sql_agent.llm = new_llm
            self.master_agent.analysis_agent.llm = new_llm
            self.master_agent.search_agent.llm = new_llm
            
            return {
                "success": True,
                "old_model": old_model,
                "new_model": new_model,
                "message": f"模型切换成功: {old_model} -> {new_model}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "模型切换失败"
            }
    
    def get_current_model(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        model_name = self.config["llm"].get("model", "unknown")
        base_url = self.config["llm"].get("base_url", "")
        return {
            "model": model_name,
            "base_url": base_url,
            "provider": self.config["llm"].get("provider", "dashscope")
        }

SQLAgent = MultiAgentSystem

def main():
    console.print(Panel.fit(
        "[cyan]LangGraph 多智能体数据查询系统 v2.1[/cyan]\n"
        "主智能体 + SQL查询 + 数据分析 + Web前端\n"
        "智能路由 · 深度分析 · 长短期记忆",
        border_style="cyan"
    ))
    console.print()
    
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]错误：未设置 OPENAI_API_KEY（可放在 .env 中）[/red]")
        return
    
    # 初始化系统
    agent = MultiAgentSystem()
    
    # 用户登录
    console.print("[bold cyan]欢迎使用智能数据查询系统！[/bold cyan]")
    user_id = Prompt.ask("[cyan]请输入用户ID（用于保存您的偏好和记忆）[/cyan]", default="guest")
    
    if agent.login(user_id):
        console.print(f"[green]欢迎 {user_id}！系统已就绪[/green]")
        console.print(f"[dim]会话ID: {agent.session_id[:8]}...[/dim]\n")
    else:
        console.print("[red]登录失败，程序退出[/red]")
        return
    
    # 显示帮助信息
    console.print("[dim]特殊命令：")
    console.print("[dim]  - 输入 'new' 开始新会话（清空短期记忆）")
    console.print("[dim]  - 输入 'info' 查看用户信息")
    console.print("[dim]  - 输入 'exit' 或 'quit' 退出系统[/dim]\n")
    
    while True:
        question = Prompt.ask("[cyan]请输入问题[/cyan]")
        
        # 处理特殊命令
        if question.lower() in ['exit', 'quit', 'q']:
            console.print("\n[yellow]再见！您的偏好和记忆已保存。[/yellow]")
            break
        
        if question.lower() == 'new':
            agent.new_session()
            continue
        
        if question.lower() == 'info':
            user_info = agent.get_user_info()
            console.print(Panel(
                f"[cyan]用户信息[/cyan]\n"
                f"用户ID: {user_info.get('user_id')}\n"
                f"会话ID: {user_info.get('session_id', '')[:8]}...\n"
                f"偏好: {user_info.get('preferences', {})}",
                border_style="blue"
            ))
            continue
        
        if not question.strip():
            continue
        
        # 执行查询
        answer = agent.query(question)
        console.print(Panel(answer, title="回答", border_style="green"))
        console.print()


if __name__ == "__main__":
    main()
