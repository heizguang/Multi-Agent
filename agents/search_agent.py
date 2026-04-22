"""
DeepSearch 联网搜索子智能体

基于 Tavily 提供联网搜索能力，支持：
1. 纯搜索问答
2. 搜索 + SQL 联合分析
3. 预取搜索上下文，便于与其他任务并行
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.language_models import BaseLLM

sys.path.append(str(Path(__file__).parent.parent))
from prompts import get_search_and_sql_prompt, get_search_synthesis_prompt


class WebSearchAgent:
    """联网搜索子智能体。"""

    def __init__(self, llm: BaseLLM, tavily_api_key: str = "", max_results: int = 5):
        self.llm = llm
        self.max_results = max_results
        self.available = False
        self.search_tool = None
        self._init_search_tool(tavily_api_key)

    def _init_search_tool(self, api_key: str):
        effective_key = api_key or os.getenv("TAVILY_API_KEY", "")

        if not effective_key or effective_key.startswith("${"):
            print("[DeepSearch] 未配置 TAVILY_API_KEY，联网搜索功能不可用。")
            print("[DeepSearch] 请在环境变量或 config.yaml 中补充配置。")
            print("[DeepSearch] 申请地址：https://tavily.com")
            return

        try:
            os.environ["TAVILY_API_KEY"] = effective_key
            from langchain_tavily import TavilySearch

            self.search_tool = TavilySearch(max_results=self.max_results)
            self.available = True
            print("[DeepSearch] Tavily 搜索工具初始化成功")
        except ImportError:
            print("[DeepSearch] 未安装 langchain-tavily，请执行 pip install langchain-tavily")
        except Exception as e:
            print(f"[DeepSearch] 搜索工具初始化失败: {e}")

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "未找到相关搜索结果"

        formatted = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "无标题")
            content = item.get("content", "")
            url = item.get("url", "")
            content_preview = content[:600] if len(content) > 600 else content
            formatted.append(f"[来源{i}] {title}\n{content_preview}\n链接: {url}")
        return "\n\n".join(formatted)

    def _invoke_search(self, question: str):
        invoke_result = self.search_tool.invoke(question)

        sources: List[str] = []
        formatted_text = ""

        if isinstance(invoke_result, dict):
            results = invoke_result.get("results", [])
            sources = [
                item.get("url", "")
                for item in results
                if isinstance(item, dict) and item.get("url")
            ]
            formatted_text = self._format_search_results(results)
        elif isinstance(invoke_result, tuple) and len(invoke_result) == 2:
            content_str, artifact = invoke_result
            formatted_text = content_str if isinstance(content_str, str) else str(content_str)
            if isinstance(artifact, list):
                sources = [
                    item.get("url", "")
                    for item in artifact
                    if isinstance(item, dict) and item.get("url")
                ]
        elif isinstance(invoke_result, list):
            sources = [
                item.get("url", "")
                for item in invoke_result
                if isinstance(item, dict) and item.get("url")
            ]
            formatted_text = self._format_search_results(invoke_result)
        elif isinstance(invoke_result, str):
            formatted_text = invoke_result
        else:
            formatted_text = str(invoke_result)

        return formatted_text, sources

    def _clean_llm_text(self, raw) -> str:
        text = raw.content if hasattr(raw, "content") else str(raw)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = re.sub(r"</think>", "", text).strip()
        return text

    def search_raw(self, question: str) -> Dict[str, Any]:
        result = {
            "formatted_text": "",
            "sources": [],
            "error": None,
        }

        if not self.available:
            result["error"] = (
                "联网搜索功能未启用。请配置 TAVILY_API_KEY 环境变量后重启系统。\n"
                "申请地址：https://tavily.com"
            )
            return result

        try:
            print(f"[DeepSearch] 正在搜索: {question}")
            formatted_text, sources = self._invoke_search(question)
            result["formatted_text"] = formatted_text
            result["sources"] = sources
            print(f"[DeepSearch] 搜索完成，来源 {len(sources)} 个")
        except Exception as e:
            result["error"] = f"联网搜索失败: {str(e)}"
            print(f"[DeepSearch] 搜索出错: {e}")

        return result

    def synthesize_search(self, question: str, formatted_text: str, sources: List[str]) -> Dict[str, Any]:
        result = {
            "answer": None,
            "sources": sources or [],
            "error": None,
        }

        try:
            prompt = get_search_synthesis_prompt(question, formatted_text)
            result["answer"] = self._clean_llm_text(self.llm.invoke(prompt))
        except Exception as e:
            result["error"] = f"搜索结果综合失败: {str(e)}"
        return result

    def synthesize_search_and_sql(
        self,
        question: str,
        formatted_text: str,
        sources: List[str],
        sql_result_json: str,
    ) -> Dict[str, Any]:
        result = {
            "answer": None,
            "sources": sources or [],
            "error": None,
        }

        try:
            prompt = get_search_and_sql_prompt(question, formatted_text, sql_result_json)
            result["answer"] = self._clean_llm_text(self.llm.invoke(prompt))
        except Exception as e:
            result["error"] = f"联合搜索分析失败: {str(e)}"
        return result

    def search(self, question: str) -> Dict[str, Any]:
        raw_result = self.search_raw(question)
        if raw_result.get("error"):
            return {
                "answer": None,
                "sources": raw_result.get("sources", []),
                "error": raw_result["error"],
            }
        return self.synthesize_search(
            question,
            raw_result.get("formatted_text", ""),
            raw_result.get("sources", []),
        )

    def search_and_compare(self, question: str, sql_result_json: str) -> Dict[str, Any]:
        raw_result = self.search_raw(question)
        if raw_result.get("error"):
            return {
                "answer": None,
                "sources": raw_result.get("sources", []),
                "error": raw_result["error"],
            }
        return self.synthesize_search_and_sql(
            question,
            raw_result.get("formatted_text", ""),
            raw_result.get("sources", []),
            sql_result_json,
        )
