"""
主智能体

负责意图识别、任务路由、协调子智能体和结果汇总。
支持6种意图：simple_answer / sql_only / analysis_only / sql_and_analysis / web_search / search_and_sql
"""

import concurrent.futures
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from typing import TypedDict, Sequence, Dict, Any, Optional, Annotated, Generator, List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLLM

from prompts import get_master_intent_prompt, get_summary_prompt
from agents.sql_agent import SQLQueryAgent
from agents.analysis_agent import DataAnalysisAgent
from agents.search_agent import WebSearchAgent
from memory.long_term_memory import LongTermMemory
from memory.memory_extractor import MemoryExtractor
from memory.vector_store import VectorStore
import requests


class MasterAgentState(TypedDict):
    """主智能体状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question: str
    intent: Optional[str]
    sql_result: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    search_result: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]
    metadata: Dict[str, Any]


class MasterAgent:
    """主智能体 - 协调SQL查询和数据分析子智能体"""
    
    @staticmethod
    def _llm_to_str(result) -> str:
        """安全地从 LLM 返回值中提取文本字符串
        
        兼容 str / AIMessage / GenerationChunk 等多种返回类型。
        自动清理思考型模型（如 qwen3.5-plus）的 <think>...</think> 标签。
        """
        import re
        if isinstance(result, str):
            text = result
        elif hasattr(result, 'content'):
            text = str(result.content)
        elif hasattr(result, 'text'):
            text = str(result.text)
        else:
            text = str(result)
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
        text = re.sub(r'</think>', '', text).strip()
        return text

    def _simple_answer_text(self, question: str) -> str:
        """生成简单问候类回答。"""
        common_responses = {
            "你好": "你好！我是智能数据查询助手，可以帮你查询员工、部门、薪资等信息，还可以进行数据分析。有什么可以帮你的吗？",
            "谢谢": "不客气！还有什么其他问题吗？",
            "再见": "再见！祝你工作顺利！",
            "帮助": "我可以帮你：\n1. 查询数据库信息（如：有多少员工？）\n2. 分析数据（如：分析各部门薪资水平）\n3. 综合查询和分析（如：找出高薪员工并分析特征）",
        }

        for key, response in common_responses.items():
            if key in question:
                return response

        return "我是智能数据查询助手。请问有什么关于员工、部门或薪资的问题需要我帮忙吗？"

    def _run_search_and_sql_parallel(self, question: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """并行执行 SQL 查询和原始联网搜索，再做联合总结。"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            sql_future = pool.submit(self.sql_agent.query, question)
            search_future = pool.submit(self.search_agent.search_raw, question)
            sql_result = sql_future.result()
            raw_search_result = search_future.result()

        if raw_search_result.get("error"):
            search_result = {
                "answer": None,
                "sources": raw_search_result.get("sources", []),
                "error": raw_search_result["error"]
            }
        else:
            sql_data = sql_result.get("data", "{}") or "{}"
            search_result = self.search_agent.synthesize_search_and_sql(
                question,
                raw_search_result.get("formatted_text", ""),
                raw_search_result.get("sources", []),
                sql_data
            )

        return sql_result, search_result

    def _rule_based_intent(self, question: str) -> str:
        """当 LLM 识别失败或不稳定时，使用规则进行意图兜底。"""
        q = question.strip().lower()

        simple_keywords = ["你好", "您好", "谢谢", "再见", "你能做什么", "帮助", "hello", "hi"]
        if any(k in q for k in simple_keywords):
            return "simple_answer"

        analysis_keywords = ["分析", "总结", "洞察", "分布", "特征", "趋势", "对比"]
        sql_keywords = ["员工", "部门", "薪资", "工资", "多少", "查询", "统计", "排名", "最高", "最低"]
        web_keywords = ["行业", "市场", "全国", "互联网", "新闻", "最新", "趋势", "外部"]
        compare_keywords = ["相比", "对比", "比较", "处于什么水平"]
        internal_keywords = ["我们公司", "公司", "内部"]

        has_analysis = any(k in q for k in analysis_keywords)
        has_sql = any(k in q for k in sql_keywords)
        has_web = any(k in q for k in web_keywords)
        has_compare = any(k in q for k in compare_keywords)
        has_internal = any(k in q for k in internal_keywords)

        if has_web and has_internal and has_compare:
            return "search_and_sql"
        if has_web and not has_internal:
            return "web_search"
        if ("上一次" in q or "刚才" in q or "之前" in q) and has_analysis and not has_sql:
            return "analysis_only"
        if has_sql and has_analysis:
            return "sql_and_analysis"
        if has_sql:
            return "sql_only"
        if has_analysis:
            return "analysis_only"
        return "simple_answer"

    def _fallback_summary(
        self,
        question: str,
        sql_result: Optional[str],
        analysis_result: Optional[str]
    ) -> str:
        """当汇总 LLM 不可用时，生成可读的本地文本回答。"""
        if analysis_result:
            return str(analysis_result)

        if not sql_result:
            return "已完成处理，但当前没有可展示的数据结果。"

        try:
            parsed = json.loads(sql_result) if isinstance(sql_result, str) else sql_result
        except Exception:
            return f"查询已完成，原始结果如下：\n{sql_result}"

        if isinstance(parsed, dict):
            if parsed.get("error"):
                return f"查询失败：{parsed['error']}"
            if parsed.get("message"):
                return str(parsed["message"])
            return f"查询结果：{json.dumps(parsed, ensure_ascii=False)}"

        if isinstance(parsed, list):
            if not parsed:
                return "查询结果为空。"
            lines = [f"查询完成，共返回 {len(parsed)} 条记录。", "前几条记录："]
            for item in parsed[:5]:
                if isinstance(item, dict):
                    row = "，".join([f"{k}: {v}" for k, v in item.items()])
                    lines.append(f"- {row}")
                else:
                    lines.append(f"- {item}")
            lines.append("说明：当前回答由本地规则生成（LLM暂不可用）。")
            return "\n".join(lines)

        return f"查询结果：{parsed}"
    
    def __init__(self, llm: BaseLLM, db_path: str, num_examples: int = 3, 
                memory_db_path: str = "./data/long_term_memory.db",
                short_term_max_tokens: int = 1000,
                tavily_api_key: str = "",
                cache_store_path: str = "./data/query_cache.json",
                session_store_path: str = "./data/session_state.json",
                conversation_store_path: str = "./data/conversation_state.json",
                max_persisted_messages: int = 24):
        """初始化主智能体
        
        Args:
            llm: 语言模型实例
            db_path: 数据库路径
            num_examples: Few-shot示例数量
            memory_db_path: 长期记忆数据库路径
            short_term_max_tokens: 短期记忆最大token数
            tavily_api_key: Tavily 搜索 API Key
        """
        self.llm = llm
        self.db_path = db_path
        self.short_term_max_tokens = short_term_max_tokens
        self.max_persisted_messages = max_persisted_messages
        self.cache_store_path = Path(cache_store_path)
        self.session_store_path = Path(session_store_path)
        self.conversation_store_path = Path(conversation_store_path)
        self._persistence_lock = threading.Lock()
        
        # 初始化子智能体
        self.sql_agent = SQLQueryAgent(llm, db_path, num_examples)
        self.analysis_agent = DataAnalysisAgent(llm)
        self.search_agent = WebSearchAgent(llm, tavily_api_key=tavily_api_key)
        
        # 初始化短期记忆（MemorySaver）
        self.memory = MemorySaver()
        
        # 初始化向量存储（可选，默认关闭，数据量超过阈值才启用）
        self.vector_store = None
        milvus_enabled = os.getenv("MILVUS_ENABLED", "false").lower() == "true"
        if milvus_enabled:
            try:
                # 使用 SiliconFlow embedding API
                embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
                embedding_api_key = os.getenv("EMBEDDING_API_KEY")
                embedding_model = os.getenv("EMBEDDING_MODEL")
                
                def embedding_func(text: str):
                    try:
                        response = requests.post(
                            f"{embedding_base_url}/embeddings",
                            headers={"Authorization": f"Bearer {embedding_api_key}", "Content-Type": "application/json"},
                            json={"model": embedding_model, "input": text},
                            timeout=60
                        )
                        if response.status_code == 200:
                            data = response.json()
                            if "data" in data and len(data["data"]) > 0:
                                return data["data"][0]["embedding"]
                    except Exception as e:
                        logger.warning(f"生成 embedding 失败: {e}")
                    return None
                
                use_embedded = os.getenv("MILVUS_EMBEDDED", "false").lower() == "true"
                self.vector_store = VectorStore(
                    host=os.getenv("MILVUS_HOST", "localhost"),
                    port=int(os.getenv("MILVUS_PORT", "19530")),
                    embedding_func=embedding_func,
                    embedding_dim=4096,
                    enabled=True,
                    use_embedded=use_embedded
                )
                logger.info("Milvus 向量存储已启用")
            except Exception as e:
                logger.warning(f"Milvus 向量存储初始化失败: {e}")
        
        # 初始化长期记忆（LongTermMemory）
        self.long_term_memory = LongTermMemory(
            memory_db_path,
            vector_store=self.vector_store,
            use_vector_threshold=int(os.getenv("VECTOR_SEARCH_THRESHOLD", "100"))
        )
        
        # 初始化记忆提取器
        self.memory_extractor = MemoryExtractor(llm)
        
        # 会话数据存储：保存每个thread_id的最近查询结果
        self.session_data = {}

        
        # 结果缓存（问题关键词 -> 结构化缓存项列表）
        self._cache = {}
        self._conversation_store: Dict[str, List[Dict[str, str]]] = {}
        self._load_persistent_state()
        
        # 构建工作流
        self.graph = self._build_graph()

    def _read_json_file(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载持久化文件失败: {path} - {e}")
            return default

    def _write_json_file(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with self._persistence_lock:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            temp_path.replace(path)

    def _load_persistent_state(self) -> None:
        cache_data = self._read_json_file(self.cache_store_path, {})
        if isinstance(cache_data, dict):
            self._cache = {
                str(key): value
                for key, value in cache_data.items()
                if isinstance(value, list)
            }

        session_data = self._read_json_file(self.session_store_path, {})
        if isinstance(session_data, dict):
            self.session_data = {
                str(key): value
                for key, value in session_data.items()
                if isinstance(value, dict)
            }

        conversation_data = self._read_json_file(self.conversation_store_path, {})
        if isinstance(conversation_data, dict):
            self._conversation_store = {
                str(key): value
                for key, value in conversation_data.items()
                if isinstance(value, list)
            }

    def _save_cache_to_disk(self) -> None:
        if self.cache_store_path.exists():
            snapshot_dir = self.cache_store_path.parent / "cache_snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            snapshot_path = snapshot_dir / f"{self.cache_store_path.stem}_{timestamp}{self.cache_store_path.suffix}"
            if not snapshot_path.exists():
                shutil.copy2(self.cache_store_path, snapshot_path)
        self._write_json_file(self.cache_store_path, self._cache)

    def _save_session_data_to_disk(self) -> None:
        self._write_json_file(self.session_store_path, self.session_data)

    def _save_conversation_store_to_disk(self) -> None:
        self._write_json_file(self.conversation_store_path, self._conversation_store)

    def _serialize_message(self, message: BaseMessage) -> Optional[Dict[str, str]]:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        else:
            return None

        content = getattr(message, "content", "")
        if isinstance(content, list):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)
        elif not isinstance(content, str):
            content = str(content)

        return {"role": role, "content": content}

    def _deserialize_messages(self, records: Any) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        if not isinstance(records, list):
            return messages

        for item in records:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content", "")
            if not isinstance(content, str):
                content = str(content)

            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))

        return messages

    def _get_persisted_messages(self, thread_id: str) -> List[BaseMessage]:
        return self._deserialize_messages(self._conversation_store.get(thread_id, []))

    def _save_conversation_messages(self, thread_id: str, messages: Sequence[BaseMessage]) -> None:
        serialized: List[Dict[str, str]] = []
        for message in list(messages)[-self.max_persisted_messages:]:
            record = self._serialize_message(message)
            if record:
                serialized.append(record)

        self._conversation_store[thread_id] = serialized
        self._save_conversation_store_to_disk()

    def _get_checkpoint_messages(self, config: Dict[str, Any]) -> List[BaseMessage]:
        try:
            snapshot = self.graph.get_state(config)
            if snapshot and snapshot.values:
                return list(snapshot.values.get("messages", []))
        except Exception:
            pass
        return []

    def _remember_last_sql_result(self, thread_id: str, result: Dict[str, Any]) -> None:
        if thread_id not in self.session_data:
            self.session_data[thread_id] = {}
        self.session_data[thread_id]["last_sql_result"] = result
        self._save_session_data_to_disk()

    def _remember_conversation_turn(self, thread_id: str, question: str, answer: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        new_messages = [HumanMessage(content=question), AIMessage(content=answer)]
        checkpoint_messages = self._get_checkpoint_messages(config)
        persisted_messages = checkpoint_messages or self._get_persisted_messages(thread_id)
        self._save_conversation_messages(thread_id, persisted_messages + new_messages)

        try:
            self.graph.update_state(
                config,
                {"messages": new_messages},
                as_node="summarize"
            )
        except Exception:
            pass
    
    def _extract_departments(self, text: str) -> List[str]:
        """提取问题中的部门，并统一规范到“XX部”格式。"""
        dept_aliases = {
            "研发": "研发部",
            "产品": "产品部",
            "设计": "设计部",
            "市场": "市场部",
            "销售": "销售部",
            "运营": "运营部",
            "人事": "人事部",
            "财务": "财务部",
            "技术": "技术部",
            "测试": "测试部",
            "运维": "运维部",
        }

        found: List[str] = []

        for match in re.findall(r"([\u4e00-\u9fff]{1,8}部)", text):
            dept_name = match.strip()
            if dept_name not in found:
                found.append(dept_name)

        for alias, canonical in dept_aliases.items():
            if alias in text and canonical not in found:
                found.append(canonical)

        return found

    def _extract_cities(self, text: str) -> List[str]:
        """Extract city names from the question."""
        city_aliases = {
            "北京": "北京",
            "上海": "上海",
            "广州": "广州",
            "深圳": "深圳",
        }

        found: List[str] = []
        for alias, canonical in city_aliases.items():
            if alias in text and canonical not in found:
                found.append(canonical)

        return found

    def _build_cache_metadata(self, question: str) -> Dict[str, Any]:
        """构建缓存匹配所需的关键词和结构化元数据。"""
        q = question.lower()
        departments = self._extract_departments(question)
        department_tokens = [
            dept[:-1] if dept.endswith("部") else dept for dept in departments
        ]

        keywords = list(department_tokens)
        matched_actions: List[str] = []

        actions = ["平均", "最高", "最低", "总", "排名", "统计", "查询", "对比", "分析"]
        for action in actions:
            if action in q:
                keywords.append(action)
                matched_actions.append(action)

        salary_related = "薪资" in q or "工资" in q or "收入" in q
        if salary_related:
            keywords.append("薪资")

        if not keywords:
            keywords = [q[:10]]

        return {
            "cache_key": "|".join(keywords),
            "departments": departments,
            "department_tokens": department_tokens,
            "actions": matched_actions,
            "salary_related": salary_related,
        }

    def _get_cache_key(self, question: str) -> str:
        """从问题中提取缓存key（关键词）"""
        return self._build_cache_metadata(question)["cache_key"]

    def _normalize_department_value(self, value: Any) -> str:
        """将数据中的部门值归一化为“XX部”格式。"""
        if not isinstance(value, str):
            return ""

        normalized = value.strip()
        if normalized.endswith("部门"):
            normalized = normalized[:-2] + "部"

        alias_map = {
            "研发": "研发部",
            "产品": "产品部",
            "设计": "设计部",
            "市场": "市场部",
            "销售": "销售部",
            "运营": "运营部",
            "人事": "人事部",
            "财务": "财务部",
            "技术": "技术部",
            "测试": "测试部",
            "运维": "运维部",
        }

        if normalized in alias_map:
            return alias_map[normalized]
        return normalized

    def _parse_cached_sql_data(self, sql_data: Any) -> Any:
        """解析缓存中的 SQL 原始结果。"""
        if not sql_data:
            return None
        if isinstance(sql_data, str):
            try:
                return json.loads(sql_data)
            except Exception:
                return None
        return sql_data

    def _filter_cached_sql_data(
        self, sql_data: Any, requested_departments: List[str]
    ) -> Optional[Any]:
        """按当前问题中的部门，从缓存的 SQL 结果里筛出真正需要的数据。"""
        parsed = self._parse_cached_sql_data(sql_data)
        if parsed is None:
            return None

        if not requested_departments:
            return parsed

        requested_set = set(requested_departments)
        requested_roots = {
            dept[:-1] if dept.endswith("部") else dept for dept in requested_departments
        }

        if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
            filtered_rows = self._filter_cached_sql_data(parsed["rows"], requested_departments)
            if filtered_rows is None:
                return None
            filtered_payload = dict(parsed)
            filtered_payload["rows"] = filtered_rows
            return filtered_payload

        if not isinstance(parsed, list):
            return None

        filtered_rows = []
        matched_any_department = False

        for row in parsed:
            if not isinstance(row, dict):
                continue

            row_match = False
            for key, value in row.items():
                normalized_value = self._normalize_department_value(value)
                if not normalized_value:
                    continue

                key_text = str(key).lower()
                root_value = (
                    normalized_value[:-1]
                    if normalized_value.endswith("部")
                    else normalized_value
                )

                if "dept" in key_text or "department" in key_text or "部门" in str(key):
                    matched_any_department = True
                    if normalized_value in requested_set or root_value in requested_roots:
                        row_match = True
                        break

            if not row_match:
                for value in row.values():
                    normalized_value = self._normalize_department_value(value)
                    if not normalized_value:
                        continue

                    root_value = (
                        normalized_value[:-1]
                        if normalized_value.endswith("部")
                        else normalized_value
                    )
                    if normalized_value in requested_set or root_value in requested_roots:
                        row_match = True
                        break

            if row_match:
                filtered_rows.append(row)

        if matched_any_department and filtered_rows:
            return filtered_rows

        return None

    def _rebuild_cached_answer(self, question: str, cache_entry: Dict[str, Any]) -> Optional[str]:
        """对可复用的结构化缓存重新生成回答，避免把未命中的部门一并返回。"""
        requested_departments = self._extract_departments(question)
        sql_result = cache_entry.get("sql_result") or {}
        filtered_sql_data = self._filter_cached_sql_data(
            sql_result.get("data"),
            requested_departments,
        )
        if filtered_sql_data is None:
            return None

        serialized_sql = json.dumps(filtered_sql_data, ensure_ascii=False)
        summary_prompt = get_summary_prompt(
            question=question,
            sql_result=serialized_sql,
            analysis_result=None,
        )

        try:
            answer = self._llm_to_str(self.llm.invoke(summary_prompt)).strip()
            if answer:
                return answer
        except Exception as e:
            logger.warning(f"缓存命中后二次生成回答失败，使用兜底总结: {e}")

        return self._fallback_summary(question, serialized_sql, None)

    def _find_cached_answer(self, question: str) -> str | None:
        """从缓存中查找可安全复用的回答。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        logger.info(f"缓存查找 - 问题: {question[:20]}, key: {cache_key}")

        # 1. 同问题或同 key，优先直接命中
        if cache_key in self._cache:
            entries = self._cache[cache_key]
            for entry in entries:
                if entry.get("question") == question and entry.get("answer"):
                    logger.info(f"精确问题命中缓存: {cache_key}")
                    return entry["answer"]

            for entry in entries:
                if entry.get("answer"):
                    logger.info(f"精确 key 命中缓存: {cache_key}")
                    return entry["answer"]

        # 2. 允许命中“更大结果集”的缓存，但只返回当前问题实际命中的部门
        requested_departments = set(metadata["departments"])
        requested_actions = set(metadata["actions"])
        salary_related = metadata["salary_related"]

        if requested_departments:
            for cached_entries in self._cache.values():
                for entry in cached_entries:
                    cached_departments = set(entry.get("departments", []))
                    if not cached_departments:
                        continue
                    if not requested_departments.issubset(cached_departments):
                        continue
                    if requested_actions != set(entry.get("actions", [])):
                        continue
                    if salary_related != entry.get("salary_related", False):
                        continue

                    rebuilt_answer = self._rebuild_cached_answer(question, entry)
                    if rebuilt_answer:
                        logger.info(
                            "命中可裁剪缓存，已按当前问题过滤部门后重建回答"
                        )
                        return rebuilt_answer

        logger.info(f"当前缓存keys: {list(self._cache.keys())}")
        logger.info("缓存未命中")
        return None

    def _add_to_cache(
        self,
        question: str,
        answer: str,
        intent: Optional[str] = None,
        sql_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        search_result: Optional[Dict[str, Any]] = None,
    ):
        """将结果添加到缓存。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        cache_entry = {
            "question": question,
            "answer": answer,
            "intent": intent,
            "sql_result": sql_result,
            "analysis_result": analysis_result,
            "search_result": search_result,
            "departments": metadata["departments"],
            "actions": metadata["actions"],
            "salary_related": metadata["salary_related"],
        }

        existing_entries = self._cache.get(cache_key, [])
        existing_entries = [
            entry for entry in existing_entries if entry.get("question") != question
        ]
        existing_entries.insert(0, cache_entry)
        self._cache[cache_key] = existing_entries[:3]
        self._save_cache_to_disk()

        logger.info(f"结果已加入缓存: {cache_key}")
    
    def _extract_cities(self, text: str) -> List[str]:
        """提取问题中的城市。"""
        city_aliases = {
            "北京": "北京",
            "上海": "上海",
            "广州": "广州",
            "深圳": "深圳",
        }

        found: List[str] = []
        for alias, canonical in city_aliases.items():
            if alias in text and canonical not in found:
                found.append(canonical)

        return found

    def _build_cache_metadata(self, question: str) -> Dict[str, Any]:
        """构建缓存匹配所需的关键词和结构化元数据。"""
        q = question.lower()
        departments = self._extract_departments(question)
        cities = self._extract_cities(question)
        department_tokens = [
            dept[:-1] if dept.endswith("部") else dept for dept in departments
        ]

        keywords = list(department_tokens) + list(cities)
        matched_actions: List[str] = []

        actions = ["平均", "最高", "最低", "总", "排名", "统计", "查询", "对比", "分析"]
        for action in actions:
            if action in q:
                keywords.append(action)
                matched_actions.append(action)

        salary_related = "薪资" in q or "工资" in q or "收入" in q
        if salary_related:
            keywords.append("薪资")

        if not keywords:
            keywords = [q[:10]]

        return {
            "cache_key": "|".join(keywords),
            "departments": departments,
            "cities": cities,
            "department_tokens": department_tokens,
            "actions": matched_actions,
            "salary_related": salary_related,
        }

    def _normalize_city_value(self, value: Any) -> str:
        """将数据中的城市值归一化。"""
        if not isinstance(value, str):
            return ""

        normalized = value.strip()
        alias_map = {
            "北京市": "北京",
            "上海市": "上海",
            "广州市": "广州",
            "深圳市": "深圳",
            "北京": "北京",
            "上海": "上海",
            "广州": "广州",
            "深圳": "深圳",
        }
        return alias_map.get(normalized, normalized)

    def _filter_cached_sql_data(
        self,
        sql_data: Any,
        requested_departments: List[str],
        requested_cities: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """按当前问题中的部门/城市，从缓存 SQL 结果里筛出真正需要的数据。"""
        parsed = self._parse_cached_sql_data(sql_data)
        if parsed is None:
            return None

        requested_cities = requested_cities or []
        if not requested_departments and not requested_cities:
            return parsed

        requested_dept_set = set(requested_departments)
        requested_dept_roots = {
            dept[:-1] if dept.endswith("部") else dept for dept in requested_departments
        }
        requested_city_set = set(requested_cities)

        if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
            filtered_rows = self._filter_cached_sql_data(
                parsed["rows"],
                requested_departments,
                requested_cities,
            )
            if filtered_rows is None:
                return None
            filtered_payload = dict(parsed)
            filtered_payload["rows"] = filtered_rows
            return filtered_payload

        if not isinstance(parsed, list):
            return None

        filtered_rows = []
        matched_any_department = False
        matched_any_city = False

        for row in parsed:
            if not isinstance(row, dict):
                continue

            department_match = not requested_departments
            city_match = not requested_cities

            for key, value in row.items():
                key_text = str(key).lower()

                if requested_departments:
                    normalized_dept = self._normalize_department_value(value)
                    dept_root = (
                        normalized_dept[:-1]
                        if normalized_dept.endswith("部")
                        else normalized_dept
                    )
                    if "dept" in key_text or "department" in key_text or "部门" in str(key):
                        matched_any_department = True
                        if (
                            normalized_dept in requested_dept_set
                            or dept_root in requested_dept_roots
                        ):
                            department_match = True

                if requested_cities:
                    normalized_city = self._normalize_city_value(value)
                    if "city" in key_text or "location" in key_text or "城市" in str(key) or "地区" in str(key):
                        matched_any_city = True
                        if normalized_city in requested_city_set:
                            city_match = True

            if requested_departments and not department_match:
                for value in row.values():
                    normalized_dept = self._normalize_department_value(value)
                    dept_root = (
                        normalized_dept[:-1]
                        if normalized_dept.endswith("部")
                        else normalized_dept
                    )
                    if (
                        normalized_dept in requested_dept_set
                        or dept_root in requested_dept_roots
                    ):
                        department_match = True
                        break

            if requested_cities and not city_match:
                for value in row.values():
                    normalized_city = self._normalize_city_value(value)
                    if normalized_city in requested_city_set:
                        city_match = True
                        break

            if department_match and city_match:
                filtered_rows.append(row)

        if requested_departments and matched_any_department and not filtered_rows:
            return None
        if requested_cities and matched_any_city and not filtered_rows:
            return None
        if filtered_rows:
            return filtered_rows
        return None

    def _rebuild_cached_answer(self, question: str, cache_entry: Dict[str, Any]) -> Optional[str]:
        """对可复用缓存按当前问题裁剪数据后重建回答。"""
        requested_departments = self._extract_departments(question)
        requested_cities = self._extract_cities(question)
        sql_result = cache_entry.get("sql_result") or {}
        filtered_sql_data = self._filter_cached_sql_data(
            sql_result.get("data"),
            requested_departments,
            requested_cities,
        )
        if filtered_sql_data is None:
            return None

        serialized_sql = json.dumps(filtered_sql_data, ensure_ascii=False)
        summary_prompt = get_summary_prompt(
            question=question,
            sql_result=serialized_sql,
            analysis_result=None,
        )

        try:
            answer = self._llm_to_str(self.llm.invoke(summary_prompt)).strip()
            if answer:
                return answer
        except Exception as e:
            logger.warning(f"缓存命中后二次生成回答失败，使用兜底总结: {e}")

        return self._fallback_summary(question, serialized_sql, None)

    def _find_cached_answer(self, question: str) -> str | None:
        """从缓存中查找可安全复用的回答。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        logger.info(f"缓存查找 - 问题: {question[:20]}, key: {cache_key}")

        if cache_key in self._cache:
            entries = self._cache[cache_key]
            for entry in entries:
                if entry.get("question") == question and entry.get("answer"):
                    logger.info(f"精确问题命中缓存: {cache_key}")
                    return entry["answer"]

            for entry in entries:
                if entry.get("answer"):
                    logger.info(f"精确 key 命中缓存: {cache_key}")
                    return entry["answer"]

        requested_departments = set(metadata["departments"])
        requested_cities = set(metadata.get("cities", []))
        requested_actions = set(metadata["actions"])
        salary_related = metadata["salary_related"]

        if requested_departments or requested_cities:
            for cached_entries in self._cache.values():
                for entry in cached_entries:
                    cached_departments = set(entry.get("departments", []))
                    cached_cities = set(entry.get("cities", []))

                    if requested_departments and not requested_departments.issubset(cached_departments):
                        continue
                    if requested_cities and not requested_cities.issubset(cached_cities):
                        continue
                    if requested_actions != set(entry.get("actions", [])):
                        continue
                    if salary_related != entry.get("salary_related", False):
                        continue

                    rebuilt_answer = self._rebuild_cached_answer(question, entry)
                    if rebuilt_answer:
                        logger.info("命中可裁剪缓存，已按当前问题过滤后重建回答")
                        return rebuilt_answer

        logger.info(f"当前缓存keys: {list(self._cache.keys())}")
        logger.info("缓存未命中")
        return None

    def _add_to_cache(
        self,
        question: str,
        answer: str,
        intent: Optional[str] = None,
        sql_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        search_result: Optional[Dict[str, Any]] = None,
    ):
        """将结果添加到缓存。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        cache_entry = {
            "question": question,
            "answer": answer,
            "intent": intent,
            "sql_result": sql_result,
            "analysis_result": analysis_result,
            "search_result": search_result,
            "departments": metadata["departments"],
            "cities": metadata.get("cities", []),
            "actions": metadata["actions"],
            "salary_related": metadata["salary_related"],
        }

        existing_entries = self._cache.get(cache_key, [])
        existing_entries = [
            entry for entry in existing_entries if entry.get("question") != question
        ]
        existing_entries.insert(0, cache_entry)
        self._cache[cache_key] = existing_entries[:3]
        self._save_cache_to_disk()

        logger.info(f"结果已加入缓存: {cache_key}")

    def _get_cache_dimension_specs(self) -> Dict[str, Dict[str, Any]]:
        """定义可裁剪缓存的维度配置。新增维度时只需扩展这里。"""
        return {
            "departments": {
                "extractor": self._extract_departments,
                "normalizer": self._normalize_department_value,
                "field_keywords": ["dept", "department", "部门"],
            },
            "cities": {
                "extractor": self._extract_cities,
                "normalizer": self._normalize_city_value,
                "field_keywords": ["city", "location", "城市", "地区"],
            },
        }

    def _extract_cache_dimensions(self, text: str) -> Dict[str, List[str]]:
        dimensions: Dict[str, List[str]] = {}
        for name, spec in self._get_cache_dimension_specs().items():
            values = spec["extractor"](text)
            if values:
                dimensions[name] = values
        return dimensions

    def _normalize_city_value(self, value: Any) -> str:
        """将数据中的城市值归一化。"""
        if not isinstance(value, str):
            return ""

        normalized = value.strip()
        alias_map = {
            "北京市": "北京",
            "上海市": "上海",
            "广州市": "广州",
            "深圳市": "深圳",
            "北京": "北京",
            "上海": "上海",
            "广州": "广州",
            "深圳": "深圳",
        }
        return alias_map.get(normalized, normalized)

    def _normalize_dimension_target(self, dimension: str, value: str) -> str:
        spec = self._get_cache_dimension_specs().get(dimension)
        if not spec:
            return value
        normalizer = spec["normalizer"]
        normalized = normalizer(value)
        return normalized or value

    def _get_entry_dimensions(self, entry: Dict[str, Any]) -> Dict[str, List[str]]:
        dimensions = entry.get("dimensions")
        if isinstance(dimensions, dict):
            return {
                str(key): [str(v) for v in values if isinstance(v, str)]
                for key, values in dimensions.items()
                if isinstance(values, list)
            }

        fallback: Dict[str, List[str]] = {}
        for key in self._get_cache_dimension_specs().keys():
            values = entry.get(key)
            if isinstance(values, list):
                fallback[key] = [str(v) for v in values if isinstance(v, str)]
        return fallback

    def _build_cache_metadata(self, question: str) -> Dict[str, Any]:
        """构建缓存匹配所需的关键词和结构化元数据。"""
        q = question.lower()
        dimensions = self._extract_cache_dimensions(question)
        keywords: List[str] = []

        for dimension_values in dimensions.values():
            keywords.extend(dimension_values)

        matched_actions: List[str] = []
        actions = ["平均", "最高", "最低", "总", "排名", "统计", "查询", "对比", "分析"]
        for action in actions:
            if action in q:
                keywords.append(action)
                matched_actions.append(action)

        salary_related = "薪资" in q or "工资" in q or "收入" in q
        if salary_related:
            keywords.append("薪资")

        if not keywords:
            keywords = [q[:10]]

        return {
            "cache_key": "|".join(keywords),
            "dimensions": dimensions,
            "departments": dimensions.get("departments", []),
            "cities": dimensions.get("cities", []),
            "actions": matched_actions,
            "salary_related": salary_related,
        }

    def _filter_cached_sql_data(
        self,
        sql_data: Any,
        requested_dimensions: Dict[str, List[str]],
    ) -> Optional[Any]:
        """按当前问题维度从缓存 SQL 结果中裁剪有效数据。"""
        parsed = self._parse_cached_sql_data(sql_data)
        if parsed is None:
            return None

        active_dimensions = {
            name: values for name, values in requested_dimensions.items() if values
        }
        if not active_dimensions:
            return parsed

        if isinstance(parsed, dict) and isinstance(parsed.get("rows"), list):
            filtered_rows = self._filter_cached_sql_data(parsed["rows"], active_dimensions)
            if filtered_rows is None:
                return None
            filtered_payload = dict(parsed)
            filtered_payload["rows"] = filtered_rows
            return filtered_payload

        if not isinstance(parsed, list):
            return None

        specs = self._get_cache_dimension_specs()
        normalized_targets = {
            name: {
                self._normalize_dimension_target(name, value)
                for value in values
                if isinstance(value, str) and value
            }
            for name, values in active_dimensions.items()
        }
        matched_by_key = {name: False for name in active_dimensions}
        filtered_rows = []

        for row in parsed:
            if not isinstance(row, dict):
                continue

            row_match = True
            for dimension, targets in normalized_targets.items():
                spec = specs.get(dimension)
                if not spec or not targets:
                    continue

                normalizer = spec["normalizer"]
                field_keywords = spec["field_keywords"]
                explicit_match = False
                fallback_match = False
                saw_dimension_key = False

                for key, value in row.items():
                    key_text = str(key).lower()
                    normalized_value = normalizer(value)
                    if not normalized_value:
                        continue

                    if any(keyword in key_text or keyword in str(key) for keyword in field_keywords):
                        saw_dimension_key = True
                        matched_by_key[dimension] = True
                        if normalized_value in targets:
                            explicit_match = True
                            break

                if not explicit_match:
                    for value in row.values():
                        normalized_value = normalizer(value)
                        if normalized_value and normalized_value in targets:
                            fallback_match = True
                            break

                dimension_match = explicit_match if saw_dimension_key else fallback_match
                if not dimension_match:
                    row_match = False
                    break

            if row_match:
                filtered_rows.append(row)

        for dimension in active_dimensions:
            if matched_by_key.get(dimension) and not filtered_rows:
                return None

        return filtered_rows or None

    def _rebuild_cached_answer(self, question: str, cache_entry: Dict[str, Any]) -> Optional[str]:
        """对可复用缓存按当前维度裁剪数据后重建回答。"""
        metadata = self._build_cache_metadata(question)
        sql_result = cache_entry.get("sql_result") or {}
        filtered_sql_data = self._filter_cached_sql_data(
            sql_result.get("data"),
            metadata["dimensions"],
        )
        if filtered_sql_data is None:
            return None

        serialized_sql = json.dumps(filtered_sql_data, ensure_ascii=False)
        summary_prompt = get_summary_prompt(
            question=question,
            sql_result=serialized_sql,
            analysis_result=None,
        )

        try:
            answer = self._llm_to_str(self.llm.invoke(summary_prompt)).strip()
            if answer:
                return answer
        except Exception as e:
            logger.warning(f"缓存命中后二次生成回答失败，使用兜底总结: {e}")

        return self._fallback_summary(question, serialized_sql, None)

    def _find_cached_answer(self, question: str) -> str | None:
        """从缓存中查找可安全复用的回答。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        logger.info(f"缓存查找 - 问题: {question[:20]}, key: {cache_key}")

        if cache_key in self._cache:
            entries = self._cache[cache_key]
            for entry in entries:
                if entry.get("question") == question and entry.get("answer"):
                    logger.info(f"精确问题命中缓存: {cache_key}")
                    return entry["answer"]

            for entry in entries:
                if entry.get("answer"):
                    logger.info(f"精确 key 命中缓存: {cache_key}")
                    return entry["answer"]

        requested_dimensions = metadata["dimensions"]
        requested_actions = set(metadata["actions"])
        salary_related = metadata["salary_related"]

        if requested_dimensions:
            for cached_entries in self._cache.values():
                for entry in cached_entries:
                    cached_dimensions = self._get_entry_dimensions(entry)

                    dimensions_match = True
                    for dimension, requested_values in requested_dimensions.items():
                        cached_values = {
                            self._normalize_dimension_target(dimension, value)
                            for value in cached_dimensions.get(dimension, [])
                        }
                        normalized_requested = {
                            self._normalize_dimension_target(dimension, value)
                            for value in requested_values
                        }
                        if not normalized_requested.issubset(cached_values):
                            dimensions_match = False
                            break

                    if not dimensions_match:
                        continue
                    if requested_actions != set(entry.get("actions", [])):
                        continue
                    if salary_related != entry.get("salary_related", False):
                        continue

                    rebuilt_answer = self._rebuild_cached_answer(question, entry)
                    if rebuilt_answer:
                        logger.info("命中可裁剪缓存，已按当前维度过滤后重建回答")
                        return rebuilt_answer

        logger.info(f"当前缓存keys: {list(self._cache.keys())}")
        logger.info("缓存未命中")
        return None

    def _add_to_cache(
        self,
        question: str,
        answer: str,
        intent: Optional[str] = None,
        sql_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        search_result: Optional[Dict[str, Any]] = None,
    ):
        """将结果添加到缓存。"""
        metadata = self._build_cache_metadata(question)
        cache_key = metadata["cache_key"]

        cache_entry = {
            "question": question,
            "answer": answer,
            "intent": intent,
            "sql_result": sql_result,
            "analysis_result": analysis_result,
            "search_result": search_result,
            "dimensions": metadata["dimensions"],
            "departments": metadata.get("departments", []),
            "cities": metadata.get("cities", []),
            "actions": metadata["actions"],
            "salary_related": metadata["salary_related"],
        }

        existing_entries = self._cache.get(cache_key, [])
        existing_entries = [
            entry for entry in existing_entries if entry.get("question") != question
        ]
        existing_entries.insert(0, cache_entry)
        self._cache[cache_key] = existing_entries[:3]
        self._save_cache_to_disk()

        logger.info(f"结果已加入缓存: {cache_key}")

    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图（支持6种意图路由）"""
        workflow = StateGraph(MasterAgentState)
        
        # 添加节点
        workflow.add_node("intent", self._intent_node)
        workflow.add_node("simple_answer", self._simple_answer_node)
        workflow.add_node("call_sql", self._call_sql_node)
        workflow.add_node("call_analysis", self._call_analysis_node)
        workflow.add_node("call_both", self._call_both_node)
        workflow.add_node("call_web_search", self._call_web_search_node)
        workflow.add_node("call_search_and_sql", self._call_search_and_sql_node)
        workflow.add_node("summarize", self._summarize_node)
        
        # 设置入口
        workflow.set_entry_point("intent")
        
        # 添加条件边 - 从意图识别到不同的处理节点（6种意图）
        workflow.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {
                "simple_answer": "simple_answer",
                "sql_only": "call_sql",
                "analysis_only": "call_analysis",
                "sql_and_analysis": "call_both",
                "web_search": "call_web_search",
                "search_and_sql": "call_search_and_sql"
            }
        )
        
        # 添加边
        workflow.add_edge("simple_answer", END)
        workflow.add_edge("call_sql", "summarize")
        workflow.add_edge("call_analysis", "summarize")
        workflow.add_edge("call_both", "summarize")
        workflow.add_edge("call_web_search", "summarize")
        workflow.add_edge("call_search_and_sql", "summarize")
        workflow.add_edge("summarize", END)
        
        # 使用MemorySaver作为checkpointer
        return workflow.compile(checkpointer=self.memory)
    
    def _get_conversation_history(self, state: MasterAgentState) -> str:
        """获取对话历史摘要（智能压缩版本）
        
        策略：
        1. 如果消息少于等于10条，直接返回所有
        2. 如果消息较多但token未超限，返回近期消息
        3. 如果消息很多且超过token限制，使用LLM总结压缩
        
        Args:
            state: 当前状态
            
        Returns:
            对话历史摘要
        """
        messages = state.get("messages", [])
        if len(messages) <= 1:
            return ""
        
        # 构建原始历史（排除当前消息）
        history_text = self._format_messages(messages[:-1])
        
        # 如果消息数量少，直接返回
        if len(messages) <= 11:  # 10条历史消息
            return history_text
        
        # 简单token估算（中文按2字符=1token，英文按4字符=1token）
        estimated_tokens = len(history_text) / 2.5
        
        if estimated_tokens <= self.short_term_max_tokens:
            return history_text
        
        # 需要压缩：使用LLM总结
        return self._compress_history_with_llm(history_text)
    
    def _format_messages(self, messages: Sequence[BaseMessage]) -> str:
        """格式化消息列表为文本
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的文本
        """
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"助手: {msg.content}")
        
        return "\n".join(history) if history else ""
    
    def _compress_history_with_llm(self, history_text: str) -> str:
        """使用LLM总结压缩对话历史
        
        Args:
            history_text: 原始对话历史文本
            
        Returns:
            压缩后的摘要文本
        """
        prompt = f"""请总结以下对话历史，保留关键信息、用户偏好和重要上下文：

{history_text}

总结要求：
1. 保留关键事实和数据（如查询的部门、员工、数据结果）
2. 提取用户关注点和偏好
3. 保留重要的上下文信息
4. 简洁但信息完整
5. 不超过300字

总结："""
        
        try:
            summary = self._llm_to_str(self.llm.invoke(prompt)).strip()
            return f"[对话历史总结]\n{summary}"
        except Exception as e:
            logger.warning(f"压缩对话历史失败: {e}")
            # 如果压缩失败，返回最近的部分对话
            lines = history_text.split("\n")
            recent_lines = lines[-20:] if len(lines) > 20 else lines
            return "\n".join(recent_lines)
    
    def _format_long_term_context(
        self, 
        knowledge: list, 
        preferences: Dict[str, str]
    ) -> str:
        """格式化长期记忆上下文
        
        Args:
            knowledge: 用户知识列表
            preferences: 用户偏好字典
            
        Returns:
            格式化的上下文文本
        """
        context_parts = []
        
        # 添加用户偏好
        if preferences:
            pref_lines = [f"- {key}: {value}" for key, value in preferences.items()]
            context_parts.append("用户偏好：\n" + "\n".join(pref_lines))
        
        # 添加相关知识
        if knowledge:
            know_lines = [f"- {item['content']}" for item in knowledge[:3]]
            context_parts.append("相关背景：\n" + "\n".join(know_lines))
        
        return "\n\n".join(context_parts) if context_parts else ""

    def _get_session_memory_facts(
        self,
        thread_id: str,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        """Return recent fact memories stored for the current thread."""
        thread_state = self.session_data.get(thread_id, {})
        if not isinstance(thread_state, dict):
            return []

        facts = thread_state.get("memory_facts", [])
        if not isinstance(facts, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for item in facts:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            cleaned.append(item)

        return cleaned[:limit]

    def _format_memory_facts(self, facts: List[Dict[str, Any]], limit: int = 5) -> str:
        """Format stored fact memories for prompting."""
        sections: List[str] = []
        for index, fact in enumerate(facts[:limit], start=1):
            question = str(fact.get("question", "")).strip()
            answer = str(fact.get("answer", "")).strip()
            if not question or not answer:
                continue
            sections.append(f"{index}. 问题：{question}\n答案：{answer}")
        return "\n\n".join(sections)

    def _remember_answer_fact(
        self,
        thread_id: str,
        question: str,
        answer: str,
        intent: Optional[str] = None,
        user_id: Optional[str] = None,
        sql_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist resolved answers as reusable memory facts."""
        clean_question = question.strip()
        clean_answer = answer.strip()
        if not clean_question or not clean_answer:
            return

        thread_state = self.session_data.setdefault(thread_id, {})
        existing_facts = thread_state.get("memory_facts", [])
        if not isinstance(existing_facts, list):
            existing_facts = []

        fact_record = {
            "question": clean_question,
            "answer": clean_answer,
            "intent": intent or "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        deduped = [
            item
            for item in existing_facts
            if isinstance(item, dict) and item.get("question") != clean_question
        ]
        thread_state["memory_facts"] = [fact_record] + deduped[:7]
        self._save_session_data_to_disk()

        if not user_id:
            return

        memory_intents = {"sql_only", "sql_and_analysis", "analysis_only", "memory_answer", "cache_answer"}
        if intent not in memory_intents:
            return

        if sql_result and isinstance(sql_result, dict) and sql_result.get("error"):
            return

        memory_content = f"问题：{clean_question}\n结论：{clean_answer}"
        try:
            self.long_term_memory.save_knowledge(
                user_id,
                "dialogue_fact",
                memory_content,
                confidence=0.92 if intent in {"sql_only", "sql_and_analysis"} else 0.85,
            )
        except Exception as e:
            logger.warning(f"保存对话事实记忆失败: {e}")

    def _build_memory_answer_context(
        self,
        question: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Collect short-term and long-term memory snippets for direct answering."""
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_messages = self._get_checkpoint_messages(config)
        persisted_messages = checkpoint_messages or self._get_persisted_messages(thread_id)
        recent_history = self._format_messages(list(persisted_messages)[-6:])

        facts = self._get_session_memory_facts(thread_id, limit=5)
        facts_text = self._format_memory_facts(facts, limit=5)

        last_sql_text = ""
        thread_state = self.session_data.get(thread_id, {})
        if isinstance(thread_state, dict):
            last_sql_result = thread_state.get("last_sql_result")
            if isinstance(last_sql_result, dict):
                parsed_data = self._parse_cached_sql_data(last_sql_result.get("data"))
                if parsed_data is not None:
                    last_sql_text = json.dumps(parsed_data, ensure_ascii=False)

        knowledge_text = ""
        knowledge_comparison = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=4)
                logger.info(f"知识检索结果数量: {len(knowledge)}")
                
                vector_results = [item for item in knowledge if item.get("search_method") == "vector"]
                keyword_results = [item for item in knowledge if item.get("search_method") == "keyword"]
                
                logger.info(f"向量结果: {len(vector_results)}, 关键词结果: {len(keyword_results)}")
                
                vector_time = 0
                keyword_time = 0
                
                if vector_results:
                    vector_time = vector_results[0].get("search_time", 0)
                    vector_lines = [f"- {item['content']}" for item in vector_results if item.get("content")]
                    knowledge_text += "【向量数据库检索结果】\n" + "\n".join(vector_lines) + f"\n(耗时: {vector_time:.3f}s)\n\n"
                
                if keyword_results:
                    keyword_time = keyword_results[0].get("search_time", 0)
                    keyword_lines = [f"- {item['content']}" for item in keyword_results if item.get("content")]
                    knowledge_text += "【普通数据库检索结果】\n" + "\n".join(keyword_lines) + f"\n(耗时: {keyword_time:.3f}s)"
                
                if vector_time > 0 and keyword_time > 0:
                    speed_diff = "向量检索" if vector_time < keyword_time else "普通检索"
                    faster_time = min(vector_time, keyword_time)
                    slower_time = max(vector_time, keyword_time)
                    knowledge_comparison = f"\n\n速度对比: {speed_diff}更快 ({faster_time:.3f}s vs {slower_time:.3f}s)"
                    knowledge_text += knowledge_comparison
                    
            except Exception as e:
                logger.warning(f"读取长期记忆失败: {e}")

        return {
            "recent_history": recent_history,
            "facts_text": facts_text,
            "last_sql_text": last_sql_text,
            "knowledge_text": knowledge_text,
        }

    def _try_answer_from_memory(
        self,
        question: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Answer from memory first; fall back to SQL when memory is insufficient."""
        memory_context = self._build_memory_answer_context(question, thread_id, user_id)
        if not any(memory_context.values()):
            return None

        prompt = f"""你是一个对话记忆助手。请根据给定的记忆回答当前问题。

当前问题：
{question}

最近对话：
{memory_context['recent_history'] or '无'}

会话事实记忆：
{memory_context['facts_text'] or '无'}

最近一次结构化查询结果：
{memory_context['last_sql_text'] or '无'}

长期记忆（包含向量检索和普通检索的对比）：
{memory_context['knowledge_text'] or '无'}

要求：
1. 必须同时展示向量数据库和普通数据库的检索结果。
2. 必须包含两种检索方式的耗时对比。
3. 如果两者结果一致，说明结果可靠；如果不一致，需要指出差异。
4. 记忆不足时只返回 NEED_FRESH_DATA。"""

        try:
            answer = self._llm_to_str(self.llm.invoke(prompt)).strip()
        except Exception as e:
            logger.warning(f"记忆直答失败，回退常规流程: {e}")
            return None

        if not answer:
            return None

        if "NEED_FRESH_DATA" in answer.upper():
            return None

        return answer
    
    def _intent_node(self, state: MasterAgentState) -> MasterAgentState:
        """意图识别节点（支持6种意图）- 使用规则+LLM混合策略"""
        question = state["user_question"]
        user_id = state["metadata"].get("user_id")
        
        # 先用规则快速判断意图
        rule_intent = self._rule_based_intent(question)
        
        # 如果规则能明确判断（不是simple_answer），直接使用
        if rule_intent != "simple_answer":
            state["intent"] = rule_intent
            state["metadata"]["intent_source"] = "rule"
            return state
        
        # 规则无法明确判断，调用LLM识别
        conversation_history = self._get_conversation_history(state)
        
        user_context = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=3)
                preferences = self.long_term_memory.get_all_preferences(user_id)
                user_context = self._format_long_term_context(knowledge, preferences)
            except Exception as e:
                logger.warning(f"获取长期记忆失败: {e}")
        
        prompt = get_master_intent_prompt(question, conversation_history, user_context)
        
        try:
            response = self._llm_to_str(self.llm.invoke(prompt)).strip()
            intent = response.lower().strip()
            
            valid_intents = [
                "simple_answer", "sql_only", "analysis_only",
                "sql_and_analysis", "web_search", "search_and_sql"
            ]
            if intent not in valid_intents:
                for valid_intent in valid_intents:
                    if valid_intent in intent:
                        intent = valid_intent
                        break
                else:
                    intent = rule_intent

            state["intent"] = intent
            state["metadata"]["intent_response"] = response
            state["metadata"]["intent_source"] = "llm"
            
        except Exception as e:
            fallback_intent = rule_intent
            state["intent"] = fallback_intent
            state["metadata"]["intent_error"] = str(e)
            state["metadata"]["intent_source"] = "fallback"
        
        return state
    
    def _route_after_intent(self, state: MasterAgentState) -> str:
        """意图识别后的路由（支持6种意图）"""
        intent = state.get("intent", "simple_answer")
        # 如果 web_search/search_and_sql 但搜索不可用，降级为 simple_answer
        if intent in ("web_search", "search_and_sql") and not self.search_agent.available:
            logger.warning("[路由] 搜索智能体不可用，意图降级为 simple_answer")
            state["final_answer"] = (
                "联网搜索功能暂未启用。请配置 TAVILY_API_KEY 环境变量后重启系统。\n"
                "申请地址：https://tavily.com（免费账户即可）"
            )
            return "simple_answer"
        return intent
    
    def _simple_answer_node(self, state: MasterAgentState) -> MasterAgentState:
        """简单回答节点"""
        question = state["user_question"]

        answer = self._simple_answer_text(question)
        
        state["final_answer"] = answer
        
        # 将AI回答添加到messages中
        state["messages"] = state["messages"] + [AIMessage(content=answer)]
        
        return state
    
    def _call_sql_node(self, state: MasterAgentState) -> MasterAgentState:
        """调用SQL查询子智能体"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        try:
            result = self.sql_agent.query(question)
            state["sql_result"] = result
            state["metadata"]["sql_result"] = result
            
            # 保存到会话数据存储
            self._remember_last_sql_result(thread_id, result)
            
        except Exception as e:
            state["error"] = f"SQL查询失败: {str(e)}"
            state["sql_result"] = {"error": str(e)}
        
        return state
    
    def _call_analysis_node(self, state: MasterAgentState) -> MasterAgentState:
        """调用数据分析子智能体"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        # 从会话数据存储中获取最近的查询结果
        data_to_analyze = None
        
        # 首先检查当前state中是否有查询结果
        if state.get("sql_result") and "data" in state["sql_result"]:
            data_to_analyze = state["sql_result"]["data"]
        # 否则从会话数据存储中获取历史查询结果
        elif thread_id in self.session_data and "last_sql_result" in self.session_data[thread_id]:
            last_sql_result = self.session_data[thread_id]["last_sql_result"]
            if last_sql_result and "data" in last_sql_result:
                data_to_analyze = last_sql_result["data"]
        
        if not data_to_analyze:
            state["error"] = "没有找到可以分析的数据。请先进行数据查询。"
            state["analysis_result"] = {"error": "无可用数据"}
            return state
        
        try:
            result = self.analysis_agent.analyze(data_to_analyze, question)
            state["analysis_result"] = result
            state["metadata"]["analysis_result"] = result
        except Exception as e:
            state["error"] = f"数据分析失败: {str(e)}"
            state["analysis_result"] = {"error": str(e)}
        
        return state
    
    def _call_both_node(self, state: MasterAgentState) -> MasterAgentState:
        """并行执行SQL查询和数据分析"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                sql_future = pool.submit(self.sql_agent.query, question)
                sql_result = sql_future.result()
            
            state["sql_result"] = sql_result
            state["metadata"]["sql_result"] = sql_result
            
            self._remember_last_sql_result(thread_id, sql_result)
            
            if sql_result.get("error"):
                state["error"] = f"SQL查询失败: {sql_result['error']}"
                return state
            
            if sql_result.get("data"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    analysis_future = pool.submit(self.analysis_agent.analyze, sql_result["data"], question)
                    analysis_result = analysis_future.result()
                
                state["analysis_result"] = analysis_result
                state["metadata"]["analysis_result"] = analysis_result
                
                if analysis_result.get("error"):
                    state["metadata"]["analysis_warning"] = analysis_result["error"]
            else:
                state["error"] = "查询结果为空，无法进行分析"
                
        except Exception as e:
            state["error"] = f"执行失败: {str(e)}"
        
        return state
    
    def _call_web_search_node(self, state: MasterAgentState) -> MasterAgentState:
        """联网搜索节点（纯搜索模式）"""
        question = state["user_question"]
        
        try:
            search_result = self.search_agent.search(question)
            state["search_result"] = search_result
            state["metadata"]["search_result"] = search_result
            
            if search_result.get("error"):
                state["error"] = search_result["error"]
                
        except Exception as e:
            state["error"] = f"联网搜索失败: {str(e)}"
        
        return state
    
    def _call_search_and_sql_node(self, state: MasterAgentState) -> MasterAgentState:
        """联网搜索 + 数据库查询联合分析节点"""
        question = state["user_question"]
        thread_id = state["metadata"].get("thread_id", "default")
        
        try:
            # 先查数据库
            sql_result, search_result = self._run_search_and_sql_parallel(question)
            state["sql_result"] = sql_result
            state["metadata"]["sql_result"] = sql_result
            
            self._remember_last_sql_result(thread_id, sql_result)
            
            # 再联网搜索 + 联合分析
            state["search_result"] = search_result
            state["metadata"]["search_result"] = search_result
            
            if search_result.get("error") and not sql_result.get("error"):
                state["error"] = search_result["error"]
                
        except Exception as e:
            state["error"] = f"联合分析失败: {str(e)}"
        
        return state
    
    def _summarize_node(self, state: MasterAgentState) -> MasterAgentState:
        """汇总结果节点（支持搜索结果和图表元数据）"""
        question = state["user_question"]
        intent = state.get("intent", "sql_only")
        
        # 预设回答已经生成（如降级处理）
        if state.get("final_answer"):
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        if state.get("error"):
            state["final_answer"] = f"抱歉，处理过程中出现错误：{state['error']}"
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        search_result = state.get("search_result")
        
        # 联网搜索相关意图：搜索智能体已生成完整回答
        if intent in ("web_search", "search_and_sql") and search_result:
            if search_result.get("error"):
                state["final_answer"] = f"搜索出错：{search_result['error']}"
            else:
                answer = search_result.get("answer", "未能获取搜索结果")
                sources = search_result.get("sources", [])
                if sources:
                    sources_text = "\n\n**参考来源：**\n" + "\n".join(
                        f"- {url}" for url in sources[:5]
                    )
                    answer = answer + sources_text
                state["final_answer"] = answer
                # 将图表元数据附加在 metadata 中供前端使用
                if search_result.get("chart"):
                    state["metadata"]["chart"] = search_result["chart"]
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
            return state
        
        # 数据库查询/分析相关意图
        sql_data = None
        analysis_data = None
        
        if sql_result:
            if sql_result.get("error"):
                state["final_answer"] = f"查询出错：{sql_result['error']}"
                state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
                return state
            sql_data = sql_result.get("data")
        
        if analysis_result:
            if analysis_result.get("error"):
                # 如果已有 SQL 数据，降级为仅基于 SQL 回答
                if sql_data:
                    state["metadata"]["analysis_warning"] = analysis_result["error"]
                    analysis_data = None
                else:
                    state["final_answer"] = f"分析出错：{analysis_result['error']}"
                    state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
                    return state
            analysis_data = analysis_result.get("analysis")
            # 将图表配置存入 metadata，流式接口和前端可读取
            if analysis_result.get("chart"):
                state["metadata"]["chart"] = analysis_result["chart"]
        
        try:
            prompt = get_summary_prompt(
                question=question,
                sql_result=sql_data,
                analysis_result=analysis_data
            )
            
            answer = self._llm_to_str(self.llm.invoke(prompt))
            state["final_answer"] = answer
            state["messages"] = list(state["messages"]) + [AIMessage(content=answer)]
            
        except Exception as e:
            state["final_answer"] = self._fallback_summary(question, sql_data, analysis_data)
            state["metadata"]["summary_warning"] = str(e)
            state["messages"] = list(state["messages"]) + [AIMessage(content=state["final_answer"])]
        
        return state
    
    def query(self, question: str, thread_id: str = "default", user_id: Optional[str] = None) -> str:
        """执行查询
        
        Args:
            question: 用户问题
            thread_id: 线程ID，用于区分不同的会话
            user_id: 用户ID，用于长期记忆
            
        Returns:
            回答结果
        """
        logger.info(f"Master Agent 收到问题: {question[:50]}...")
        
        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存，直接返回结果")
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     return cached_answer
        
        # 记忆查找
        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆，跳过 SQL/分析流程")
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return memory_answer

        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆，跳过 SQL/分析流程")
            yield sse("status", message="命中记忆，直接回答...")
            for char in memory_answer:
                yield sse("chunk", content=char)
                time.sleep(0.01)
            yield sse("done", answer=memory_answer)
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return

        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存，直接返回结果")
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     return cached_answer
        
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_messages = self._get_checkpoint_messages(config)
        initial_messages = [HumanMessage(content=question)]
        if not checkpoint_messages:
            persisted_messages = self._get_persisted_messages(thread_id)
            if persisted_messages:
                initial_messages = persisted_messages + initial_messages

        initial_state = {
            "messages": initial_messages,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # 使用checkpointer保存会话状态
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info("Master Agent 正在处理...")
        final_state = self.graph.invoke(initial_state, config)
        
        answer = final_state.get("final_answer", "抱歉，无法处理你的问题。")
        
        # 获取完整的对话历史（已经包含了当前的问题和回答）
        all_messages = list(final_state["messages"])
        self._save_conversation_messages(thread_id, all_messages)
        
        logger.info(f"Master Agent 处理完成，回答长度: {len(answer)} 字符")
        logger.info(f"Master Agent 当前会话共有 {len(all_messages)} 条消息")
        
        # 自动提取并保存长期记忆
        if user_id:
            self._extract_and_save_memory(all_messages, user_id)
        
        # 缓存结果
        self._remember_answer_fact(
            thread_id,
            question,
            answer,
            intent=final_state.get("intent"),
            user_id=user_id,
            sql_result=final_state.get("sql_result"),
        )

        # # 添加到缓存
        # self._add_to_cache(
        #     question,
        #     answer,
        #     intent=final_state.get("intent"),
        #     sql_result=final_state.get("sql_result"),
        #     analysis_result=final_state.get("analysis_result"),
        #     search_result=final_state.get("search_result"),
        # )
        
        return answer
    
    def _extract_and_save_memory(self, messages: Sequence[BaseMessage], user_id: str):
        """自动提取并保存长期记忆"""
        try:
            if not self.memory_extractor.should_extract(messages, threshold=0):
                return
            
            preferences = self.memory_extractor.extract_preferences_from_conversation(
                messages, user_id
            )
            for key, value in preferences.items():
                self.long_term_memory.save_preference(user_id, key, str(value))
                logger.info(f"保存偏好: {key} = {value}")
            
            knowledge_list = self.memory_extractor.extract_knowledge_from_conversation(
                messages, user_id
            )
            for knowledge in knowledge_list:
                self.long_term_memory.save_knowledge(
                    user_id,
                    knowledge.get("category", "其他"),
                    knowledge.get("content", ""),
                    knowledge.get("confidence", 0.8)
                )
                logger.info(f"保存知识: {knowledge.get('category')} - {knowledge.get('content', '')[:50]}...")
        except Exception as e:
            logger.warning(f"提取记忆失败: {e}")
    
    def stream_query(
        self,
        question: str,
        thread_id: str = "default",
        user_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """流式查询，以 SSE 格式生成事件流
        
        使用 LangGraph 的 graph.stream() 在每个节点完成后推送状态更新，
        最终 LLM 汇总回答以流式方式逐字输出。
        
        Yields:
            SSE 格式字符串：data: {...}\n\n
        """
        def sse(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs}, ensure_ascii=False)}\n\n"
        
        # # 缓存检查
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存，直接返回结果")
        #     for char in cached_answer:
        #         yield sse("chunk", content=char)
        #         time.sleep(0.01)
        #     yield sse("done", answer=cached_answer)
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     return
        
        # --- 意图识别（直接调用，以便立即推送状态）---
        yield sse("status", message="正在识别问题意图...")
        
        user_context = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=3)
                preferences = self.long_term_memory.get_all_preferences(user_id)
                user_context = self._format_long_term_context(knowledge, preferences)
            except Exception:
                pass
        
        # 从 checkpointer 获取对话历史
        config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = self.graph.get_state(config)
            existing_msgs = list(snapshot.values.get("messages", []))
        except Exception:
            existing_msgs = []
        if not existing_msgs:
            existing_msgs = self._get_persisted_messages(thread_id)
        
        temp_state: MasterAgentState = {
            "messages": existing_msgs,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {"thread_id": thread_id, "user_id": user_id}
        }
        conversation_history = self._get_conversation_history(temp_state)
        
        intent_prompt = get_master_intent_prompt(question, conversation_history, user_context)
        try:
            raw = self.llm.invoke(intent_prompt)
            intent_raw = self._llm_to_str(raw).strip().lower()
            logger.info(f"[意图识别] LLM 原始返回: {repr(intent_raw)}")
            valid_intents = [
                "simple_answer", "sql_only", "analysis_only",
                "sql_and_analysis", "web_search", "search_and_sql"
            ]
            intent = "sql_only"
            for vi in valid_intents:
                if vi in intent_raw:
                    intent = vi
                    break
            if intent == "simple_answer":
                rule_intent = self._rule_based_intent(question)
                if rule_intent != "simple_answer":
                    intent = rule_intent
        except Exception as e:
            intent = self._rule_based_intent(question)
            err_msg = f"{type(e).__name__}: {e}"
            logger.warning(f"[意图识别] LLM 调用失败: {err_msg}")
            # 尝试获取更详细的错误信息（如 API 配额不足等）
            if "FreeTierOnly" in str(e) or "Quota" in str(e):
                err_msg = "通义千问 API 免费额度已用完，请在控制台开通付费或更换模型。"
            elif "InvalidApiKey" in str(e) or "Unauthorized" in str(e):
                err_msg = "OPENAI_API_KEY 无效，请检查 API Key 是否正确。"
            yield sse("error", message=f"LLM 调用失败，已使用规则路由兜底: {err_msg}")
        
        yield sse("intent", intent=intent)
        
        # --- 搜索不可用时降级 ---
        if intent in ("web_search", "search_and_sql") and not self.search_agent.available:
            msg = "联网搜索功能暂未启用，请配置 TAVILY_API_KEY 环境变量后重启。申请地址：https://tavily.com"
            yield sse("chunk", content=msg)
            yield sse("done", answer=msg)
            return
        
        sql_result = None
        analysis_result = None
        search_result = None
        final_answer = ""
        
        # --- 执行各子任务 ---
        if intent == "simple_answer":
            yield sse("status", message="正在生成回答...")
            final_answer = self._simple_answer_text(question)
            yield sse("chunk", content=final_answer)
        
        else:
            # SQL 查询（适用于 sql_only / sql_and_analysis / search_and_sql）
            if intent in ("sql_only", "sql_and_analysis"):
                yield sse("status", message="正在查询数据库...")
                sql_result = self.sql_agent.query(question)
                
                if sql_result.get("sql"):
                    yield sse(
                        "sql",
                        sql=sql_result["sql"],
                        retry_count=sql_result.get("retry_count", 0)
                    )
                if sql_result.get("error"):
                    yield sse("error", message=f"数据库查询出错: {sql_result['error']}")
                
                # 保存会话数据
                self._remember_last_sql_result(thread_id, sql_result)
            
            # 数据分析（适用于 analysis_only / sql_and_analysis）
            if intent in ("analysis_only", "sql_and_analysis"):
                yield sse("status", message="正在分析数据...")
                
                data_to_analyze = None
                if sql_result and sql_result.get("data"):
                    data_to_analyze = sql_result["data"]
                elif thread_id in self.session_data:
                    last = self.session_data[thread_id].get("last_sql_result", {})
                    data_to_analyze = last.get("data") if last else None
                
                if data_to_analyze:
                    analysis_result = self.analysis_agent.analyze(data_to_analyze, question)
                    if analysis_result.get("chart"):
                        yield sse("chart", config=analysis_result["chart"])
                else:
                    yield sse("error", message="没有可分析的数据，请先执行数据查询")
            
            # 纯联网搜索
            if intent == "web_search":
                yield sse("status", message="正在联网搜索...")
                search_result = self.search_agent.search(question)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])
            
            # 搜索 + 数据库联合分析
            if intent == "search_and_sql":
                yield sse("status", message="正在联网搜索行业数据...")
                sql_result, search_result = self._run_search_and_sql_parallel(question)
                if sql_result.get("sql"):
                    yield sse(
                        "sql",
                        sql=sql_result["sql"],
                        retry_count=sql_result.get("retry_count", 0)
                    )
                if sql_result.get("error"):
                    yield sse("error", message=f"数据库查询出错: {sql_result['error']}")
                self._remember_last_sql_result(thread_id, sql_result)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])
            
            # --- 生成最终回答（流式输出 LLM 结果）---
            yield sse("status", message="正在生成回答...")
            
            if intent in ("web_search", "search_and_sql") and search_result:
                # 搜索智能体已生成完整回答，直接流式输出
                answer_text = search_result.get("answer", "未能获取搜索结果")
                if search_result.get("error"):
                    answer_text = f"搜索出错：{search_result['error']}"
                else:
                    sources = search_result.get("sources", [])
                    if sources:
                        answer_text += "\n\n**参考来源：**\n" + "\n".join(
                            f"- {url}" for url in sources[:5]
                        )
                final_answer = answer_text
                yield sse("chunk", content=final_answer)
            
            elif sql_result and sql_result.get("error"):
                final_answer = f"数据库查询出错：{sql_result['error']}"
                yield sse("chunk", content=final_answer)
            
            else:
                # 使用 LLM 流式生成汇总回答
                sql_data = sql_result.get("data") if sql_result else None
                analysis_data = analysis_result.get("analysis") if analysis_result else None
                
                summary_prompt = get_summary_prompt(
                    question=question,
                    sql_result=sql_data,
                    analysis_result=analysis_data
                )
                
                try:
                    import re
                    in_think = False
                    think_buffer = ""
                    for chunk in self.llm.stream(summary_prompt):
                        if isinstance(chunk, str):
                            chunk_text = chunk
                        elif hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                        elif hasattr(chunk, 'text'):
                            chunk_text = chunk.text
                        else:
                            chunk_text = str(chunk)
                        
                        # 过滤 <think>...</think> 思考内容，不发送给前端
                        think_buffer += chunk_text
                        if '<think>' in think_buffer and not in_think:
                            in_think = True
                        if in_think:
                            if '</think>' in think_buffer:
                                cleaned = re.sub(r'<think>[\s\S]*?</think>', '', think_buffer).strip()
                                if cleaned:
                                    final_answer += cleaned
                                    yield sse("chunk", content=cleaned)
                                think_buffer = ""
                                in_think = False
                            continue
                        
                        think_buffer = ""
                        final_answer += chunk_text
                        yield sse("chunk", content=chunk_text)
                except Exception as e:
                    final_answer = self._fallback_summary(question, sql_data, analysis_data)
                    yield sse("chunk", content=final_answer)
        
        yield sse("done", answer=final_answer)
        
        # 缓存结果
        self._remember_answer_fact(
            thread_id,
            question,
            final_answer,
            intent=intent,
            user_id=user_id,
            sql_result=sql_result,
        )

        # # 添加到缓存
        # self._add_to_cache(
        #     question,
        #     final_answer,
        #     intent=intent,
        #     sql_result=sql_result,
        #     analysis_result=analysis_result,
        #     search_result=search_result,
        # )
        
        # --- 保存对话历史到 LangGraph checkpointer ---
        try:
            new_messages = [HumanMessage(content=question), AIMessage(content=final_answer)]
            self.graph.update_state(
                config,
                {"messages": new_messages},
                as_node="summarize"
            )
            all_msgs = existing_msgs + new_messages
            self._save_conversation_messages(thread_id, all_msgs)
            if user_id:
                self._extract_and_save_memory(all_msgs, user_id)
        except Exception as e:
            logger.warning(f"保存对话历史失败（不影响本次回答）: {e}")

    def query(self, question: str, thread_id: str = "default", user_id: Optional[str] = None) -> str:
        """Final query implementation with memory-first answering."""
        logger.info(f"Master Agent query: {question[:50]}...")

        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆，跳过 SQL/分析流程")
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return memory_answer

        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存")
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     self._remember_answer_fact(
        #         thread_id,
        #         question,
        #         cached_answer,
        #         intent="cache_answer",
        #         user_id=user_id,
        #     )
        #     return cached_answer

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_messages = self._get_checkpoint_messages(config)
        initial_messages = [HumanMessage(content=question)]
        if not checkpoint_messages:
            persisted_messages = self._get_persisted_messages(thread_id)
            if persisted_messages:
                initial_messages = persisted_messages + initial_messages

        initial_state = {
            "messages": initial_messages,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
        }

        final_state = self.graph.invoke(initial_state, config)
        answer = final_state.get("final_answer", "未能生成有效回答")
        all_messages = list(final_state["messages"])
        self._save_conversation_messages(thread_id, all_messages)

        if user_id:
            self._extract_and_save_memory(all_messages, user_id)

        self._remember_answer_fact(
            thread_id,
            question,
            answer,
            intent=final_state.get("intent"),
            user_id=user_id,
            sql_result=final_state.get("sql_result"),
        )

        # # 添加到缓存
        # self._add_to_cache(
        #     question,
        #     answer,
        #     intent=final_state.get("intent"),
        #     sql_result=final_state.get("sql_result"),
        #     analysis_result=final_state.get("analysis_result"),
        #     search_result=final_state.get("search_result"),
        # )

        return answer

    def stream_query(
        self,
        question: str,
        thread_id: str = "default",
        user_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Final stream implementation with memory-first answering."""

        def sse(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs}, ensure_ascii=False)}\n\n"

        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆（流式），跳过 SQL/分析流程")
            yield sse("status", message="命中记忆，直接回答...")
            for char in memory_answer:
                yield sse("chunk", content=char)
                time.sleep(0.01)
            yield sse("done", answer=memory_answer)
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return

        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存（流式）")
        #     yield sse("status", message="命中缓存，直接回答...")
        #     for char in cached_answer:
        #         yield sse("chunk", content=char)
        #         time.sleep(0.01)
        #     yield sse("done", answer=cached_answer)
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     return

        yield sse("status", message="正在处理问题...")
        answer = self.query(question, thread_id=thread_id, user_id=user_id)
        yield sse("intent", intent="memory_or_pipeline")
        for char in answer:
            yield sse("chunk", content=char)
            time.sleep(0.01)
        yield sse("done", answer=answer)

    def query(self, question: str, thread_id: str = "default", user_id: Optional[str] = None) -> str:
        """Final query implementation with memory-first answering."""
        logger.info(f"Master Agent query: {question[:50]}...")

        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆，跳过 SQL/分析流程")
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return memory_answer

        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存")
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     self._remember_answer_fact(
        #         thread_id,
        #         question,
        #         cached_answer,
        #         intent="cache_answer",
        #         user_id=user_id,
        #     )
        #     return cached_answer

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_messages = self._get_checkpoint_messages(config)
        initial_messages = [HumanMessage(content=question)]
        if not checkpoint_messages:
            persisted_messages = self._get_persisted_messages(thread_id)
            if persisted_messages:
                initial_messages = persisted_messages + initial_messages

        initial_state = {
            "messages": initial_messages,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {
                "thread_id": thread_id,
                "user_id": user_id,
            },
        }

        final_state = self.graph.invoke(initial_state, config)
        answer = final_state.get("final_answer", "未能生成有效回答")
        all_messages = list(final_state["messages"])
        self._save_conversation_messages(thread_id, all_messages)

        if user_id:
            self._extract_and_save_memory(all_messages, user_id)

        self._remember_answer_fact(
            thread_id,
            question,
            answer,
            intent=final_state.get("intent"),
            user_id=user_id,
            sql_result=final_state.get("sql_result"),
        )

        # # 添加到缓存
        # self._add_to_cache(
        #     question,
        #     answer,
        #     intent=final_state.get("intent"),
        #     sql_result=final_state.get("sql_result"),
        #     analysis_result=final_state.get("analysis_result"),
        #     search_result=final_state.get("search_result"),
        # )

        return answer

    def stream_query(
        self,
        question: str,
        thread_id: str = "default",
        user_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Final stream implementation with memory-first answering."""

        def sse(type_: str, **kwargs) -> str:
            return f"data: {json.dumps({'type': type_, **kwargs}, ensure_ascii=False)}\n\n"

        memory_answer = self._try_answer_from_memory(question, thread_id, user_id)
        if memory_answer:
            logger.info("命中记忆（流式），跳过 SQL/分析流程")
            yield sse("status", message="命中记忆，直接回答...")
            for char in memory_answer:
                yield sse("chunk", content=char)
                time.sleep(0.01)
            yield sse("done", answer=memory_answer)
            self._remember_conversation_turn(thread_id, question, memory_answer)
            self._remember_answer_fact(
                thread_id,
                question,
                memory_answer,
                intent="memory_answer",
                user_id=user_id,
            )
            return

        # # 缓存查找
        # cached_answer = self._find_cached_answer(question)
        # if cached_answer:
        #     logger.info("命中缓存（流式）")
        #     yield sse("status", message="命中缓存，直接回答...")
        #     for char in cached_answer:
        #         yield sse("chunk", content=char)
        #         time.sleep(0.01)
        #     yield sse("done", answer=cached_answer)
        #     self._remember_conversation_turn(thread_id, question, cached_answer)
        #     self._remember_answer_fact(
        #         thread_id,
        #         question,
        #         cached_answer,
        #         intent="cache_answer",
        #         user_id=user_id,
        #     )
        #     return

        yield sse("status", message="正在识别问题意图...")

        user_context = ""
        if user_id:
            try:
                knowledge = self.long_term_memory.get_relevant_knowledge(user_id, question, top_k=3)
                preferences = self.long_term_memory.get_all_preferences(user_id)
                user_context = self._format_long_term_context(knowledge, preferences)
            except Exception as e:
                logger.warning(f"Load long-term context failed: {e}")

        config = {"configurable": {"thread_id": thread_id}}
        existing_msgs = self._get_checkpoint_messages(config)
        if not existing_msgs:
            existing_msgs = self._get_persisted_messages(thread_id)

        temp_state: MasterAgentState = {
            "messages": existing_msgs,
            "user_question": question,
            "intent": None,
            "sql_result": None,
            "analysis_result": None,
            "search_result": None,
            "final_answer": None,
            "error": None,
            "metadata": {"thread_id": thread_id, "user_id": user_id},
        }
        conversation_history = self._get_conversation_history(temp_state)

        intent_prompt = get_master_intent_prompt(question, conversation_history, user_context)
        try:
            raw = self.llm.invoke(intent_prompt)
            intent_raw = self._llm_to_str(raw).strip().lower()
            valid_intents = [
                "simple_answer",
                "sql_only",
                "analysis_only",
                "sql_and_analysis",
                "web_search",
                "search_and_sql",
            ]
            intent = "sql_only"
            for valid_intent in valid_intents:
                if valid_intent in intent_raw:
                    intent = valid_intent
                    break
            if intent == "simple_answer":
                rule_intent = self._rule_based_intent(question)
                if rule_intent != "simple_answer":
                    intent = rule_intent
        except Exception as e:
            intent = self._rule_based_intent(question)
            yield sse("error", message=f"意图识别失败，已回退规则路由: {type(e).__name__}: {e}")

        yield sse("intent", intent=intent)

        if intent in ("web_search", "search_and_sql") and not self.search_agent.available:
            final_answer = "联网搜索功能暂未启用，请先配置 TAVILY_API_KEY。"
            yield sse("chunk", content=final_answer)
            yield sse("done", answer=final_answer)
            self._remember_conversation_turn(thread_id, question, final_answer)
            return

        sql_result = None
        analysis_result = None
        search_result = None
        final_answer = ""

        if intent == "simple_answer":
            yield sse("status", message="正在生成回答...")
            final_answer = self._simple_answer_text(question)
            yield sse("chunk", content=final_answer)
        else:
            if intent in ("sql_only", "sql_and_analysis"):
                yield sse("status", message="正在查询数据库...")
                sql_result = self.sql_agent.query(question)
                if sql_result.get("sql"):
                    yield sse(
                        "sql",
                        sql=sql_result["sql"],
                        retry_count=sql_result.get("retry_count", 0),
                    )
                if sql_result.get("error"):
                    yield sse("error", message=f"数据库查询出错: {sql_result['error']}")
                self._remember_last_sql_result(thread_id, sql_result)

            if intent in ("analysis_only", "sql_and_analysis"):
                yield sse("status", message="正在分析数据...")
                data_to_analyze = None
                if sql_result and sql_result.get("data"):
                    data_to_analyze = sql_result["data"]
                elif thread_id in self.session_data:
                    last_result = self.session_data[thread_id].get("last_sql_result", {})
                    if isinstance(last_result, dict):
                        data_to_analyze = last_result.get("data")

                if data_to_analyze:
                    analysis_result = self.analysis_agent.analyze(data_to_analyze, question)
                    if analysis_result.get("chart"):
                        yield sse("chart", config=analysis_result["chart"])
                    if analysis_result.get("error"):
                        yield sse("error", message=f"数据分析出错: {analysis_result['error']}")
                else:
                    yield sse("error", message="没有可分析的数据，请先执行相关查询。")

            if intent == "web_search":
                yield sse("status", message="正在联网搜索...")
                search_result = self.search_agent.search(question)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])

            if intent == "search_and_sql":
                yield sse("status", message="正在执行搜索与数据库联合分析...")
                sql_result, search_result = self._run_search_and_sql_parallel(question)
                if sql_result.get("sql"):
                    yield sse(
                        "sql",
                        sql=sql_result["sql"],
                        retry_count=sql_result.get("retry_count", 0),
                    )
                if sql_result.get("error"):
                    yield sse("error", message=f"数据库查询出错: {sql_result['error']}")
                self._remember_last_sql_result(thread_id, sql_result)
                if search_result.get("sources"):
                    yield sse("sources", sources=search_result["sources"])
                if search_result.get("error"):
                    yield sse("error", message=search_result["error"])

            yield sse("status", message="正在生成回答...")

            if intent in ("web_search", "search_and_sql") and search_result:
                if search_result.get("error"):
                    final_answer = f"搜索出错：{search_result['error']}"
                else:
                    final_answer = search_result.get("answer", "未能获取搜索结果")
                    sources = search_result.get("sources", [])
                    if sources:
                        final_answer += "\n\n参考来源：\n" + "\n".join(f"- {url}" for url in sources[:5])
                yield sse("chunk", content=final_answer)
            elif sql_result and sql_result.get("error"):
                final_answer = f"数据库查询出错：{sql_result['error']}"
                yield sse("chunk", content=final_answer)
            else:
                sql_data = sql_result.get("data") if sql_result else None
                analysis_data = analysis_result.get("analysis") if analysis_result else None
                summary_prompt = get_summary_prompt(
                    question=question,
                    sql_result=sql_data,
                    analysis_result=analysis_data,
                )

                try:
                    think_buffer = ""
                    in_think = False
                    for chunk in self.llm.stream(summary_prompt):
                        if isinstance(chunk, str):
                            chunk_text = chunk
                        elif hasattr(chunk, "content"):
                            chunk_text = str(chunk.content)
                        elif hasattr(chunk, "text"):
                            chunk_text = str(chunk.text)
                        else:
                            chunk_text = str(chunk)

                        think_buffer += chunk_text
                        if "<think>" in think_buffer and not in_think:
                            in_think = True
                        if in_think:
                            if "</think>" in think_buffer:
                                cleaned = re.sub(r"<think>[\s\S]*?</think>", "", think_buffer).strip()
                                if cleaned:
                                    final_answer += cleaned
                                    yield sse("chunk", content=cleaned)
                                think_buffer = ""
                                in_think = False
                            continue

                        think_buffer = ""
                        final_answer += chunk_text
                        yield sse("chunk", content=chunk_text)
                except Exception as e:
                    logger.warning(f"Stream summary failed, fallback summary used: {e}")
                    final_answer = self._fallback_summary(question, sql_data, analysis_data)
                    yield sse("chunk", content=final_answer)

        yield sse("done", answer=final_answer)

        self._remember_answer_fact(
            thread_id,
            question,
            final_answer,
            intent=intent,
            user_id=user_id,
            sql_result=sql_result,
        )

        # # 添加到缓存
        # self._add_to_cache(
        #     question,
        #     final_answer,
        #     intent=intent,
        #     sql_result=sql_result,
        #     analysis_result=analysis_result,
        #     search_result=search_result,
        # )

        try:
            new_messages = [HumanMessage(content=question), AIMessage(content=final_answer)]
            self.graph.update_state(
                config,
                {"messages": new_messages},
                as_node="summarize",
            )
            all_msgs = existing_msgs + new_messages
            self._save_conversation_messages(thread_id, all_msgs)
            if user_id:
                self._extract_and_save_memory(all_msgs, user_id)
        except Exception as e:
            logger.warning(f"Save stream conversation failed: {e}")
