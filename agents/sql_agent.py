"""
SQL 查询子智能体

负责将自然语言转换为 SQL 并执行查询，支持自动纠错与相关 schema 检索。
"""

import asyncio
import concurrent.futures
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import os
sys.path.append(str(Path(__file__).parent.parent))
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from langchain_core.language_models import BaseLLM
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from prompts import get_few_shot_prompt, get_sql_correction_prompt


class SQLQueryAgent:
    """SQL 查询子智能体，支持 Reflection 自纠错与 schema 智能召回。"""

    def __init__(self, llm: BaseLLM, db_path: str, num_examples: int = 3):
        self.llm = llm
        self.db_path = db_path
        self.num_examples = num_examples
        self._schema_cache: List[Dict[str, Any]] | None = None
        self._schema_selection_cache: Dict[str, str] = {}

    @staticmethod
    def _llm_to_str(result) -> str:
        import re

        if isinstance(result, str):
            text = result
        elif hasattr(result, "content"):
            text = str(result.content)
        elif hasattr(result, "text"):
            text = str(result.text)
        else:
            text = str(result)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = re.sub(r"</think>", "", text).strip()
        return text

    def _load_schema_cache(self) -> List[Dict[str, Any]]:
        if self._schema_cache is not None:
            return self._schema_cache

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        tables = cursor.fetchall()

        schema_items: List[Dict[str, Any]] = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            sample_values: Dict[str, List[str]] = {}
            for _, name, dtype, *_ in columns:
                dtype_text = str(dtype or "").upper()
                if not any(token in dtype_text for token in ("CHAR", "TEXT", "VARCHAR")):
                    continue
                try:
                    cursor.execute(
                        f"""
                        SELECT DISTINCT {name}
                        FROM {table_name}
                        WHERE {name} IS NOT NULL AND TRIM(CAST({name} AS TEXT)) != ''
                        LIMIT 3
                        """
                    )
                    sample_rows = cursor.fetchall()
                    sample_values[name] = [str(row[0]) for row in sample_rows if row and row[0] is not None]
                except Exception:
                    sample_values[name] = []
            schema_items.append(
                {
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": name,
                            "dtype": dtype,
                            "notnull": bool(notnull),
                            "pk": bool(pk),
                        }
                        for _, name, dtype, notnull, _, pk in columns
                    ],
                    "foreign_keys": [
                        {
                            "from": fk[3],
                            "to_table": fk[2],
                            "to_column": fk[4],
                        }
                        for fk in foreign_keys
                    ],
                    "sample_values": sample_values,
                }
            )

        conn.close()
        self._schema_cache = schema_items
        return schema_items

    def _format_schema(self, schema_items: List[Dict[str, Any]]) -> str:
        schema_text = ""
        for item in schema_items:
            schema_text += f"\n表：{item['table_name']}\n"
            for col in item["columns"]:
                pk_text = " (主键)" if col["pk"] else ""
                notnull_text = " NOT NULL" if col["notnull"] else ""
                schema_text += (
                    f"  - {col['name']}: {col['dtype']}{notnull_text}{pk_text}\n"
                )
                sample_values = item.get("sample_values", {}).get(col["name"], [])
                if sample_values:
                    schema_text += f"    示例值: {', '.join(sample_values)}\n"
            foreign_keys = item.get("foreign_keys", [])
            if foreign_keys:
                for fk in foreign_keys:
                    schema_text += (
                        f"  - 关联: {fk['from']} -> {fk['to_table']}.{fk['to_column']}\n"
                    )
        return schema_text.strip()

    def _extract_query_terms(self, question: str) -> List[str]:
        lowered = question.lower()
        ascii_terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", lowered)
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", question)

        synonyms = {
            "employees": ["员工", "人员", "入职", "岗位", "职位"],
            "departments": ["部门", "团队", "城市", "地点"],
            "salaries": ["薪资", "工资", "奖金", "薪酬", "收入"],
        }

        ordered_terms: List[str] = []
        seen = set()
        for term in ascii_terms + chinese_terms:
            clean = term.strip()
            if len(clean) < 2 or clean in seen:
                continue
            seen.add(clean)
            ordered_terms.append(clean)

        for table_name, alias_terms in synonyms.items():
            if any(alias in question for alias in alias_terms) and table_name not in seen:
                ordered_terms.append(table_name)
                seen.add(table_name)

        return ordered_terms

    def _score_schema_item(self, schema_item: Dict[str, Any], question: str) -> float:
        terms = self._extract_query_terms(question)
        if not terms:
            return 0.0

        table_name = schema_item["table_name"].lower()
        column_names = [col["name"].lower() for col in schema_item["columns"]]
        related_tables = [
            str(fk.get("to_table", "")).lower()
            for fk in schema_item.get("foreign_keys", [])
        ]
        sample_values = [
            str(value).lower()
            for values in schema_item.get("sample_values", {}).values()
            for value in values
        ]
        score = 0.0

        for term in terms:
            lowered = term.lower()
            if lowered == table_name or lowered in table_name:
                score += 2.5
            if any(lowered == col or lowered in col for col in column_names):
                score += 1.5
            if any(lowered and lowered in sample for sample in sample_values):
                score += 2.2
            if any(lowered == related or lowered in related for related in related_tables):
                score += 1.0

        if any(alias in question for alias in ["薪资", "工资", "奖金"]) and table_name == "salaries":
            score += 2.0
        if any(alias in question for alias in ["员工", "岗位", "职位"]) and table_name == "employees":
            score += 2.0
        if any(alias in question for alias in ["部门", "城市", "地点"]) and table_name == "departments":
            score += 2.0
        if any(city in question for city in ["北京", "上海", "广州", "深圳"]):
            if table_name == "departments":
                score += 1.5
        if any(dept in question for dept in ["研发部", "市场部", "产品部", "设计部", "销售部"]):
            if table_name in {"departments", "employees", "salaries"}:
                score += 1.5

        return score

    def _get_schema(self, question: str = "", max_tables: int = 4) -> str:
        schema_items = self._load_schema_cache()
        if not question:
            return self._format_schema(schema_items)

        cache_key = f"{question.strip().lower()}|{max_tables}"
        cached_schema = self._schema_selection_cache.get(cache_key)
        if cached_schema:
            return cached_schema

        scored_items = [
            (self._score_schema_item(item, question), item) for item in schema_items
        ]
        scored_items.sort(key=lambda pair: pair[0], reverse=True)

        selected_items = [item for score, item in scored_items if score > 0][:max_tables]
        if not selected_items:
            selected_items = schema_items[:max_tables]

        selected_names = {item["table_name"] for item in selected_items}
        if "employees" in selected_names and "departments" not in selected_names:
            for item in schema_items:
                if item["table_name"] == "departments":
                    selected_items.append(item)
                    selected_names.add("departments")
                    break
        if "employees" in selected_names and "salaries" not in selected_names and any(
            kw in question for kw in ["薪资", "工资", "奖金", "薪酬"]
        ):
            for item in schema_items:
                if item["table_name"] == "salaries":
                    selected_items.append(item)
                    break

        formatted_schema = self._format_schema(selected_items)
        self._schema_selection_cache[cache_key] = formatted_schema
        return formatted_schema

    def _clean_sql(self, sql: str) -> str:
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]

        prefixes = ["SQL：", "SQL:", "sql:", "sql："]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix) :]
                break

        if sql.endswith("```"):
            sql = sql[:-3]

        sql = sql.strip().rstrip(";").strip()
        return sql

    def _is_safe_sql(self, sql: str) -> bool:
        normalized = re.sub(r"\s+", " ", sql.strip()).lower()
        if not normalized:
            return False
        if ";" in normalized:
            return False
        if not (normalized.startswith("select ") or normalized.startswith("with ")):
            return False
        blocked = [
            " insert ",
            " update ",
            " delete ",
            " drop ",
            " alter ",
            " truncate ",
            " attach ",
            " detach ",
            " pragma ",
            " create ",
            " replace ",
        ]
        wrapped = f" {normalized} "
        return not any(token in wrapped for token in blocked)

    def _rule_based_sql(self, question: str) -> str:
        q = question.strip()

        if "部门" in q and ("多少" in q or "几个" in q):
            if "每个" in q and ("城市" in q or "在哪" in q):
                return "SELECT dept_name, location FROM departments ORDER BY dept_id"
            return "SELECT COUNT(*) AS department_count FROM departments"

        if "研发部" in q and ("多少" in q or "几" in q) and "员工" in q:
            return (
                "SELECT COUNT(*) AS employee_count "
                "FROM employees e "
                "JOIN departments d ON e.dept_id = d.dept_id "
                "WHERE d.dept_name = '研发部'"
            )

        if "职位分布" in q or "岗位分布" in q:
            dept_match = re.search(r"([\u4e00-\u9fa5]{1,8}部)", q)
            if dept_match:
                dept_name = dept_match.group(1)
                return (
                    "SELECT e.position, COUNT(*) AS employee_count "
                    "FROM employees e "
                    "JOIN departments d ON e.dept_id = d.dept_id "
                    f"WHERE d.dept_name = '{dept_name}' "
                    "GROUP BY e.position "
                    "ORDER BY employee_count DESC"
                )
            return (
                "SELECT position, COUNT(*) AS employee_count "
                "FROM employees "
                "GROUP BY position "
                "ORDER BY employee_count DESC"
            )

        if ("薪资最高" in q or "工资最高" in q) and ("员工" in q or "名" in q):
            top_n_match = re.search(r"(\d+)\s*名", q)
            top_n = int(top_n_match.group(1)) if top_n_match else 10
            return (
                "SELECT e.emp_name, d.dept_name, e.position, "
                "(s.base_salary + s.bonus) AS total_salary "
                "FROM employees e "
                "JOIN departments d ON e.dept_id = d.dept_id "
                "JOIN salaries s ON e.emp_id = s.emp_id "
                "ORDER BY total_salary DESC "
                f"LIMIT {top_n}"
            )

        if "基本工资" in q and ("超过" in q or "大于" in q):
            amount_match = re.search(r"(\d{4,})", q)
            amount = int(amount_match.group(1)) if amount_match else 30000
            return (
                "SELECT e.emp_name, d.dept_name, e.position, s.base_salary, s.bonus "
                "FROM employees e "
                "JOIN departments d ON e.dept_id = d.dept_id "
                "JOIN salaries s ON e.emp_id = s.emp_id "
                f"WHERE s.base_salary > {amount} "
                "ORDER BY s.base_salary DESC"
            )

        if "平均薪资" in q and ("对比" in q or "比较" in q):
            departments = re.findall(r"([\u4e00-\u9fa5]{1,8}部)", q)
            departments = list(dict.fromkeys(departments))
            if departments:
                in_values = ", ".join([f"'{d}'" for d in departments])
                return (
                    "SELECT d.dept_name, AVG(s.base_salary + s.bonus) AS avg_salary "
                    "FROM departments d "
                    "JOIN employees e ON d.dept_id = e.dept_id "
                    "JOIN salaries s ON e.emp_id = s.emp_id "
                    f"WHERE d.dept_name IN ({in_values}) "
                    "GROUP BY d.dept_name "
                    "ORDER BY avg_salary DESC"
                )

        return ""

    def _generate_sql(self, question: str) -> str:
        schema = self._get_schema(question=question)
        prompt = get_few_shot_prompt(
            question=question,
            schema=schema,
            num_examples=self.num_examples,
        )
        sql = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(sql)

    def _correct_sql(
        self, question: str, original_sql: str, error_msg: str, attempt: int
    ) -> str:
        schema = self._get_schema(question=question, max_tables=6)
        prompt = get_sql_correction_prompt(
            question=question,
            schema=schema,
            original_sql=original_sql,
            error_msg=error_msg,
            attempt=attempt,
        )
        corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(corrected)

    async def _execute_sql_via_mcp(self, sql: str) -> str:
        mcp_script = Path(__file__).parent.parent / "mcp_sql_server.py"
        server_params = StdioServerParameters(command=sys.executable, args=[str(mcp_script)])

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("execute_sql", arguments={"sql": sql})
                if result.content:
                    return result.content[0].text
                return json.dumps({"error": "无返回结果"})

    def _run_async(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def query(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        logger.info(f"SQL Agent 收到查询请求: {question[:50]}...")
        
        result = {
            "sql": None,
            "data": None,
            "error": None,
            "retry_count": 0,
        }

        try:
            try:
                logger.info("SQL Agent 正在生成 SQL...")
                sql = self._generate_sql(question)
                logger.info(f"SQL Agent 生成的 SQL: {sql[:100]}...")
            except Exception as llm_error:
                logger.warning("SQL Agent LLM 生成失败，尝试规则匹配...")
                sql = self._rule_based_sql(question)
                if not sql:
                    result["error"] = (
                        f"查询失败（LLM 不可用，且未命中规则 SQL）: {str(llm_error)}"
                    )
                    logger.error(f"SQL Agent 查询失败: {result['error']}")
                    return result

            result["sql"] = sql
            if not sql:
                result["error"] = "未能生成有效的 SQL"
                logger.error("SQL Agent 未能生成有效的 SQL")
                return result
            if not self._is_safe_sql(sql):
                result["error"] = "生成的 SQL 未通过安全校验，仅允许单条 SELECT / WITH 查询"
                logger.error(result["error"])
                return result

            for attempt in range(max_retries):
                logger.info(f"SQL Agent 执行 SQL (尝试 {attempt + 1}/{max_retries})...")
                query_result = self._run_async(self._execute_sql_via_mcp(sql))
                result_data = json.loads(query_result)

                if isinstance(result_data, dict) and "error" in result_data:
                    error_msg = result_data["error"]
                    logger.warning(f"SQL Agent SQL 执行错误: {error_msg}")
                    if attempt < max_retries - 1:
                        logger.info(f"SQL Agent 第{attempt + 1}次执行失败，正在让 LLM 自动修复...")
                        try:
                            sql = self._correct_sql(question, sql, error_msg, attempt + 1)
                        except Exception:
                            fallback_sql = self._rule_based_sql(question)
                            if fallback_sql:
                                sql = fallback_sql
                            else:
                                result["error"] = (
                                    f"SQL 执行失败（自动修复不可用）: {error_msg}"
                                )
                                break
                        result["sql"] = sql
                        result["retry_count"] = attempt + 1
                    else:
                        result["error"] = (
                            f"SQL 执行失败（已自动重试{attempt}次）: {error_msg}"
                        )
                else:
                    result["data"] = query_result
                    logger.info(f"SQL Agent 查询成功")
                    if attempt > 0:
                        logger.info(f"SQL Agent 第{attempt}次修复后执行成功")
                    break

        except Exception as e:
            result["error"] = f"查询失败: {str(e)}"
            logger.error(f"SQL Agent 异常: {str(e)}")

        return result
