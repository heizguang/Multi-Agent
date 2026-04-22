"""
SQL查询子智能体

负责将自然语言转换为SQL并执行查询，支持自动纠错循环（最多3次重试）。
Reflection 模式：执行失败时将错误信息反馈给 LLM 重新生成。
"""

import json
import re
import sqlite3
import sys
import asyncio
import concurrent.futures
from typing import Dict, Any
from pathlib import Path

from langchain_core.language_models import BaseLLM
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.append(str(Path(__file__).parent.parent))
from prompts import get_few_shot_prompt, get_sql_correction_prompt


class SQLQueryAgent:
    """SQL查询子智能体，支持自动纠错循环（ReAct/Reflection 模式）"""
    
    def __init__(self, llm: BaseLLM, db_path: str, num_examples: int = 3):
        """初始化SQL查询智能体
        
        Args:
            llm: 语言模型实例
            db_path: 数据库路径
            num_examples: Few-shot示例数量
        """
        self.llm = llm
        self.db_path = db_path
        self.num_examples = num_examples
    
    @staticmethod
    def _llm_to_str(result) -> str:
        """安全地从 LLM 返回值中提取文本，清理思考标签"""
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
    
    def _get_schema(self) -> str:
        """获取数据库Schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = cursor.fetchall()
        
        schema_text = ""
        for table in tables:
            table_name = table[0]
            schema_text += f"\n表：{table_name}\n"
            
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                cid, name, dtype, notnull, default, pk = col
                pk_text = " (主键)" if pk else ""
                notnull_text = " NOT NULL" if notnull else ""
                schema_text += f"  - {name}: {dtype}{notnull_text}{pk_text}\n"
        
        conn.close()
        return schema_text.strip()
    
    def _clean_sql(self, sql: str) -> str:
        """清理SQL语句（移除代码块标记和多余前缀）"""
        sql = sql.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        
        prefixes = ["SQL：", "SQL:", "sql:", "sql："]
        for prefix in prefixes:
            if sql.startswith(prefix):
                sql = sql[len(prefix):]
                break
        
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip()

    def _rule_based_sql(self, question: str) -> str:
        """当 LLM 不可用时，使用规则生成常见查询 SQL。"""
        q = question.strip()

        # 部门数量
        if "部门" in q and ("多少" in q or "几个" in q):
            if "每个" in q and ("城市" in q or "在哪" in q):
                return (
                    "SELECT dept_name, location "
                    "FROM departments "
                    "ORDER BY dept_id"
                )
            return "SELECT COUNT(*) AS department_count FROM departments"

        # 研发部人数
        if "研发部" in q and ("多少" in q or "几" in q) and "员工" in q:
            return (
                "SELECT COUNT(*) AS employee_count "
                "FROM employees e "
                "JOIN departments d ON e.dept_id = d.dept_id "
                "WHERE d.dept_name = '研发部'"
            )

        # 职位分布（研发部或全公司）
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

        # 薪资最高前 N 名
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

        # 基本工资超过某值
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

        # 指定部门平均薪资对比
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
        """生成SQL语句"""
        schema = self._get_schema()
        prompt = get_few_shot_prompt(
            question=question,
            schema=schema,
            num_examples=self.num_examples
        )
        sql = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(sql)
    
    def _correct_sql(self, question: str, original_sql: str, error_msg: str, attempt: int) -> str:
        """SQL 自动纠错（Reflection 模式）
        
        将错误信息和原始 SQL 反馈给 LLM，要求其重新生成正确的 SQL。
        
        Args:
            question: 用户原始问题
            original_sql: 出错的 SQL 语句
            error_msg: 错误信息
            attempt: 当前重试次数（从1开始）
            
        Returns:
            修正后的 SQL 语句
        """
        schema = self._get_schema()
        prompt = get_sql_correction_prompt(
            question=question,
            schema=schema,
            original_sql=original_sql,
            error_msg=error_msg,
            attempt=attempt
        )
        corrected = self._llm_to_str(self.llm.invoke(prompt)).strip()
        return self._clean_sql(corrected)
    
    async def _execute_sql_via_mcp(self, sql: str) -> str:
        """通过MCP工具执行SQL
        
        Args:
            sql: SQL语句
            
        Returns:
            查询结果JSON字符串
        """
        mcp_script = Path(__file__).parent.parent / "mcp_sql_server.py"
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_script)]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                result = await session.call_tool(
                    "execute_sql", 
                    arguments={"sql": sql}
                )
                
                if result.content:
                    return result.content[0].text
                return json.dumps({"error": "无返回结果"})
    
    def _run_async(self, coro):
        """安全地执行异步代码，兼容已有事件循环（如 httpx/openai 遗留的）"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)
    
    def query(self, question: str, max_retries: int = 3) -> Dict[str, Any]:
        """执行查询，失败时自动纠错并重试（Reflection 循环）
        
        流程：SQL生成 → 执行 → [失败] → 错误反馈给LLM → 重新生成 → 最多重试 max_retries 次
        
        Args:
            question: 用户问题
            max_retries: 最大重试次数（默认3次）
            
        Returns:
            {
                "sql": 最终执行的SQL,
                "data": 查询结果JSON字符串（成功时）,
                "error": 错误信息（成功时为None）,
                "retry_count": 实际重试次数（0表示首次成功）
            }
        """
        result = {
            "sql": None,
            "data": None,
            "error": None,
            "retry_count": 0
        }
        
        try:
            try:
                sql = self._generate_sql(question)
            except Exception as llm_error:
                sql = self._rule_based_sql(question)
                if not sql:
                    result["error"] = f"查询失败（LLM不可用，且未命中规则SQL）: {str(llm_error)}"
                    return result
            result["sql"] = sql
            
            if not sql:
                result["error"] = "未能生成有效的SQL"
                return result
            
            for attempt in range(max_retries):
                query_result = self._run_async(self._execute_sql_via_mcp(sql))
                result_data = json.loads(query_result)
                
                if isinstance(result_data, dict) and "error" in result_data:
                    error_msg = result_data["error"]
                    
                    if attempt < max_retries - 1:
                        print(f"[SQL纠错] 第{attempt + 1}次执行失败: {error_msg}，正在让LLM自动修复...")
                        try:
                            sql = self._correct_sql(question, sql, error_msg, attempt + 1)
                        except Exception:
                            fallback_sql = self._rule_based_sql(question)
                            if fallback_sql:
                                sql = fallback_sql
                            else:
                                result["error"] = f"SQL执行失败（自动修复不可用）: {error_msg}"
                                break
                        result["sql"] = sql
                        result["retry_count"] = attempt + 1
                    else:
                        result["error"] = f"SQL执行失败（已自动重试{attempt}次）: {error_msg}"
                else:
                    result["data"] = query_result
                    if attempt > 0:
                        print(f"[SQL纠错] 第{attempt}次修复后执行成功")
                    break
                    
        except Exception as e:
            result["error"] = f"查询失败: {str(e)}"
        
        return result

