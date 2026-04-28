"""
基于FastMCP的SQL查询工具服务器

提供简单的SQL执行功能，方便扩展不同的数据源。
"""

import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 创建FastMCP服务器
mcp = FastMCP("sql-query-server")

# 数据库配置
DB_CONFIG = {
    "type": "sqlite",
    "path": Path(__file__).parent / "data" / "company.db"
}
AUDIT_LOG_PATH = Path(__file__).parent / "logs" / "sql_audit.jsonl"


def _append_audit_log(record: dict) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_safe_read_only_sql(sql: str) -> tuple[bool, str]:
    normalized = re.sub(r"\s+", " ", (sql or "").strip()).lower()
    if not normalized:
        return False, "SQL 为空"
    if ";" in normalized:
        return False, "仅允许单条 SQL 查询"
    if not (normalized.startswith("select ") or normalized.startswith("with ")):
        return False, "仅允许 SELECT / WITH 只读查询"

    blocked_tokens = [
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
    for token in blocked_tokens:
        if token in wrapped:
            return False, f"命中风险语句: {token.strip()}"
    return True, ""


@mcp.tool()
def execute_sql(sql: str) -> str:
    """执行SQL查询
    
    Args:
        sql: SQL查询语句
        db_type: 数据库类型（sqlite）
        
    Returns:
        JSON格式的查询结果
    """
    allowed, reason = _is_safe_read_only_sql(sql)
    audit_base = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "db_type": DB_CONFIG["type"],
        "sql": sql,
        "allowed": allowed,
    }
    if not allowed:
        _append_audit_log({**audit_base, "success": False, "error": reason})
        return json.dumps({"error": f"SQL安全校验失败: {reason}", "sql": sql}, ensure_ascii=False)

    result = _execute_sqlite(sql)
    try:
        parsed = json.loads(result)
        if isinstance(parsed, list):
            _append_audit_log({**audit_base, "success": True, "row_count": len(parsed)})
        else:
            _append_audit_log(
                {
                    **audit_base,
                    "success": "error" not in parsed,
                    "row_count": 0,
                    "error": parsed.get("error"),
                }
            )
    except Exception:
        _append_audit_log({**audit_base, "success": False, "error": "审计解析失败"})
    return result


def _execute_sqlite(sql: str) -> str:
    """执行SQLite查询"""
    db_path = DB_CONFIG["path"]
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        if rows:
            result = [dict(row) for row in rows]
            output = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            output = json.dumps({"message": "查询结果为空"})
        
        conn.close()
        return output
        
    except sqlite3.Error as e:
        error_output = json.dumps({
            "error": f"SQL执行错误: {str(e)}",
            "sql": sql
        }, ensure_ascii=False)
        return error_output
    except Exception as e:
        error_output = json.dumps({
            "error": f"未知错误: {str(e)}"
        }, ensure_ascii=False)
        return error_output

if __name__ == "__main__":
    mcp.run()

