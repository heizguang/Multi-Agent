"""
快速测试 NL2SQL vLLM 是否能生成 SQL。

用法：
python test_nl2sql_quick.py
python test_nl2sql_quick.py --question "研发部平均工资是多少？"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import requests

from prompts import get_few_shot_prompt


ROOT = Path(__file__).resolve().parent


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def normalize_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE
    ).strip()
    return text


def extract_sql(text: str) -> str:
    text = strip_thinking(text)
    block = re.search(r"```sql\s*([\s\S]*?)```", text, re.IGNORECASE)
    if block:
        return block.group(1).strip()
    # 必须从行首开始，避免把英文说明里的 "with ..." 误判为 SQL。
    for line_start in re.finditer(r"(?im)^\s*(SELECT|WITH)\b", text):
        sql = text[line_start.start() :].strip()
        sql = re.sub(r"\s*```[\s\S]*$", "", sql).strip()
        sql = sql.rstrip(";").strip()
        if sql:
            return sql
    return text.strip()


def is_valid_sql(sql: str) -> bool:
    return bool(
        re.search(r"^\s*(SELECT|WITH)\b", sql, flags=re.IGNORECASE)
        and re.search(r"\bFROM\b", sql, flags=re.IGNORECASE)
    )


def build_demo_schema() -> str:
    return """
表：employees
  - emp_id: INTEGER (主键)
  - emp_name: TEXT
  - gender: TEXT
  - hire_date: TEXT
  - dept_id: INTEGER
  - position: TEXT

表：departments
  - dept_id: INTEGER (主键)
  - dept_name: TEXT
  - location: TEXT

表：salaries
  - emp_id: INTEGER (主键)
  - base_salary: REAL
  - bonus: REAL
    """.strip()


def test_chat_completion(
    chat_url: str,
    headers: dict[str, str],
    model: str,
    user_content: str,
    max_tokens: int,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 SQL 生成器。只输出一条 SQLite SQL。"
                    "严禁输出思考过程、分析、解释、Markdown、代码围栏。"
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": max(128, max_tokens),
    }
    resp = requests.post(chat_url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"chat 请求失败: {resp.status_code} {resp.text[:500]}")
    data = resp.json()
    content = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return content, extract_sql(content)


def test_completion_fallback(
    completion_url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max(128, max_tokens),
    }
    resp = requests.post(completion_url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"completion 请求失败: {resp.status_code} {resp.text[:500]}")
    data = resp.json()
    text = str(((data.get("choices") or [{}])[0]).get("text") or "")
    return text, extract_sql(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="快速测试 NL2SQL SQL 生成")
    parser.add_argument(
        "--question",
        default="各部门分别有多少人？",
        help="测试问题（默认：各部门分别有多少人？）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="输出 token 上限（默认 256）",
    )
    parser.add_argument(
        "--with-schema",
        action="store_true",
        help="使用 few-shot + schema 提示词（更贴近项目真实 NL2SQL）",
    )
    args = parser.parse_args()

    load_dotenv_file(ROOT / ".env")
    base_url = os.getenv("NL2SQL_BASE_URL", "").strip()
    model = os.getenv("NL2SQL_MODEL", "").strip()
    api_key = os.getenv("NL2SQL_API_KEY", "").strip() or "dummy"

    if not base_url or not model:
        print("缺少 NL2SQL_BASE_URL 或 NL2SQL_MODEL（请检查 .env）", file=sys.stderr)
        return 1

    api_base = normalize_base_url(base_url)
    chat_url = f"{api_base}/chat/completions"
    completion_url = f"{api_base}/completions"

    user_content = f"问题：{args.question}\nSQL:"
    if args.with_schema:
        user_content = get_few_shot_prompt(args.question, build_demo_schema(), num_examples=3)

    completion_prompt = user_content
    if not args.with_schema:
        completion_prompt = (
            "你是SQL专家。只输出一条SQLite SQL，不要解释，不要Markdown。\n"
            f"问题：{args.question}\nSQL:"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"POST {chat_url}")
    print(f"model={model}")
    try:
        content, sql = test_chat_completion(chat_url, headers, model, user_content, args.max_tokens)
        print("\n=== 模型原始输出 ===")
        print(content.strip() or "<empty>")
        print("\n=== 解析出的 SQL ===")
        print(sql or "<empty>")

        if is_valid_sql(sql):
            print("\n[PASS] 已生成看起来有效的 SQL。")
            return 0

        print("\n[INFO] chat/completions 未返回有效 SQL，尝试 completions 回退...")
        print(f"POST {completion_url}")
        raw2, sql2 = test_completion_fallback(
            completion_url,
            headers,
            model,
            completion_prompt,
            max(args.max_tokens, 512),
        )
        print("\n=== completions 原始输出 ===")
        print(raw2.strip() or "<empty>")
        print("\n=== completions 解析 SQL ===")
        print(sql2 or "<empty>")

        if is_valid_sql(sql2):
            print("\n[PASS] 回退路径成功，已生成有效 SQL。")
            return 0
        print("\n[FAIL] 两条路径都未识别到有效 SQL。", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    except requests.RequestException as e:
        print(f"请求异常: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
