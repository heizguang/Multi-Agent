"""
测试 .env 中的 NL2SQL 专用 vLLM 配置（NL2SQL_BASE_URL / NL2SQL_MODEL / NL2SQL_API_KEY）。

1) 连通性：GET /v1/models + 一句闲聊 chat。
2) NL2SQL（默认开启）：用 prompts.get_few_shot_prompt 构造与线上一致的提示，
   请求模型生成 SQL，验证「能否正常产出 SQL」（与 sql_agent 相同 chat/completions）。

仅测连通、不测 SQL：python test_nl2sql_vllm_env.py --ping-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prompts import get_few_shot_prompt  # noqa: E402


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val


def normalize_base_url(base_url: str) -> str:
    b = base_url.strip().rstrip("/")
    if not b.endswith("/v1"):
        b = b + "/v1"
    return b


# 与 test_nl2sql.py / 业务库一致的示例 schema
DEMO_SCHEMA = """
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
"""


def strip_model_thinking(text: str) -> str:
    """与 agents/sql_agent.SQLQueryAgent._llm_to_str 一致，去掉 thinking 包裹。"""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"</think>", "", text).strip()
    return text


def extract_sql_from_reply(text: str) -> str:
    """从回复中尽量抽出可执行的 SQL（避免把 Thinking 里的 SELECT 当真）。"""
    text = strip_model_thinking(text)
    m = re.search(r"```sql\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(SELECT[\s\S]*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # 取最后一个以 SELECT 开头的连续片段（简单启发式）
    blocks = list(re.finditer(r"(?is)\bSELECT\b[\s\S]+?(?:;|\n\n|$)", text))
    if blocks:
        return blocks[-1].group(0).strip().rstrip(";").strip()
    return text.strip()


def chat_completion(
    chat_url: str,
    headers: dict,
    model: str,
    user_content: str,
    *,
    system: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 120,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(chat_url, headers=headers, json=payload, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"chat failed ({r.status_code}): {r.text[:2000]}")
    data = r.json()
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    return str(msg.get("content") or "")


SQL_ONLY_SYSTEM = (
    "你是 SQL 生成器。禁止输出思考过程、分析或英文说明。"
    "只输出一条完整、可执行的 SQLite SELECT 语句，不要 Markdown、不要代码围栏。"
)


def run_nl2sql_samples(
    chat_url: str,
    headers: dict,
    model: str,
    questions: list[str],
    *,
    max_tokens: int,
) -> bool:
    ok = True
    for i, question in enumerate(questions, 1):
        prompt = get_few_shot_prompt(question, DEMO_SCHEMA.strip(), num_examples=3)
        print()
        print(f"--- NL2SQL 样例 [{i}/{len(questions)}] 问题: {question}")
        try:
            raw = chat_completion(
                chat_url,
                headers,
                model,
                prompt,
                system=SQL_ONLY_SYSTEM,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = strip_model_thinking(raw)
            sql_guess = extract_sql_from_reply(raw)
            print("  解析出的 SQL（供人工核对）:")
            for line in sql_guess.splitlines()[:25]:
                print(f"    {line}")
            if len(sql_guess.splitlines()) > 25:
                print("    ...")
            print("  --- 原始回复节选（前 15 行）---")
            for line in text.splitlines()[:15]:
                print(f"    {line}")
            if len(text.splitlines()) > 15:
                print("    ...")
            looks_like_sql = bool(
                re.search(r"^\s*SELECT\b", sql_guess, re.IGNORECASE | re.MULTILINE)
                and re.search(r"\bFROM\b", sql_guess, re.IGNORECASE)
            )
            if not looks_like_sql:
                print(
                    "  [警告] 未能从回复中解析出带 FROM 的 SELECT，"
                    "可能全是思考过程；可加大 max_tokens 或换用不吐 thinking 的推理配置。",
                    file=sys.stderr,
                )
                ok = False
        except Exception as e:
            print(f"  [错误] {e}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 NL2SQL .env vLLM：连通性 + 可选 SQL 生成")
    parser.add_argument(
        "--ping-only",
        action="store_true",
        help="只测 models + 一句闲聊，不跑 NL2SQL 提示词",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        metavar="N",
        help="NL2SQL 样例请求的 max_tokens（默认 1024，思考型模型可适当加大）",
    )
    args = parser.parse_args()
    load_dotenv_file(ROOT / ".env")

    base_url = os.getenv("NL2SQL_BASE_URL", "").strip()
    model = os.getenv("NL2SQL_MODEL", "").strip()
    api_key = os.getenv("NL2SQL_API_KEY", "").strip() or "dummy"

    if not base_url or not model:
        print("缺少 NL2SQL_BASE_URL 或 NL2SQL_MODEL，请检查 .env 第 34–37 行附近配置。", file=sys.stderr)
        return 1

    api_base = normalize_base_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"API base (normalized): {api_base}")
    print(f"Model: {model}")
    print()

    models_url = f"{api_base}/models"
    print(f"GET {models_url}")
    try:
        r = requests.get(models_url, headers=headers, timeout=30)
        print(f"  status: {r.status_code}")
        if r.ok:
            data = r.json()
            ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
            print(f"  models ({len(ids)}): {ids[:5]}{' ...' if len(ids) > 5 else ''}")
            if model not in ids and ids:
                print(
                    f"  [提示] 当前 NL2SQL_MODEL 不在列表中，服务端可能使用其它 id；"
                    f"若 chat 失败可尝试改为列表中的某个 id。",
                    file=sys.stderr,
                )
        else:
            print(f"  body: {r.text[:500]}")
    except requests.RequestException as e:
        print(f"  [跳过] 无法拉取模型列表: {e}")

    chat_url = f"{api_base}/chat/completions"

    print()
    print(f"POST {chat_url} (ping)")
    try:
        ping_reply = chat_completion(
            chat_url,
            headers,
            model,
            "用一句话回复：连接成功。",
            max_tokens=64,
            temperature=0.0,
        )
        shown = strip_model_thinking(ping_reply) or ping_reply
        print("  reply:", shown.strip()[:500])
    except requests.RequestException as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"  {e}", file=sys.stderr)
        return 1

    print()
    print("NL2SQL vLLM 连通性测试通过。")

    if args.ping_only:
        return 0

    samples_ok = run_nl2sql_samples(
        chat_url,
        headers,
        model,
        [
            "各部门分别有多少人？",
            "研发部平均工资是多少？",
        ],
        max_tokens=max(256, args.max_tokens),
    )
    print()
    if samples_ok:
        print("NL2SQL 样例生成检查完成（输出中含 SELECT）。")
        return 0
    print("NL2SQL 样例存在问题，请查看上方警告/错误。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
