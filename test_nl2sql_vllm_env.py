"""
测试 .env 中的 NL2SQL 专用 vLLM 配置（NL2SQL_BASE_URL / NL2SQL_MODEL / NL2SQL_API_KEY）。

行为与 agents/sql_agent.py 中 _init_sql_llm 一致：若 BASE_URL 未以 /v1 结尾则自动补上，
并使用 OpenAI 兼容的 POST .../chat/completions。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent


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


def main() -> int:
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
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "用一句话回复：连接成功。"}],
        "temperature": 0,
        "max_tokens": 64,
    }

    print()
    print(f"POST {chat_url}")
    try:
        r = requests.post(chat_url, headers=headers, json=payload, timeout=120)
        print(f"  status: {r.status_code}")
        if not r.ok:
            print(f"  body: {r.text[:2000]}")
            return 1
        data = r.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content", "")
        print("  reply:", content.strip() or json.dumps(data, ensure_ascii=False)[:500])
    except requests.RequestException as e:
        print(f"  请求失败: {e}", file=sys.stderr)
        return 1

    print()
    print("NL2SQL vLLM 连通性测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
