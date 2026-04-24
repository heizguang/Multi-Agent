import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from env_loader import load_env_file
load_env_file()

import requests
import json

model = os.getenv("EMBEDDING_MODEL")
base_url = os.getenv("EMBEDDING_BASE_URL")
api_key = os.getenv("EMBEDDING_API_KEY")

print(f"Model: {model}")
print(f"Base URL: {base_url}")

payload = {
    "model": model,
    "input": "测试文本",
}

response = requests.post(
    f"{base_url}/embeddings",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=60,
)

print(f"Status: {response.status_code}")

with open("d:/1/Multi-Agent-Exp/embedding_test.txt", "w", encoding="utf-8") as f:
    f.write(response.text)

data = response.json()
if "data" in data and len(data["data"]) > 0:
    embedding = data["data"][0]["embedding"]
    print(f"向量维度: {len(embedding)}")
    print(f"前5个值: {embedding[:5]}")
else:
    print(f"Response: {response.text[:500]}")
