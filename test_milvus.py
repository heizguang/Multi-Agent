import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from env_loader import load_env_file
load_env_file()

print(f"MILVUS_ENABLED: {os.getenv('MILVUS_ENABLED')}")
print(f"MILVUS_EMBEDDED: {os.getenv('MILVUS_EMBEDDED')}")
print(f"MILVUS_HOST: {os.getenv('MILVUS_HOST')}")
print(f"MILVUS_PORT: {os.getenv('MILVUS_PORT')}")

try:
    from pymilvus import connections
    print("\n尝试连接 Milvus...")
    
    use_embedded = os.getenv("MILVUS_EMBEDDED", "false").lower() == "true"
    print(f"use_embedded: {use_embedded}")
    
    if use_embedded:
        connections.connect(alias="default", uri="./milvus_data.db")
    else:
        host = os.getenv("MILVUS_HOST", "localhost")
        port = int(os.getenv("MILVUS_PORT", "19530"))
        connections.connect(host=host, port=port)
    
    print("连接成功!")
except Exception as e:
    print(f"连接失败: {e}")
