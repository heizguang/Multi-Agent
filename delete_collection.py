import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from env_loader import load_env_file
load_env_file()

try:
    from pymilvus import connections, utility
    
    print("连接 Milvus...")
    connections.connect(host="localhost", port=19530)
    print("已连接")
    
    collection_name = "user_memory"
    
    if utility.has_collection(collection_name):
        print(f"删除 Collection: {collection_name}")
        utility.drop_collection(collection_name)
        print("Collection 已删除")
    else:
        print("Collection 不存在")
        
except Exception as e:
    print(f"错误: {e}")
