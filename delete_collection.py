"""
在 WSL 中安装 Milvus

# 1. 下载并安装 Milvus
cd ~
wget https://github.com/milvus-io/milvus/releases/download/v2.6.4/milvus_2.6.4-1_amd64.deb -O milvus.deb
sudo apt install -y ./milvus.deb

# 2. 启动 Milvus
sudo systemctl start milvus

# 3. 检查状态
sudo systemctl status milvus

# 常用命令
sudo systemctl start milvus   # 启动
sudo systemctl stop milvus    # 停止
sudo systemctl status milvus  # 状态
"""

import sys
import os
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent))

from env_loader import load_env_file
load_env_file()

print("检查并删除旧的 Milvus Collection...")

try:
    from pymilvus import connections, utility
    
    connections.connect(host="localhost", port=19530)
    print("已连接到 Milvus")
    
    collection_name = "user_memory"
    
    if utility.has_collection(collection_name):
        print(f"删除 Collection: {collection_name}")
        utility.drop_collection(collection_name)
        print("Collection 已删除")
    else:
        print("Collection 不存在")
        
    connections.disconnect("default")
    print("连接已关闭")
        
except Exception as e:
    print(f"错误: {e}")
