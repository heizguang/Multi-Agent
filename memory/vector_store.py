"""
Milvus 向量存储模块

用于长期记忆的向量检索。
支持 Milvus standalone 模式。
"""

import logging
import os
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
except (ImportError, AttributeError) as e:
    logger.warning(f"pymilvus 导入失败: {e}，向量检索功能不可用")
    connections = None
    Collection = None


class VectorStore:
    """Milvus 向量存储管理器，支持 standalone 模式"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "user_memory",
        embedding_dim: int = 1024,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        enabled: bool = True,
        use_embedded: bool = True
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_func = embedding_func
        self.enabled = enabled and embedding_func is not None
        self.use_embedded = use_embedded
        self.connected = False
        self.collection = None

        if self.enabled:
            self._connect()

    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        return self.enabled and self.connected

    def _connect(self):
        """连接 Milvus"""
        if not self.enabled:
            return

        try:
            if connections is None:
                logger.warning("pymilvus 未安装，向量检索功能不可用")
                self.enabled = False
                return

            if self.use_embedded:
                connections.connect(alias="default", uri="./milvus_data.db")
                self.connected = True
                logger.info("已连接到 Embedded Milvus (milvus-lite)")
            else:
                connections.connect(host=self.host, port=self.port)
                self.connected = True
                logger.info(f"已连接到 Milvus: {self.host}:{self.port}")
            
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"连接 Milvus 失败: {e}，向量检索功能将不可用")
            self.enabled = False
            self.connected = False

    def _ensure_collection(self):
        """确保 Collection 存在"""
        if not self.connected:
            return

        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
                logger.info(f"Collection '{self.collection_name}' 已加载")
            else:
                self._create_collection()
        except Exception as e:
            logger.warning(f"初始化 Collection 失败: {e}")

    def _create_collection(self):
        """创建 Collection"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
        ]
        schema = CollectionSchema(fields=fields, description="用户记忆向量存储")
        self.collection = Collection(name=self.collection_name, schema=schema)

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="vector", index_params=index_params)
        self.collection.load()
        logger.info(f"Collection '{self.collection_name}' 已创建")

    def add_memory(
        self,
        user_id: str,
        content: str,
        category: str
    ) -> bool:
        """添加记忆到向量库（自动生成向量）"""
        if not self.enabled or not self.connected:
            return False

        if not self.embedding_func:
            return False

        try:
            vector = self.embedding_func(content)
            if not vector or len(vector) == 0:
                logger.warning("生成向量失败")
                return False

            self.collection.insert([
                [user_id],
                [content],
                [category],
                [vector]
            ])
            self.collection.flush()
            return True
        except Exception as e:
            logger.warning(f"添加向量记忆失败: {e}")
            return False

    def search(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """搜索向量（传入文本，自动生成向量）"""
        if not self.enabled or not self.connected:
            return []

        if not self.embedding_func:
            return []

        try:
            query_vector = self.embedding_func(query_text)
            if not query_vector or len(query_vector) == 0:
                logger.warning("生成查询向量失败")
                return []

            self.collection.load()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=f'user_id == "{user_id}"',
                output_fields=["id", "user_id", "content", "category"]
            )

            hits = []
            for result in results:
                for hit in result:
                    hits.append({
                        "id": hit.id,
                        "content": hit.entity.get("content"),
                        "category": hit.entity.get("category"),
                        "distance": hit.distance
                    })
            return hits
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
            return []

    def delete_by_user(self, user_id: str) -> bool:
        """删除用户的所有向量记忆"""
        if not self.enabled or not self.connected:
            return False

        try:
            expr = f'user_id == "{user_id}"'
            self.collection.delete(expr)
            self.collection.flush()
            return True
        except Exception as e:
            logger.warning(f"删除向量记忆失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.connected:
            try:
                connections.disconnect("default")
            except:
                pass
            self.connected = False

    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        return self.enabled and self.connected


def create_embedding_function(llm, dim: int = 1024):
    """创建 embedding 函数"""
    def embed_text(text: str) -> List[float]:
        try:
            prompt = f"""请将以下文本转换为一个 {dim} 维的向量表示。
只返回一个 JSON 数组，不要其他文字。
文本：{text}
向量："""

            response = llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            import json
            data = json.loads(response_text)
            if isinstance(data, list):
                return data[:dim]
            
            return [0.0] * dim
        except Exception as e:
            logger.warning(f"生成 embedding 失败: {e}")
            return [0.0] * dim

    return embed_text
