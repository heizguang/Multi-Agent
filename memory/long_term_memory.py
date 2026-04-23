"""
长期记忆管理器

负责用户偏好、知识的存储、检索、更新与访问热度维护。
支持向量检索（Milvus）和传统 FTS5/LIKE 检索。
"""

import logging
import math
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LongTermMemory:
    """长期记忆管理器，支持 FTS5、访问热度与时间衰减排序，集成向量检索。"""

    def __init__(
        self,
        memory_db_path: str = "./data/long_term_memory.db",
        vector_store=None,
        use_vector_threshold: int = 100
    ):
        """
        初始化长期记忆管理器
        
        Args:
            memory_db_path: SQLite 数据库路径
            vector_store: 向量存储实例 (VectorStore)
            use_vector_threshold: 记忆数量超过此值时启用向量检索，默认 100
        """
        self.db_path = memory_db_path
        self.vector_store = vector_store
        self.use_vector_threshold = use_vector_threshold
        self.fts_enabled = False
        self._ensure_database()

    def _get_knowledge_count(self, user_id: str) -> int:
        """获取用户的知识数量"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM user_knowledge WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result["cnt"] if result else 0
        finally:
            conn.close()

    def _ensure_database(self):
        """确保数据库存在，并对旧库执行轻量迁移。"""
        if not Path(self.db_path).exists():
            from data.init_memory_db import init_memory_database

            init_memory_database(self.db_path)

        self._run_migrations()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _run_migrations(self) -> None:
        """对旧版数据库执行兼容迁移。"""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("PRAGMA table_info(user_knowledge)")
            columns = {row["name"] for row in cursor.fetchall()}

            if "access_count" not in columns:
                cursor.execute(
                    "ALTER TABLE user_knowledge ADD COLUMN access_count INTEGER DEFAULT 0"
                )
            if "last_accessed" not in columns:
                cursor.execute(
                    "ALTER TABLE user_knowledge ADD COLUMN last_accessed TIMESTAMP"
                )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_last_accessed
                ON user_knowledge(last_accessed)
                """
            )

            self.fts_enabled = self._ensure_fts(cursor)
            conn.commit()
        finally:
            conn.close()

    def _ensure_fts(self, cursor: sqlite3.Cursor) -> bool:
        """创建 FTS5 索引与同步触发器。"""
        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS user_knowledge_fts
                USING fts5(
                    content,
                    category UNINDEXED,
                    user_id UNINDEXED,
                    content='user_knowledge',
                    content_rowid='knowledge_id'
                )
                """
            )

            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS user_knowledge_ai
                AFTER INSERT ON user_knowledge
                BEGIN
                    INSERT INTO user_knowledge_fts(rowid, content, category, user_id)
                    VALUES (new.knowledge_id, new.content, new.category, new.user_id);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS user_knowledge_ad
                AFTER DELETE ON user_knowledge
                BEGIN
                    INSERT INTO user_knowledge_fts(user_knowledge_fts, rowid, content, category, user_id)
                    VALUES ('delete', old.knowledge_id, old.content, old.category, old.user_id);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS user_knowledge_au
                AFTER UPDATE ON user_knowledge
                BEGIN
                    INSERT INTO user_knowledge_fts(user_knowledge_fts, rowid, content, category, user_id)
                    VALUES ('delete', old.knowledge_id, old.content, old.category, old.user_id);
                    INSERT INTO user_knowledge_fts(rowid, content, category, user_id)
                    VALUES (new.knowledge_id, new.content, new.category, new.user_id);
                END
                """
            )
            cursor.execute(
                """
                INSERT INTO user_knowledge_fts(user_knowledge_fts)
                VALUES ('rebuild')
                """
            )
            return True
        except sqlite3.OperationalError:
            return False

    # ==================== 用户管理 ====================

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT user_id, created_at, last_active
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        )

        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def create_or_update_user(self, user_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO users (user_id, created_at, last_active)
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_active = CURRENT_TIMESTAMP
                """,
                (user_id,),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"创建/更新用户失败: {e}")
            return False
        finally:
            conn.close()

    def update_user_activity(self, user_id: str) -> bool:
        return self.create_or_update_user(user_id)

    # ==================== 用户偏好管理 ====================

    def save_preference(self, user_id: str, key: str, value: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            self.create_or_update_user(user_id)
            cursor.execute(
                """
                INSERT INTO user_preferences (user_id, key, value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, key, value),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"保存偏好失败: {e}")
            return False
        finally:
            conn.close()

    def get_preference(
        self, user_id: str, key: str, default: Optional[str] = None
    ) -> Optional[str]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT value
            FROM user_preferences
            WHERE user_id = ? AND key = ?
            """,
            (user_id, key),
        )

        row = cursor.fetchone()
        conn.close()
        return row["value"] if row else default

    def get_all_preferences(self, user_id: str) -> Dict[str, str]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT key, value
            FROM user_preferences
            WHERE user_id = ?
            """,
            (user_id,),
        )

        rows = cursor.fetchall()
        conn.close()
        return {row["key"]: row["value"] for row in rows}

    # ==================== 用户知识管理 ====================

    def save_knowledge(
        self,
        user_id: str,
        category: str,
        content: str,
        confidence: float = 0.8,
    ) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            self.create_or_update_user(user_id)
            cursor.execute(
                """
                INSERT INTO user_knowledge (
                    user_id, category, content, confidence, access_count, last_accessed, created_at
                )
                VALUES (?, ?, ?, ?, 0, NULL, CURRENT_TIMESTAMP)
                """,
                (user_id, category, content, confidence),
            )
            conn.commit()

            if self.vector_store and self.vector_store.is_available():
                self.vector_store.add_memory(user_id, content, category)
                logger.info(f"知识已同时保存到向量存储")

            return True
        except Exception as e:
            logger.warning(f"保存知识失败: {e}")
            return False
        finally:
            conn.close()

    def get_knowledge_by_category(
        self, user_id: str, category: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT knowledge_id, category, content, confidence, access_count, last_accessed, created_at
            FROM user_knowledge
            WHERE user_id = ? AND category = ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, category, limit),
        )

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        self._mark_knowledge_accessed([row["knowledge_id"] for row in rows])
        return rows

    def get_relevant_knowledge(
        self, user_id: str, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        knowledge_count = self._get_knowledge_count(user_id)
        
        use_vector = (
            self.vector_store and 
            self.vector_store.is_available() and 
            knowledge_count >= self.use_vector_threshold
        )
        
        if use_vector:
            logger.info(f"使用向量检索（知识数量: {knowledge_count} >= {self.use_vector_threshold}）")
            return self._vector_search(user_id, query, top_k)
        
        logger.info(f"使用传统检索（知识数量: {knowledge_count} < {self.use_vector_threshold}）")
        return self._keyword_search(user_id, query, top_k)

    def _vector_search(
        self, user_id: str, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            vector_results = self.vector_store.search(user_id, query, top_k * 2)
            if not vector_results:
                return self._keyword_search(user_id, query, top_k)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            results = []
            for vec_result in vector_results[:top_k]:
                content = vec_result.get("content", "")
                category = vec_result.get("category", "")
                
                cursor.execute(
                    """
                    SELECT knowledge_id, category, content, confidence, access_count, last_accessed, created_at
                    FROM user_knowledge
                    WHERE user_id = ? AND content = ? AND category = ?
                    LIMIT 1
                    """,
                    (user_id, content, category)
                )
                row = cursor.fetchone()
                if row:
                    row_dict = dict(row)
                    row_dict["effective_score"] = 1.0 - vec_result.get("distance", 1.0)
                    row_dict["vector_distance"] = vec_result.get("distance")
                    results.append(row_dict)
            
            conn.close()
            
            if results:
                self._mark_knowledge_accessed([r["knowledge_id"] for r in results])
            
            return results
        except Exception as e:
            logger.warning(f"向量检索失败，回退到关键词检索: {e}")
            return self._keyword_search(user_id, query, top_k)

    def _keyword_search(
        self, user_id: str, query: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """关键词检索（原有逻辑）"""
        candidates = self._fetch_candidate_knowledge(user_id, query, max(top_k * 4, 8))

        scored_rows = []
        for row in candidates:
            row_dict = dict(row)
            row_dict["effective_score"] = round(
                self._compute_effective_score(row_dict, query), 4
            )
            scored_rows.append(row_dict)

        scored_rows.sort(
            key=lambda item: (
                item.get("effective_score", 0),
                float(item.get("confidence") or 0),
                item.get("created_at") or "",
            ),
            reverse=True,
        )

        selected = scored_rows[:top_k]
        self._mark_knowledge_accessed([row["knowledge_id"] for row in selected])
        return selected

    def _fetch_candidate_knowledge(
        self, user_id: str, query: str, limit: int
    ) -> List[sqlite3.Row]:
        conn = self._get_connection()
        cursor = conn.cursor()
        tokens = self._extract_query_tokens(query)

        try:
            if self.fts_enabled and tokens:
                match_query = " OR ".join(f'"{token}"' for token in tokens[:8])
                cursor.execute(
                    """
                    SELECT uk.knowledge_id, uk.category, uk.content, uk.confidence,
                           uk.access_count, uk.last_accessed, uk.created_at
                    FROM user_knowledge uk
                    JOIN user_knowledge_fts fts
                      ON fts.rowid = uk.knowledge_id
                    WHERE uk.user_id = ? AND user_knowledge_fts MATCH ?
                    LIMIT ?
                    """,
                    (user_id, match_query, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    return rows

            where_clauses = ["user_id = ?"]
            params: List[Any] = [user_id]

            if tokens:
                token_clauses = []
                for token in tokens[:8]:
                    token_clauses.append("content LIKE ?")
                    params.append(f"%{token}%")
                where_clauses.append("(" + " OR ".join(token_clauses) + ")")
            else:
                where_clauses.append("content LIKE ?")
                params.append(f"%{query.strip()}%")

            params.append(limit)
            sql = f"""
                SELECT knowledge_id, category, content, confidence,
                       access_count, last_accessed, created_at
                FROM user_knowledge
                WHERE {' AND '.join(where_clauses)}
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            if rows:
                return rows

            cursor.execute(
                """
                SELECT knowledge_id, category, content, confidence,
                       access_count, last_accessed, created_at
                FROM user_knowledge
                WHERE user_id = ?
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def _extract_query_tokens(self, query: str) -> List[str]:
        normalized = query.strip().lower()
        if not normalized:
            return []

        ascii_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", normalized)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)

        keyword_map = {
            "薪资": ["薪资", "工资", "奖金", "薪酬"],
            "员工": ["员工", "人员", "同事"],
            "部门": ["部门", "团队", "组织"],
            "分析": ["分析", "总结", "洞察", "趋势"],
        }

        expanded = []
        for token in chinese_tokens:
            expanded.append(token)
            for canonical, aliases in keyword_map.items():
                if any(alias in token for alias in aliases):
                    expanded.append(canonical)
                    expanded.extend(aliases)

        ordered = []
        seen = set()
        for token in ascii_tokens + expanded:
            clean = token.strip()
            if len(clean) < 2 or clean in seen:
                continue
            seen.add(clean)
            ordered.append(clean)
        return ordered

    def _compute_effective_score(self, row: Dict[str, Any], query: str) -> float:
        confidence = float(row.get("confidence") or 0.0)
        access_count = int(row.get("access_count") or 0)

        last_touch = row.get("last_accessed") or row.get("created_at")
        age_days = self._days_since(last_touch)
        recency_factor = math.exp(-age_days / 30.0)
        access_boost = min(math.log1p(access_count) / 4.0, 0.35)

        overlap_tokens = self._extract_query_tokens(query)
        content = str(row.get("content") or "").lower()
        overlap = 0.0
        if overlap_tokens:
            hit_count = sum(1 for token in overlap_tokens if token in content)
            overlap = hit_count / len(overlap_tokens)

        return (
            confidence * 0.55
            + recency_factor * 0.20
            + access_boost * 0.10
            + overlap * 0.15
        )

    def _days_since(self, dt_value: Optional[str]) -> float:
        if not dt_value:
            return 365.0

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(str(dt_value), fmt)
                return max((datetime.now() - dt).total_seconds() / 86400.0, 0.0)
            except ValueError:
                continue
        return 365.0

    def _mark_knowledge_accessed(self, knowledge_ids: List[int]) -> None:
        if not knowledge_ids:
            return

        placeholders = ", ".join("?" for _ in knowledge_ids)
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                f"""
                UPDATE user_knowledge
                SET access_count = COALESCE(access_count, 0) + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE knowledge_id IN ({placeholders})
                """,
                knowledge_ids,
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"更新知识访问热度失败: {e}")
        finally:
            conn.close()

    def get_all_knowledge(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT knowledge_id, category, content, confidence,
                   access_count, last_accessed, created_at
            FROM user_knowledge
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )

        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    # ==================== 工具方法 ====================

    def delete_preference(self, user_id: str, key: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM user_preferences
                WHERE user_id = ? AND key = ?
                """,
                (user_id, key),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"删除偏好失败: {e}")
            return False
        finally:
            conn.close()

    def delete_knowledge(self, knowledge_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                DELETE FROM user_knowledge
                WHERE knowledge_id = ?
                """,
                (knowledge_id,),
            )
            conn.commit()
            return True
        except Exception as e:
            logger.warning(f"删除知识失败: {e}")
            return False
        finally:
            conn.close()
