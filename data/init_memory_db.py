"""
长期记忆数据库初始化

创建用户信息、偏好和知识存储表。
"""

import sqlite3
from pathlib import Path


def init_memory_database(db_path: str = "./data/long_term_memory.db") -> None:
    """初始化长期记忆数据库
    
    Args:
        db_path: 数据库文件路径
    """
    # 确保目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建用户信息表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 创建用户偏好表（键值对存储）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            pref_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            UNIQUE(user_id, key)
        )
    """)
    
    # 创建索引加速查询
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_preferences_user 
        ON user_preferences(user_id)
    """)
    
    # 创建用户交互历史摘要表（长期知识）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_knowledge (
            knowledge_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.8,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # 创建索引加速查询
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_user 
        ON user_knowledge(user_id, category)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_knowledge_last_accessed
        ON user_knowledge(last_accessed)
    """)

    # FTS5 全文索引：用于替代简单 LIKE 匹配
    try:
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS user_knowledge_fts
            USING fts5(
                content,
                category UNINDEXED,
                user_id UNINDEXED,
                content='user_knowledge',
                content_rowid='knowledge_id'
            )
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS user_knowledge_ai
            AFTER INSERT ON user_knowledge
            BEGIN
                INSERT INTO user_knowledge_fts(rowid, content, category, user_id)
                VALUES (new.knowledge_id, new.content, new.category, new.user_id);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS user_knowledge_ad
            AFTER DELETE ON user_knowledge
            BEGIN
                INSERT INTO user_knowledge_fts(user_knowledge_fts, rowid, content, category, user_id)
                VALUES ('delete', old.knowledge_id, old.content, old.category, old.user_id);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS user_knowledge_au
            AFTER UPDATE ON user_knowledge
            BEGIN
                INSERT INTO user_knowledge_fts(user_knowledge_fts, rowid, content, category, user_id)
                VALUES ('delete', old.knowledge_id, old.content, old.category, old.user_id);
                INSERT INTO user_knowledge_fts(rowid, content, category, user_id)
                VALUES (new.knowledge_id, new.content, new.category, new.user_id);
            END
        """)

        cursor.execute("""
            INSERT INTO user_knowledge_fts(user_knowledge_fts)
            VALUES ('rebuild')
        """)
    except sqlite3.OperationalError:
        # 某些 SQLite 构建可能不包含 FTS5，此时保留普通表结构并在上层回退到 LIKE。
        pass

    conn.commit()
    conn.close()
    
    print(f"长期记忆数据库初始化完成: {db_path}")


if __name__ == "__main__":
    init_memory_database()

