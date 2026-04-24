import sqlite3

db_path = "d:/1/Multi-Agent-Exp/data/long_term_memory.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT user_id, COUNT(*) FROM user_knowledge GROUP BY user_id")
counts = cursor.fetchall()

print("每个用户的知识数量:")
for user_id, count in counts:
    print(f"  {user_id}: {count} 条")

conn.close()
