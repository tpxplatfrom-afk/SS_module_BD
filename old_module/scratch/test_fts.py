import sys
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("packs/class8_math/index.db")
cursor = conn.cursor()

print("--- Tables ---")
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
for row in cursor.fetchall():
    print(row[0])

print("\n--- FTS direct test ---")
try:
    cursor.execute("SELECT chunk_id, bm25(fts_knowledge) FROM fts_knowledge WHERE fts_knowledge MATCH 'ছাঁকনি'")
    rows = cursor.fetchall()
    print("Match 'ছাঁকনি':", rows)
except Exception as e:
    print("FTS Error:", e)

try:
    # Test with token prefix
    cursor.execute("SELECT chunk_id, bm25(fts_knowledge) FROM fts_knowledge WHERE fts_knowledge MATCH 'মৌলিক'")
    rows = cursor.fetchall()
    print("Match 'মৌলিক':", rows)
except Exception as e:
    print("FTS Error:", e)

conn.close()
