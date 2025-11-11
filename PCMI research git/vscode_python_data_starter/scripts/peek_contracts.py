import sqlite3, pandas as pd

db = "out/data.sqlite"
conn = sqlite3.connect(db)

# 列出所有表
tabs = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
print("TABLES:", tabs)

# 找包含 'contract' 的表，打印列结构与前几行
for t in [x for x in tabs if "contract" in x.lower()]:
    print("\n==", t, "==")
    print(pd.read_sql(f"PRAGMA table_info({t});", conn))
    try:
        print(pd.read_sql(f"SELECT * FROM {t} LIMIT 5;", conn))
    except Exception as e:
        print("Sample read error:", e)

conn.close()
