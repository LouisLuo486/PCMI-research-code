# src/load_overlap_to_sqlite.py
import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join("out", "data.sqlite")
CSV_PATH = os.path.join("out", "p3_p4_overlap_claims.csv")
TABLE = "p3_p4_overlap"

def create_indexes(conn, table):
    # 按字段存在性有选择地建索引，避免报错
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    idx_plan = [
        ("idx_p3p4_claim",      "iId"),
        ("idx_p3p4_contract",   "iContractId"),
        ("idx_p3p4_servicer",   "iServicerId"),
    ]
    for idx, col in idx_plan:
        if col in cols:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON {table}({col});")

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(DB_PATH):
        # 若 DB 不存在则先创建空库
        open(DB_PATH, "a").close()

    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        # 覆盖式写入（如要追加改为 if_exists='append'）
        df.to_sql(TABLE, conn, if_exists='replace', index=False)
        create_indexes(conn, TABLE)
        total, = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
        print(f"[OK] Wrote {total} rows into table '{TABLE}' in {DB_PATH}")

if __name__ == "__main__":
    main()
