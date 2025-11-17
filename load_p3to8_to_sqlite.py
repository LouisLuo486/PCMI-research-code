#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sqlite3, pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB   = BASE / 'out' / 'data.sqlite'
CSV  = BASE / 'out' / 'p3p4p5p6p7p8_claims.csv'
TABLE= 'p3p4p5p6p7p8_claims'   # 你也可以改成 'p345678_claims'

def create_indexes(conn, table: str):
    # 动态检查列再建索引，避免无列时报错
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()}
    plan = [
        ("idx_p3to8_claim",   "claim_id"),
        ("idx_p3to8_vin",     "vin"),
        ("idx_p3to8_serv",    "servicer_id"),  # 若你的列名不是 servicer_id，下面会尝试别名
        ("idx_p3to8_date",    "service_date"), # 同理，尝试日期别名
    ]
    # 尝试一些常见别名
    alias = {
        "servicer_id": next((c for c in cols if c.lower() in {"servicer_id","iservicerid","iServicerId","servicer"}), None),
        "service_date": next((c for c in cols if c.lower() in
                              {"service_date","servicedate","dservicedate","claim_date","dclaimdate","created_at","dentrydate"}), None),
    }

    for idx_name, col in plan:
        use_col = col if col in cols else alias.get(col, None)
        if use_col and use_col in cols:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({use_col});")

def main():
    if not CSV.exists():
        raise FileNotFoundError(f"CSV 不存在：{CSV}")
    DB.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV, dtype=str, low_memory=False)
    # 统一去空白
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    with sqlite3.connect(DB) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {TABLE};")   # 想保留旧表可改成 if_exists='append'
        df.to_sql(TABLE, conn, if_exists='replace', index=False)
        create_indexes(conn, TABLE)
        total, = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
        print(f"[OK] 写入 {DB} 表 '{TABLE}' 行数={total}")

if __name__ == "__main__":
    main()
