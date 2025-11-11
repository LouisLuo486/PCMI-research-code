# src/make_in_claims_tables.py
import argparse, sqlite3
from pathlib import Path

def detect_key(cur, table):
    cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')]
    for k in ("iid","iId","ICLAIMID","claim_id","Id","ID"):
        for c in cols:
            if c.lower() == k.lower():
                return c
    # 兜底：第一列但避开 rowid
    for c in cols:
        if c.lower() != "rowid":
            return c
    return cols[0]

def build_join(db_path: Path, claims_table: str, ids_table: str, out_table: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    try:
        ck = detect_key(cur, claims_table)
        ik = detect_key(cur, ids_table)
        cur.execute(f'DROP TABLE IF EXISTS "{out_table}"')
        cur.execute(
            f'CREATE TABLE "{out_table}" AS '
            f'SELECT c.* FROM "{claims_table}" c '
            f'JOIN "{ids_table}" x ON c."{ck}" = x."{ik}"'
        )
        con.commit()
        cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{out_table}_claim_id ON "{out_table}"("{ck}")')
        con.commit()
        n = cur.execute(f'SELECT COUNT(*) FROM "{out_table}"').fetchone()[0]
        print(f"[OK] {out_table} rows={n}  (join key: {claims_table}.{ck} ↔ {ids_table}.{ik})")
    finally:
        con.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db",   required=True, help="SQLite 文件路径（例如 .\\out\\data.sqlite）")
    ap.add_argument("--base", required=True, help="claims 明细表名（通常是 claims）")
    ap.add_argument("--ids",  required=True, help="只含交集 ID 的表名（例如 p3p4p5p6p7p8_claims）")
    ap.add_argument("--out",  required=True, help="输出表名（例如 p3p4p5p6p7p8_claims_in_claims）")
    args = ap.parse_args()
    build_join(Path(args.db), args.base, args.ids, args.out)
