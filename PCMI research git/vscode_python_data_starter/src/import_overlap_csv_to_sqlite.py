#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → SQLite 导入（写入固定到 out/data.sqlite）
- 清洗列名（小写、去空格、非法字符→下划线、重复列自动加后缀）
- 类型推断（整数/浮点/日期），其余 TEXT
- 始终将结果写入 项目根/out/data.sqlite
"""

import argparse, sqlite3, re
from pathlib import Path
import pandas as pd

# ---------- 列名清洗 ----------
def normalize_name(name: str) -> str:
    import re
    s = str(name).strip().lower()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s or 'col'

def dedupe_columns(cols):
    seen, out = {}, []
    for c in cols:
        base = normalize_name(c)
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

# ---------- 类型推断 ----------
def guess_types(df: pd.DataFrame, int_cols_hint=None, date_cols_hint=None):
    int_cols_hint = set(int_cols_hint or [])
    date_cols_hint = set(date_cols_hint or [])
    dtypes, out = {}, df.copy()

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip().replace({"": None, "nan": None})

    for c in out.columns:
        lc, s = c.lower(), out[c]

        if c in date_cols_hint or lc in {"dtservicedate","dtentrydate"}:
            parsed = pd.to_datetime(s, errors="coerce")
            out[c] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
            dtypes[c] = "TEXT"
            continue

        if c in int_cols_hint or lc in {"claim_id","iid","iclaimid","icontractid","iservicerid"}:
            num = pd.to_numeric(s, errors="coerce")
            if (num.dropna() % 1 == 0).mean() >= 0.95:
                out[c] = num.dropna().astype("Int64")
                dtypes[c] = "INTEGER"
                continue

        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.95:
            if (num.dropna() % 1 == 0).mean() >= 0.95:
                out[c] = num.dropna().astype("Int64"); dtypes[c] = "INTEGER"
            else:
                out[c] = num; dtypes[c] = "REAL"
        else:
            sample = s.dropna().astype(str).head(200)
            if not sample.empty and sample.str.match(r"\d{4}[-/]\d{2}[-/]\d{2}").mean() >= 0.8:
                parsed = pd.to_datetime(s, errors="coerce")
                out[c] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
            dtypes[c] = "TEXT"
    return out, dtypes

def ensure_table(con, table, mode):
    if mode == "replace":
        con.execute(f'DROP TABLE IF EXISTS "{table}"'); con.commit()

def create_table(con, table, dtypes):
    cols_sql = ", ".join([f'"{c}" {t}' for c, t in dtypes.items()])
    con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols_sql});'); con.commit()

def insert_bulk(con, table, df: pd.DataFrame, chunksize=5000):
    cols = list(df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    colnames = ", ".join([f'"{c}"' for c in cols])
    sql = f'INSERT INTO "{table}" ({colnames}) VALUES ({placeholders});'
    cur, total = con.cursor(), 0
    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i:i+chunksize]
        cur.executemany(sql, [tuple(None if pd.isna(v) else v for v in row) for row in chunk.to_numpy()])
        total += len(chunk)
    con.commit(); return total

def main():
    # 固定 DB 到 out/data.sqlite
    ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = ROOT / "out" / "data.sqlite"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="要导入的 CSV 路径")
    ap.add_argument("--table", required=True, help="目标表名")
    ap.add_argument("--mode", choices=["replace","append"], default="replace")
    ap.add_argument("--index", default="claim_id", help="建索引列（逗号分隔）")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    table    = normalize_name(args.table)  # 表名也清洗，防止大小写/非法字符
    print(f"[INFO] DB     : {DB_PATH}")
    print(f"[INFO] CSV    : {csv_path}")
    print(f"[INFO] TABLE  : {table}")
    print(f"[INFO] MODE   : {args.mode}")

    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    print(f"[INFO] CSV rows={len(df)}, cols={len(df.columns)}")

    # 列名清洗 + 去重
    old, new = list(df.columns), dedupe_columns(df.columns)
    if old != new:
        print("[INFO] Renamed columns:")
        for o, n in zip(old, new):
            if o != n: print(f"    {o}  ->  {n}")
        df.columns = new

    df2, dtypes = guess_types(
        df,
        int_cols_hint={"claim_id","iid","iclaimid","icontractid","iservicerid"},
        date_cols_hint={"dtservicedate","dtentrydate"}
    )

    con = sqlite3.connect(DB_PATH)
    try:
        ensure_table(con, table, args.mode)
        create_table(con, table, dtypes)
        n = insert_bulk(con, table, df2)
        cnt = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        print(f"[OK] inserted {n} rows; table now has {cnt} rows")

        if args.index:
            idx_cols = [normalize_name(c) for c in args.index.split(",") if c.strip()]
            idx_name = "idx_" + table + "_" + "_".join(idx_cols)
            cols_sql = ", ".join([f'"{c}"' for c in idx_cols if c in df2.columns])
            if cols_sql:
                con.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ({cols_sql});')
                con.commit()
                print(f"[OK] index created: {idx_name} ON ({cols_sql})")
            else:
                print("[WARN] index columns not found; skip index.")
    finally:
        con.close()

if __name__ == "__main__":
    main()
