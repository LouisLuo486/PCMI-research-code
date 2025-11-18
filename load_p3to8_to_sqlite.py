#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 out/p3p4p5p6p7p8_claims.csv 导入到 MSSQL 中，写成表 p3p4p5p6p7p8_claims，
并根据实际存在的列创建索引（claim_id / vin / servicer / date）。
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()

BASE  = Path(__file__).resolve().parents[1]
CSV   = BASE / 'out' / 'p3p4p5p6p7p8_claims.csv'
TABLE = 'p3p4p5p6p7p8_claims'   # 你也可以改成 'p345678_claims'

def _get_columns(conn, table: str) -> list[str]:
    """
    从 MSSQL INFORMATION_SCHEMA.COLUMNS 读取某表的全部列名。
    """
    rows = conn.execute(
        sa.text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = :t
            ORDER BY ORDINAL_POSITION
            """
        ),
        {"t": table}
    ).fetchall()
    return [r[0] for r in rows]


def create_indexes(conn, table: str):
    """
    动态检查列再建索引，避免列不存在时报错。
    在 MSSQL 上使用 sys.indexes + IF NOT EXISTS。
    """
    cols = set(_get_columns(conn, table))
    if not cols:
        print(f"[warn] table {table} has no columns, skip index creation")
        return

    plan = [
        ("idx_p3to8_claim",   "claim_id"),
        ("idx_p3to8_vin",     "vin"),
        ("idx_p3to8_serv",    "servicer_id"),
        ("idx_p3to8_date",    "service_date"),
    ]

    # 尝试一些常见“别名”列名（不区分大小写）
    def _find_alias(targets: set[str]) -> str | None:
        for c in cols:
            if c.lower() in targets:
                return c
        return None

    alias_map = {
        "servicer_id": _find_alias(
            {"servicer_id", "iservicerid", "iservicer", "iservicerid", "iservicer", "servicer"}
        ),
        "service_date": _find_alias(
            {
                "service_date", "servicedate", "dservicedate",
                "claim_date", "dclaimdate", "created_at", "dentrydate"
            }
        ),
    }

    for idx_name, logical_col in plan:
        # 优先用同名列，否则用 alias_map 里的匹配结果
        use_col = None
        for c in cols:
            if c.lower() == logical_col.lower():
                use_col = c
                break
        if use_col is None:
            use_col = alias_map.get(logical_col)

        if use_col and use_col in cols:
            idx_sql = f"""
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = '{idx_name}'
      AND object_id = OBJECT_ID('{table}')
)
BEGIN
    CREATE INDEX {idx_name} ON {table}({use_col});
END;
"""
            conn.exec_driver_sql(idx_sql)
            print(f"[info] created (or existed) index {idx_name} on {table}({use_col})")
        else:
            print(f"[info] skip index {idx_name}: no suitable column for '{logical_col}'")


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"CSV 不存在：{CSV}")

    print(f"[PATH] CSV = {CSV}")
    print(f"[INFO] target MSSQL table = {TABLE}")

    # 读 CSV（和你原来一样，dtype=str + 去空白）
    df = pd.read_csv(CSV, dtype=str, low_memory=False, encoding="utf-8-sig")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # 先把旧表删掉（可选；to_sql(if_exists='replace') 本身也会 drop）
        drop_sql = f"""
IF OBJECT_ID('{TABLE}', 'U') IS NOT NULL
    DROP TABLE {TABLE};
"""
        conn.exec_driver_sql(drop_sql)
        print(f"[info] dropped old table {TABLE} if existed")

        # 写入 MSSQL
        df.to_sql(
            TABLE,
            con=conn,
            if_exists='replace',   # 保持和原脚本语义一致：覆盖旧表
            index=False,
            method='multi'
        )
        print(f"[OK] wrote table {TABLE} to MSSQL")

        # 建索引
        create_indexes(conn, TABLE)

        # 统计行数
        n = conn.execute(sa.text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
        print(f"[OK] table '{TABLE}' rows = {n}")


if __name__ == "__main__":
    main()
