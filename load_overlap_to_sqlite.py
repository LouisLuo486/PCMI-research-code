#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 out/p3_p4_overlap_claims.csv 导入到 MSSQL，写成表 p3_p4_overlap，
并根据列存在性在 iId / iContractId / iServicerId 上创建索引。
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()

BASE      = Path(__file__).resolve().parents[1]
CSV_PATH  = BASE / "out" / "p3_p4_overlap_claims.csv"
TABLE     = "p3_p4_overlap"


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


def create_indexes(conn, table: str) -> None:
    """
    按字段存在性有选择地建索引（iId / iContractId / iServicerId），避免报错。
    在 MSSQL 上通过 sys.indexes 检查索引是否已存在。
    """
    cols = set(_get_columns(conn, table))
    if not cols:
        print(f"[warn] table {table} has no columns, skip index creation")
        return

    # 原脚本只针对这三个列名，这里保持行为一致
    idx_plan = [
        ("idx_p3p4_claim",    "iId"),
        ("idx_p3p4_contract", "iContractId"),
        ("idx_p3p4_servicer", "iServicerId"),
    ]

    lower_map = {c.lower(): c for c in cols}

    for idx_name, logical_col in idx_plan:
        # MSSQL 列名不区分大小写，这里用 lower 匹配
        col_actual = lower_map.get(logical_col.lower())
        if not col_actual:
            print(f"[info] skip index {idx_name}: column '{logical_col}' not found")
            continue

        idx_sql = f"""
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = '{idx_name}'
      AND object_id = OBJECT_ID('{table}')
)
BEGIN
    CREATE INDEX {idx_name} ON {table}({col_actual});
END;
"""
        conn.exec_driver_sql(idx_sql)
        print(f"[info] created (or existed) index {idx_name} on {table}({col_actual})")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print(f"[PATH] CSV = {CSV_PATH}")
    print(f"[INFO] target MSSQL table = {TABLE}")

    # 读 CSV，保持和原来类似的行为
    df = pd.read_csv(CSV_PATH, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()

    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # 先手动 DROP（to_sql(replace) 也会 drop，这里只是显式一点）
        drop_sql = f"""
IF OBJECT_ID('{TABLE}', 'U') IS NOT NULL
    DROP TABLE {TABLE};
"""
        conn.exec_driver_sql(drop_sql)
        print(f"[info] dropped old table {TABLE} if existed")

        # 覆盖式写入（和原来 sqlite 版 if_exists='replace' 语义一致）
        df.to_sql(
            TABLE,
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
        )
        print(f"[OK] wrote table {TABLE} to MSSQL")

        # 建索引
        create_indexes(conn, TABLE)

        # 统计行数
        total = conn.execute(sa.text(f"SELECT COUNT(*) FROM {TABLE}")).scalar()
        print(f"[OK] wrote {total} rows into table '{TABLE}' on MSSQL")


if __name__ == "__main__":
    main()
