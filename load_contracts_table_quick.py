#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 data/contracts.csv 或 data/contracts (1).csv 里抽取 contract_id / coverage_id，
写入 MSSQL 表 contracts，并在 contract_id / coverage_id 上建索引。
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()

BASE = Path(__file__).resolve().parents[1]

# 优先使用 contracts.csv，找不到再用 contracts (1).csv
CANDIDATES = [
    BASE / "data" / "contracts.csv",
    BASE / "data" / "contracts (1).csv",
]

CSV = next((p for p in CANDIDATES if p.exists()), None)
if CSV is None:
    raise FileNotFoundError(f"找不到 contracts CSV，尝试过：{CANDIDATES}")

print(f"[PATH] using CSV = {CSV}")

# 读原始 contracts CSV
df = pd.read_csv(CSV, dtype=str, low_memory=False, encoding="utf-8-sig")


def pick(df: pd.DataFrame, cands) -> str | None:
    """
    在 df 里按候选名找列：先精确匹配（忽略大小写），再子串匹配。
    """
    cols = {c.lower(): c for c in df.columns}
    # 精确匹配
    for k in cands:
        if k.lower() in cols:
            return cols[k.lower()]
    # 子串匹配
    for k in cands:
        kl = k.lower()
        for lc, orig in cols.items():
            if kl in lc:
                return orig
    return None


cid = pick(df, ['iId', 'iid', 'contract_id', 'id', 'icontractid'])
cov = pick(df, ['iCoverageId', 'coverage_id', 'icoverageid',
                'icoverageplanid', 'coverage_plan_id', 'iCoveragePlanId'])

if not cid or not cov:
    raise SystemExit(
        f"[contracts.csv] 找不到必要列：\n"
        f"  contract_id 候选 = {['iId','iid','contract_id','id','icontractid']}\n"
        f"  coverage_id 候选 = {['iCoverageId','coverage_id','icoverageid','icoverageplanid','coverage_plan_id','iCoveragePlanId']}\n"
        f"  实际列：{list(df.columns)}"
    )

# 只保留两列并清洗
out = (
    df[[cid, cov]]
    .rename(columns={cid: 'contract_id', cov: 'coverage_id'})
    .astype(str)
)

for c in out.columns:
    out[c] = out[c].str.strip()

out = out.dropna().drop_duplicates()

print("> contracts 预览：")
print(out.head())


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
    在 MSSQL 上为 contracts(contract_id) / contracts(coverage_id) 建索引（如果列存在）。
    """
    cols = set(_get_columns(conn, table))
    if not cols:
        print(f"[warn] table {table} has no columns, skip index creation")
        return

    lower_map = {c.lower(): c for c in cols}

    plan = [
        ("idx_contracts_id",  "contract_id"),
        ("idx_contracts_cov", "coverage_id"),
    ]

    for idx_name, logical_col in plan:
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
        print(f"[info] created (or existed) index {idx_name} ON {table}({col_actual})")


def main():
    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)
    table = "contracts"

    with engine.begin() as conn:
        # 先删旧表（和原来 SQLite 里 DROP TABLE 行为对应）
        drop_sql = f"""
IF OBJECT_ID('{table}', 'U') IS NOT NULL
    DROP TABLE {table};
"""
        conn.exec_driver_sql(drop_sql)
        print(f"[info] dropped old table {table} if existed")

        # 写入 MSSQL
        out.to_sql(
            table,
            con=conn,
            if_exists="replace",   # 覆盖式
            index=False,
            method="multi",
        )
        print(f"[OK] wrote table {table} to MSSQL")

        # 建索引
        create_indexes(conn, table)

        # 统计行数
        n = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"> 已写入表 {table}，行数 = {n}")


if __name__ == "__main__":
    main()
