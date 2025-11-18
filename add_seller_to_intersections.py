#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给 p3p4p5p6p7p8 交集结果补充 seller_id 和 seller_name：

- 输入：
    1) out/p3p4p5p6p7p8_claims_in_claims.csv    （交集结果 CSV）
    2) MSSQL 里的 contracts / coverage_plans_seller_inclusion / entity_sellers 三张表
- 输出：
    1) out/p3p4p5p6p7p8_claims_in_claims_with_seller.csv
    2) MSSQL 里的表 p3p4p5p6p7p8_claims_in_claims_with_seller
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

HERE = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = HERE.parent                      # 项目根
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 输入：交集结果（CSV）
IN_CSV = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims.csv"

# 3) 输出
OUT_CSV  = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims_with_seller.csv"
OUT_TABLE = "p3p4p5p6p7p8_claims_in_claims_with_seller"


def _create_indexes(conn, table: str) -> None:
    """
    在 MSSQL 上为结果表创建一些常用索引（如果列存在的话）：
      - (iid)         // 理赔主键
      - (seller_id)   // 卖家
    """
    cols = conn.execute(
        sa.text(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = :t
            """
        ),
        {"t": table},
    ).fetchall()
    if not cols:
        print(f"[WARN] table {table} has no columns, skip index creation.")
        return

    col_map = {r[0].lower(): r[0] for r in cols}

    plans = [
        ("idx_p3p4p5p6p7p8_claims_iid", ["iid"]),
        ("idx_p3p4p5p6p7p8_claims_seller", ["seller_id"]),
    ]

    for idx_name, want_cols in plans:
        real_cols = [col_map[c.lower()] for c in want_cols if c.lower() in col_map]
        if not real_cols:
            continue
        col_sql = ", ".join(real_cols)
        sql = f"""
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = '{idx_name}'
      AND object_id = OBJECT_ID('{table}')
)
BEGIN
    CREATE INDEX {idx_name} ON {table}({col_sql});
END;
"""
        conn.exec_driver_sql(sql)
        print(f"[OK] index created (or existed): {idx_name} ON {table}({col_sql})")


def main():
    cfg = Config()
    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    # ---------- 1. 读交集 CSV ----------
    print(f"Reading intersection file from: {IN_CSV}")
    nodes = pd.read_csv(IN_CSV, low_memory=False)

    # 处理 icontractid 大小写/命名问题
    lower_map = {c.lower(): c for c in nodes.columns}
    if "icontractid" not in lower_map:
        raise ValueError(
            f"交集文件列名里找不到 icontractid（不区分大小写）。实际列：{list(nodes.columns)}"
        )
    real_contract_col = lower_map["icontractid"]
    if real_contract_col != "icontractid":
        nodes = nodes.rename(columns={real_contract_col: "icontractid"})

    # ---------- 2~4. 从 MSSQL 读三张维表 ----------
    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # contracts
        print("Reading contracts from MSSQL table: contracts")
        contracts = pd.read_sql("SELECT * FROM contracts", conn)
        c_lower = {c.lower(): c for c in contracts.columns}
        for needed in ["iid", "icoverageid"]:
            if needed not in c_lower:
                raise ValueError(
                    f"contracts 表中找不到列 {needed}（不区分大小写）。实际列：{list(contracts.columns)}"
                )
        contracts = contracts[[c_lower["iid"], c_lower["icoverageid"]]].rename(
            columns={c_lower["iid"]: "icontractid", c_lower["icoverageid"]: "iCoverageId"}
        )

        # coverage_plans_seller_inclusion
        print("Reading coverage_plans_seller_inclusion from MSSQL table: coverage_plans_seller_inclusion")
        cov_seller = pd.read_sql("SELECT * FROM coverage_plans_seller_inclusion", conn)
        cs_lower = {c.lower(): c for c in cov_seller.columns}
        for needed in ["icoverageid", "isellerid"]:
            if needed not in cs_lower:
                raise ValueError(
                    f"coverage_plans_seller_inclusion 表中找不到列 {needed}。实际列：{list(cov_seller.columns)}"
                )
        cov_seller = cov_seller[
            [cs_lower["icoverageid"], cs_lower["isellerid"]]
        ].rename(
            columns={
                cs_lower["icoverageid"]: "iCoverageId",
                cs_lower["isellerid"]: "iSellerId",
            }
        ).drop_duplicates()

        # entity_sellers
        print("Reading entity_sellers from MSSQL table: entity_sellers")
        sellers = pd.read_sql("SELECT * FROM entity_sellers", conn)
        s_lower = {c.lower(): c for c in sellers.columns}
        for needed in ["iid", "ssellername"]:
            if needed not in s_lower:
                raise ValueError(
                    f"entity_sellers 表中找不到列 {needed}。实际列：{list(sellers.columns)}"
                )
        sellers = sellers[
            [s_lower["iid"], s_lower["ssellername"]]
        ].rename(
            columns={
                s_lower["iid"]: "seller_id",
                s_lower["ssellername"]: "seller_name",
            }
        )

    # ---------- 5. 用 pandas 逐步 merge ----------
    print("Merging contracts (icontractid -> iCoverageId)...")
    df = nodes.merge(contracts, on="icontractid", how="left")

    print("Merging coverage_plans_seller_inclusion (iCoverageId -> iSellerId)...")
    df = df.merge(cov_seller, on="iCoverageId", how="left")

    print("Merging entity_sellers (iSellerId -> seller_name)...")
    df = df.merge(sellers, left_on="iSellerId", right_on="seller_id", how="left")

    # 只保留：原始列 + seller_id + seller_name
    keep_cols = list(nodes.columns) + ["seller_id", "seller_name"]
    df = df[keep_cols]

    # ---------- 6. 写新的 CSV ----------
    print(f"Writing updated CSV with seller info to: {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)

    # ---------- 7. 写回 MSSQL ----------
    print(f"Writing result table to MSSQL: {OUT_TABLE}")
    with engine.begin() as conn:
        # 覆盖式写入
        df.to_sql(
            OUT_TABLE,
            con=conn,
            if_exists="replace",
            index=False,
            method="multi",
        )
        # 建索引（如果相关列存在）
        _create_indexes(conn, OUT_TABLE)

        n = conn.execute(sa.text(f"SELECT COUNT(*) FROM {OUT_TABLE}")).scalar()
        print(f"[OK] table {OUT_TABLE} rows={n}")

    print("Done! seller_id 和 seller_name 已经写入 CSV 和 MSSQL 表。")


if __name__ == "__main__":
    main()
