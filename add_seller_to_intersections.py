#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
给 p3p4p5p6p7p8 交集结果补充 seller_id 和 seller_name，
从 data/*.csv 读数据，写回 out/*.csv 和 out/data.sqlite。
"""

import sqlite3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent          # .../src
PROJECT_ROOT = HERE.parent                      # .../vscode_python_data_starter
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"

# 1) 输入：交集结果
IN_CSV = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims.csv"

# 2) 输入：原始 CSV
CONTRACTS_CSV   = DATA_DIR / "contracts.csv"
COV_SELLER_CSV  = DATA_DIR / "coverage_plans_seller_inclusion.csv"
SELLERS_CSV     = DATA_DIR / "entity_sellers.csv"

# 3) 输出
OUT_CSV = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims_with_seller.csv"
OUT_DB  = OUT_DIR / "data.sqlite"


def main():
    # ---------- 1. 读交集 CSV ----------
    print(f"Reading intersection file from: {IN_CSV}")
    nodes = pd.read_csv(IN_CSV)

    # 处理 icontractid 大小写/命名问题
    lower_map = {c.lower(): c for c in nodes.columns}
    if "icontractid" not in lower_map:
        raise ValueError(
            f"交集文件列名里找不到 icontractid（不区分大小写）。实际列：{list(nodes.columns)}"
        )
    real_contract_col = lower_map["icontractid"]
    if real_contract_col != "icontractid":
        nodes = nodes.rename(columns={real_contract_col: "icontractid"})

    # ---------- 2. 读 contracts ----------
    print(f"Reading contracts from: {CONTRACTS_CSV}")
    contracts = pd.read_csv(CONTRACTS_CSV, low_memory=False)

    c_lower = {c.lower(): c for c in contracts.columns}
    for needed in ["iid", "icoverageid"]:
        if needed not in c_lower:
            raise ValueError(
                f"contracts.csv 中找不到列 {needed}（不区分大小写）。实际列：{list(contracts.columns)}"
            )

    contracts = contracts[[c_lower["iid"], c_lower["icoverageid"]]].rename(
        columns={c_lower["iid"]: "icontractid", c_lower["icoverageid"]: "iCoverageId"}
    )

    # ---------- 3. 读 coverage_plans_seller_inclusion ----------
    print(f"Reading coverage_plans_seller_inclusion from: {COV_SELLER_CSV}")
    cov_seller = pd.read_csv(COV_SELLER_CSV, low_memory=False)
    cs_lower = {c.lower(): c for c in cov_seller.columns}
    for needed in ["icoverageid", "isellerid"]:
        if needed not in cs_lower:
            raise ValueError(
                f"coverage_plans_seller_inclusion.csv 中找不到列 {needed}。实际列：{list(cov_seller.columns)}"
            )
    cov_seller = cov_seller[[cs_lower["icoverageid"], cs_lower["isellerid"]]].rename(
        columns={cs_lower["icoverageid"]: "iCoverageId", cs_lower["isellerid"]: "iSellerId"}
    ).drop_duplicates()

    # ---------- 4. 读 entity_sellers ----------
    print(f"Reading entity_sellers from: {SELLERS_CSV}")
    sellers = pd.read_csv(SELLERS_CSV, low_memory=False)
    s_lower = {c.lower(): c for c in sellers.columns}
    for needed in ["iid", "ssellername"]:
        if needed not in s_lower:
            raise ValueError(
                f"entity_sellers.csv 中找不到列 {needed}。实际列：{list(sellers.columns)}"
            )
    sellers = sellers[[s_lower["iid"], s_lower["ssellername"]]].rename(
        columns={s_lower["iid"]: "seller_id", s_lower["ssellername"]: "seller_name"}
    )

    # ---------- 5. 逐步 merge ----------
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

    # ---------- 7. 写进 out/data.sqlite ----------
    print(f"Writing to SQLite DB: {OUT_DB}")
    con_out = sqlite3.connect(OUT_DB)
    df.to_sql(
        "p3p4p5p6p7p8_claims_in_claims_with_seller",
        con_out,
        if_exists="replace",
        index=False
    )
    con_out.close()

    print("Done! seller_id 和 seller_name 已经写入 CSV 和 out/data.sqlite。")


if __name__ == "__main__":
    main()
