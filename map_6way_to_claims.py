#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
map_6way_to_claims.py (MSSQL version, robust key detection)

功能：
- 从 MSSQL 读取 claims 表 和 p3p4p5p6p7p8_claims 表；
- 自动检测两张表中的 claim 主键列（iid / claim_id / id 等）；
- 取 6-way 交集中的 claim_id，在 claims 里找到对应明细；
- 输出一个只包含这些 claim 的 CSV，列顺序前置常用字段：
    [claim_id, icontractid, sclaimnumber, sclaimstatus, dtservicedate, iservicerid, ...]

English:
- Load claims and six-way-intersection claims from MSSQL;
- Detect key columns in both tables;
- Filter claims table to ids present in six-way set;
- Dump to CSV under OUT_DIR/mapped_claims.
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()
OUT_DIR = Path(cfg.out_dir) / "mapped_claims"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims.csv"


def find_key(df: pd.DataFrame, candidates=("iid", "iId", "iclaimid", "claim_id", "id")) -> str:
    """
    在 df 里按候选键（大小写不敏感）找第一列命中者；
    若没有则用第一列但避开 rowid。

    English:
    Try to find a primary-key-like column by name; fall back to the first non-rowid column.
    """
    lowers = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in lowers:
            return lowers[k.lower()]
    for c in df.columns:
        if c.lower() != "rowid":
            return c
    return df.columns[0]


def to_id_set(series: pd.Series) -> set[int]:
    """
    将列值规范化为 int64 集合：
    - strip 空格
    - 尝试转数字
    - 丢掉 NaN
    """
    return set(
        pd.to_numeric(series.astype(str).str.strip(), errors="coerce")
        .dropna()
        .astype("int64")
        .tolist()
    )


def main():
    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)
    print(f"[PATH] MSSQL = {mssql_conn_str}")
    print(f"[PATH] out csv = {OUT_CSV}")

    # 1) 读 claims 表
    with engine.begin() as conn:
        try:
            claims = pd.read_sql("SELECT * FROM claims", conn)
        except Exception as e:
            raise RuntimeError(f"Failed to read table 'claims' from MSSQL: {e}")

        claim_key = find_key(claims, candidates=("iid", "iId", "iclaimid", "claim_id", "id"))
        print(f"[INFO] detected claims key column = '{claim_key}'")

        claims_ids = to_id_set(claims[claim_key])
        # 统一成 claim_id 字段名
        claims = claims.rename(columns={claim_key: "claim_id"})

        # 2) 读 6-way 交集表 p3p4p5p6p7p8_claims
        try:
            src = pd.read_sql("SELECT * FROM p3p4p5p6p7p8_claims", conn)
        except Exception as e:
            raise RuntimeError("Failed to read table 'p3p4p5p6p7p8_claims' from MSSQL: "
                               f"{e}")

    if src.empty:
        print("[WARN] p3p4p5p6p7p8_claims is empty; nothing to map.")
        src_key = "claim_id"
        hit_ids = set()
    else:
        src_key = find_key(src, candidates=("iid", "iId", "iclaimid", "claim_id", "id"))
        print(f"[INFO] detected 6way key column = '{src_key}'")
        hit_ids = to_id_set(src[src_key])

    print(
        f"[CHECK] ids_in_6way = {len(hit_ids)} ; "
        f"ids_in_claims = {len(claims_ids)} ; "
        f"overlap = {len(hit_ids & claims_ids)}"
    )

    # 3) 过滤 + 排列列顺序
    # 前置常用字段，不存在的就先补 NA
    front = [
        "claim_id",
        "icontractid",
        "sclaimnumber",
        "sclaimstatus",
        "dtservicedate",
        "iservicerid",
    ]
    for c in front:
        if c not in claims.columns:
            claims[c] = pd.NA

    subset = claims[claims["claim_id"].isin(hit_ids)].copy()
    # 保证列顺序：front 在前，其余在后
    subset = subset[front + [c for c in subset.columns if c not in front]]

    subset.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] matched rows = {len(subset)} -> {OUT_CSV}")


if __name__ == "__main__":
    main()
