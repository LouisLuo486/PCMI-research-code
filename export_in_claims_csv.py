#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 MSSQL 里的表 p3p4p5p6p7p8_claims_in_claims 导出到本地 CSV

- 通过 config.Config 读取 MSSQL 连接串和 out 目录
- 不再依赖本地 out/data.sqlite
"""

from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

# 加载配置（包含 mssql_conn_str, out_dir 等）
cfg = Config()
OUT_DIR = Path(cfg.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims.csv"
TABLE   = "p3p4p5p6p7p8_claims_in_claims"


def main():
    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env or Config.mssql_conn_str")

    # 建立 MSSQL 连接
    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # 直接把整张表读出来（包含 seller_id / seller_name 等所有列）
        df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)

    # 导出为 CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] 导出 {len(df)} 行到 {OUT_CSV}")


if __name__ == "__main__":
    main()
