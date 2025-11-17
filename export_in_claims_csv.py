#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
from pathlib import Path

# 项目根目录：.../PCMI research
BASE = Path(__file__).resolve().parents[1]
DB   = BASE / "out" / "data.sqlite"
OUT  = BASE / "out" / "p3p4p5p6p7p8_claims_in_claims.csv"
TABLE = "p3p4p5p6p7p8_claims_in_claims"

def main():
    if not DB.exists():
        raise FileNotFoundError(f"DB not found: {DB}")
    with sqlite3.connect(DB) as conn:
        # 直接把整张表读出来（包含 seller_id / seller_name 等所有列）
        df = pd.read_sql(f'SELECT * FROM "{TABLE}"', conn)
    df.to_csv(OUT, index=False)
    print(f"[OK] 导出 {len(df)} 行到 {OUT}")

if __name__ == "__main__":
    main()
