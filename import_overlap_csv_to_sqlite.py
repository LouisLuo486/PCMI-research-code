#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → MSSQL 导入
- 清洗列名（小写、去空格、非法字符→下划线、重复列自动加后缀）
- 类型大致推断（整数/浮点/日期），其余按文本处理
- 始终将结果写入 MSSQL（连接字符串从 config/.env 读取），不再使用 out/data.sqlite
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()

# ---------- 列名清洗 ----------

def normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "col"


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


# ---------- 类型推断（仍然用 pandas 预处理，但交给 to_sql 建表） ----------

def guess_types(df: pd.DataFrame, int_cols_hint=None, date_cols_hint=None):
    """
    返回：清洗后的 DataFrame（值做了一些类型转换），第二个返回值 dtypes 保留但不再直接用于建表。
    """
    int_cols_hint = set(int_cols_hint or [])
    date_cols_hint = set(date_cols_hint or [])
    dtypes, out = {}, df.copy()

    # 先把 object 列里的空白统一成 None
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = (
                out[c]
                .astype(str)
                .str.strip()
                .replace({"": None, "nan": None, "NaN": None})
            )

    for c in out.columns:
        lc, s = c.lower(), out[c]

        # 明确提示是日期列
        if c in date_cols_hint or lc in {"dtservicedate", "dtentrydate"}:
            parsed = pd.to_datetime(s, errors="coerce")
            out[c] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
            dtypes[c] = "TEXT"
            continue

        # 明确提示 / 常见 id 列 → 尝试整数
        if c in int_cols_hint or lc in {
            "claim_id",
            "iid",
            "iclaimid",
            "icontractid",
            "iservicerid",
        }:
            num = pd.to_numeric(s, errors="coerce")
            if (num.dropna() % 1 == 0).mean() >= 0.95:
                out[c] = num.dropna().astype("Int64")
                dtypes[c] = "INTEGER"
                continue

        # 一般数值尝试
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.95:
            if (num.dropna() % 1 == 0).mean() >= 0.95:
                out[c] = num.dropna().astype("Int64")
                dtypes[c] = "INTEGER"
            else:
                out[c] = num
                dtypes[c] = "REAL"
        else:
            # 看看是不是日期格式
            sample = s.dropna().astype(str).head(200)
            if (
                not sample.empty
                and sample.str.match(r"\d{4}[-/]\d{2}[-/]\d{2}").mean() >= 0.8
            ):
                parsed = pd.to_datetime(s, errors="coerce")
                out[c] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)
            dtypes[c] = "TEXT"

    return out, dtypes


# ---------- MSSQL 辅助 ----------

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
        {"t": table},
    ).fetchall()
    return [r[0] for r in rows]


def create_index(conn, table: str, idx_cols: list[str]) -> None:
    """
    在 MSSQL 上为指定列建一个组合索引（如存在则跳过）。
    idx_cols 已是规范化后的列名；这里会用 INFORMATION_SCHEMA 映射实际大小写。
    """
    if not idx_cols:
        return

    cols = set(_get_columns(conn, table))
    if not cols:
        print(f"[WARN] table {table} has no columns, skip index creation")
        return

    lower_map = {c.lower(): c for c in cols}
    used = []
    for c in idx_cols:
        real = lower_map.get(c.lower())
        if real:
            used.append(real)

    if not used:
        print(f"[WARN] none of index columns {idx_cols} found in table {table}, skip index.")
        return

    idx_name = "idx_" + table + "_" + "_".join([normalize_name(c) for c in used])
    cols_sql = ", ".join(used)

    sql = f"""
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = '{idx_name}'
      AND object_id = OBJECT_ID('{table}')
)
BEGIN
    CREATE INDEX {idx_name} ON {table}({cols_sql});
END;
"""
    conn.exec_driver_sql(sql)
    print(f"[OK] index created (or existed): {idx_name} ON {table}({cols_sql})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="要导入的 CSV 路径")
    ap.add_argument("--table", required=True, help="目标表名（会做列名清洗）")
    ap.add_argument(
        "--mode",
        choices=["replace", "append"],
        default="replace",
        help="导入模式：replace=覆盖重建, append=追加",
    )
    ap.add_argument(
        "--index",
        default="claim_id",
        help="建索引列（逗号分隔，默认 claim_id；会自动做 normalize_name）",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")

    table = normalize_name(args.table)  # 表名也清洗，防止大小写/非法字符

    print(f"[INFO] CSV    : {csv_path}")
    print(f"[INFO] TABLE  : {table}")
    print(f"[INFO] MODE   : {args.mode}")

    # 读 CSV
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    print(f"[INFO] CSV rows={len(df)}, cols={len(df.columns)}")

    # 列名清洗 + 去重
    old, new = list(df.columns), dedupe_columns(df.columns)
    if old != new:
        print("[INFO] Renamed columns:")
        for o, n in zip(old, new):
            if o != n:
                print(f"    {o}  ->  {n}")
        df.columns = new

    # 预处理类型
    df2, _ = guess_types(
        df,
        int_cols_hint={"claim_id", "iid", "iclaimid", "icontractid", "iservicerid"},
        date_cols_hint={"dtservicedate", "dtentrydate"},
    )

    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # pandas.to_sql 会自动建表：
        #  - replace: DROP+CREATE
        #  - append : INSERT INTO
        df2.to_sql(
            table,
            con=conn,
            if_exists=args.mode,
            index=False,
            method="multi",
        )

        # 统计行数
        cnt = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table}")).scalar()
        print(f"[OK] inserted {len(df2)} rows; table '{table}' now has {cnt} rows")

        # 建索引
        if args.index:
            idx_cols = [normalize_name(c) for c in args.index.split(",") if c.strip()]
            create_index(conn, table, idx_cols)


if __name__ == "__main__":
    main()
