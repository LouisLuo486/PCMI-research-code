# src/make_in_claims_tables.py
import argparse
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from config import Config

cfg = Config()
OUT_DIR = Path(cfg.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers on MSSQL ----------

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


def detect_key(conn, table: str) -> str:
    """
    在一个表里自动找“主键/claim_id”风格的列名（基于列名模式）。
    """
    cols = _get_columns(conn, table)
    if not cols:
        raise RuntimeError(f"Table {table} has no columns")

    # 优先匹配常见 claim_id 风格
    for k in ("iid", "iId", "ICLAIMID", "claim_id", "Id", "ID"):
        for c in cols:
            if c.lower() == k.lower():
                return c

    # 否则随便挑一列
    for c in cols:
        if c.lower() != "rowid":
            return c
    return cols[0]


def find_col(conn, table: str, candidates) -> str | None:
    """
    在某个表里按候选列表找列名（先全匹配，再子串匹配）。
    """
    cols = _get_columns(conn, table)
    if not cols:
        return None

    # 精确匹配
    for cand in candidates:
        for c in cols:
            if cand.lower() == c.lower():
                return c

    # 子串匹配
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c

    return None


def table_exists(conn, name: str) -> bool:
    """
    检查 MSSQL 中是否存在指定表（不含 schema 的简单表名）。
    """
    row = conn.execute(
        sa.text(
            """
            SELECT 1
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = :t
            """
        ),
        {"t": name}
    ).fetchone()
    return row is not None


def build_join(claims_table: str, ids_table: str, out_table: str):
    """
    生成 out_table：

    out_table = claims_table ⨝ ids_table （按 claim_id 连接）
        + 额外几列（如果能找到的话）：
          - sVinNumber   （来自 contract_vehicle）
          - invoice_hash （来自 p8_invoice_claims）
          - seller_name  （来自 entity_sellers）

    English:
    Create `out_table` as a join of base claims and the IDs table,
    and additionally left-join contract_vehicle, p8_invoice_claims,
    and entity_sellers to bring in sVinNumber, invoice_hash, seller_name.
    """
    mssql_conn_str = cfg.mssql_conn_str
    if not mssql_conn_str:
        raise RuntimeError("MSSQL_CONN_STR not set in config/.env")

    engine = sa.create_engine(mssql_conn_str, fast_executemany=True)

    with engine.begin() as conn:
        # ---------- 1) 基础 join：claims ⨝ ids ----------
        ck = detect_key(conn, claims_table)   # claims 的 claim_id 列
        ik = detect_key(conn, ids_table)      # ids 表里的 claim_id 列

        print(f"[info] join key: {claims_table}.{ck} ↔ {ids_table}.{ik}")

        # ---------- 2) 找 contract_vehicle 上的 VIN 列 ----------
        cv_table = "contract_vehicle"
        cv_join = ""
        select_extra = []

        if table_exists(conn, cv_table):
            # claims 表里与合同相关的列（通常 iContractId / icontractid / contract_id）
            claims_contract_col = find_col(
                conn, claims_table, ["iContractId", "icontractid", "contract_id"]
            )
            cv_contract_col = find_col(
                conn, cv_table, ["iContractId", "icontractid", "contract_id"]
            )
            cv_vin_col = find_col(
                conn,
                cv_table,
                ["sVinNumber", "vin", "svin", "vehicle_vin"],
            )

            if claims_contract_col and cv_contract_col and cv_vin_col:
                # LEFT JOIN contract_vehicle
                cv_join = (
                    f"\n    LEFT JOIN {cv_table} AS cv "
                    f"ON c.{claims_contract_col} = cv.{cv_contract_col}"
                )
                select_extra.append(f"cv.{cv_vin_col} AS sVinNumber")
                print(
                    f"[info] will attach sVinNumber from {cv_table}"
                    f" ({claims_contract_col} -> {cv_vin_col})"
                )
            else:
                print(
                    f"[warn] contract_vehicle exists but cannot find contract/VIN columns; "
                    f"skip sVinNumber."
                )
        else:
            print("[info] table contract_vehicle not found; skip sVinNumber.")

        # ---------- 3) 找 p8_invoice_claims 上的 invoice_hash ----------
        p8_table = "p8_invoice_claims"
        p8_join = ""

        if table_exists(conn, p8_table):
            p8_key_col = detect_key(conn, p8_table)
            p8_hash_col = find_col(conn, p8_table, ["invoice_hash"])

            if p8_key_col and p8_hash_col:
                p8_join = (
                    f"\n    LEFT JOIN {p8_table} AS p8 "
                    f"ON c.{ck} = p8.{p8_key_col}"
                )
                select_extra.append(f"p8.{p8_hash_col} AS invoice_hash")
                print(
                    f"[info] will attach invoice_hash from {p8_table}"
                    f" (key={p8_key_col})"
                )
            else:
                print(
                    f"[warn] {p8_table} exists but cannot find key/hash columns; "
                    f"skip invoice_hash."
                )
        else:
            print("[info] table p8_invoice_claims not found; skip invoice_hash.")

        # ---------- 4) 找 entity_sellers 上的 seller_name ----------
        es_table = "entity_sellers"
        es_join = ""
        if table_exists(conn, es_table):
            # ids_table(x) 里 seller 主键列
            ids_seller_col = find_col(conn, ids_table, ["seller_id", "isellerid", "iSellerId"])
            # entity_sellers 主键 & 名称列
            es_id_col = find_col(conn, es_table, ["iid", "iId", "seller_id", "id"])
            es_name_col = find_col(
                conn,
                es_table,
                ["seller_name", "name", "legal_name", "display_name"],
            )

            if ids_seller_col and es_id_col and es_name_col:
                es_join = (
                    f"\n    LEFT JOIN {es_table} AS es "
                    f"ON x.{ids_seller_col} = es.{es_id_col}"
                )
                select_extra.append(f"es.{es_name_col} AS seller_name")
                print(
                    f"[info] will attach seller_name from {es_table} "
                    f"(x.{ids_seller_col} -> es.{es_name_col})"
                )
            else:
                print(
                    f"[warn] entity_sellers exists but cannot find id/name columns; "
                    f"skip seller_name."
                )
        else:
            print("[info] table entity_sellers not found; skip seller_name.")

        # ---------- 5) 组装 SELECT INTO 语句（MSSQL 语法） ----------
        extra_sql = ""
        if select_extra:
            extra_sql = ",\n    " + ",\n    ".join(select_extra)

        sql = f"""
IF OBJECT_ID('{out_table}', 'U') IS NOT NULL
    DROP TABLE {out_table};

SELECT
    c.*,
    x.*{extra_sql}
INTO {out_table}
FROM {claims_table} AS c
JOIN {ids_table}   AS x
    ON c.{ck} = x.{ik}{cv_join}{p8_join}{es_join};
"""
        conn.exec_driver_sql(sql)

        # ---------- 6) 建索引（如果不存在） ----------
        idx_name = f"idx_{out_table}_claim_id"
        idx_sql = f"""
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = '{idx_name}'
      AND object_id = OBJECT_ID('{out_table}')
)
BEGIN
    CREATE INDEX {idx_name} ON {out_table}({ck});
END;
"""
        conn.exec_driver_sql(idx_sql)

        # ---------- 7) 统计行数 ----------
        n = conn.execute(
            sa.text(f"SELECT COUNT(*) FROM {out_table}")
        ).scalar()
        print(
            f"[OK] {out_table} rows={n}  "
            f"(extra cols attempted: {', '.join(['sVinNumber', 'invoice_hash', 'seller_name'])})"
        )

        # ---------- 8) 导出 CSV ----------
        csv_path = OUT_DIR / f"{out_table}.csv"
        df = pd.read_sql(f"SELECT * FROM {out_table}", conn)
        df.to_csv(csv_path, index=False)
        print(f"[OK] also wrote CSV -> {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db",
        required=False,
        help="(ignored for MSSQL) 原来是 SQLite 文件路径，现在连接字符串从 config/.env 读取",
    )
    ap.add_argument(
        "--base",
        required=True,
        help="claims 明细表名（通常是 claims）",
    )
    ap.add_argument(
        "--ids",
        required=True,
        help="只含交集 ID 的表名（例如 p3p4p5p6p7p8_claims）",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="输出表名（例如 p3p4p5p6p7p8_claims_in_claims）",
    )
    args = ap.parse_args()

    build_join(args.base, args.ids, args.out)
