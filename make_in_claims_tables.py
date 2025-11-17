# src/make_in_claims_tables.py
import argparse
import sqlite3
from pathlib import Path
import pandas as pd 


def detect_key(cur, table: str) -> str:
    """在一个表里自动找“主键/claim_id”风格的列名。"""
    cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')]
    # 优先匹配常见 claim_id 风格
    for k in ("iid", "iId", "ICLAIMID", "claim_id", "Id", "ID"):
        for c in cols:
            if c.lower() == k.lower():
                return c
    # 否则随便挑一列（避开 rowid）
    for c in cols:
        if c.lower() != "rowid":
            return c
    return cols[0]


def find_col(cur, table: str, candidates) -> str | None:
    """在某个表里按候选列表找列名（先全匹配，再子串匹配）。"""
    cols = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')]
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


def table_exists(cur, name: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
        (name,),
    ).fetchone()
    return row is not None


def build_join(db_path: Path, claims_table: str, ids_table: str, out_table: str):
    """
    生成 out_table：

    out_table = claims_table ⨝ ids_table （按 claim_id 连接）
        + 额外两列：
          - sVinNumber  （来自 contract_vehicle）
          - invoice_hash（来自 p8_invoice_claims）

    English:
    Create `out_table` as a join of base claims and the IDs table,
    and additionally left-join contract_vehicle and p8_invoice_claims
    to bring in `sVinNumber` and `invoice_hash`.
    """
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    try:
        # ---------- 1) 基础 join：claims ⨝ ids ----------
        ck = detect_key(cur, claims_table)   # claims 的 claim_id 列
        ik = detect_key(cur, ids_table)      # ids 表里的 claim_id 列

        print(f"[info] join key: {claims_table}.{ck} ↔ {ids_table}.{ik}")

        # ---------- 2) 找 contract_vehicle 上的 VIN 列 ----------
        cv_table = "contract_vehicle"
        cv_join = ""
        select_extra = []

        if table_exists(cur, cv_table):
            # claims 表里与合同相关的列（通常 iContractId / icontractid）
            claims_contract_col = find_col(
                cur, claims_table, ["iContractId", "icontractid", "contract_id"]
            )
            cv_contract_col = find_col(
                cur, cv_table, ["iContractId", "icontractid", "contract_id"]
            )
            cv_vin_col = find_col(
                cur,
                cv_table,
                ["sVinNumber", "vin", "svin", "vehicle_vin"],
            )

            if claims_contract_col and cv_contract_col and cv_vin_col:
                # LEFT JOIN contract_vehicle
                cv_join = (
                    f'\nLEFT JOIN "{cv_table}" cv '
                    f'ON c."{claims_contract_col}" = cv."{cv_contract_col}"'
                )
                select_extra.append(f'cv."{cv_vin_col}" AS "sVinNumber"')
                print(
                    f"[info] will attach sVinNumber from {cv_table}"
                    f" ({cv_contract_col} -> {cv_vin_col})"
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

        if table_exists(cur, p8_table):
            p8_key_col = detect_key(cur, p8_table)
            p8_hash_col = find_col(cur, p8_table, ["invoice_hash"])

            if p8_key_col and p8_hash_col:
                p8_join = (
                    f'\nLEFT JOIN "{p8_table}" p8 '
                    f'ON c."{ck}" = p8."{p8_key_col}"'
                )
                select_extra.append(f'p8."{p8_hash_col}" AS "invoice_hash"')
                print(
                    f"[info] will attach invoice_hash from {p8_table}"
                    f" (key={p8_key_col})"
                )
            else:
                print(
                    f"[warn] p8_invoice_claims exists but cannot find key/hash columns; "
                    f"skip invoice_hash."
                )
        else:
            print("[info] table p8_invoice_claims not found; skip invoice_hash.")

        # ---------- 4) 组装 SELECT 语句 ----------
        extra_sql = ""
        if select_extra:
            extra_sql = ",\n            " + ",\n            ".join(select_extra)

        sql = f'''
        DROP TABLE IF EXISTS "{out_table}";

        CREATE TABLE "{out_table}" AS
        SELECT
            c.*,
            x.*{extra_sql}
        FROM "{claims_table}" c
        JOIN "{ids_table}"   x
              ON c."{ck}" = x."{ik}"{cv_join}{p8_join};
        '''
        # 去掉缩进里的多余空格
        cur.executescript(sql)
        con.commit()

        # ---------- 5) 建索引 ----------
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{out_table}_claim_id '
            f'ON "{out_table}"("{ck}")'
        )
        con.commit()

        n = cur.execute(f'SELECT COUNT(*) FROM "{out_table}"').fetchone()[0]
        print(
            f"[OK] {out_table} rows={n}  "
            f"(extra cols: {', '.join(['sVinNumber', 'invoice_hash'])})"
        )
        # === 额外：把结果表写成 CSV 文件 ===
        # English: also dump the result table into a CSV file next to data.sqlite
        out_dir = db_path.parent                 # 通常是 out/
        csv_path = out_dir / f"{out_table}.csv"  # 例如 out/p3p4p5p6p7p8_claims_in_claims.csv
        df = pd.read_sql(f'SELECT * FROM "{out_table}"', con)
        df.to_csv(csv_path, index=False)
        print(f"[OK] also wrote CSV -> {csv_path}")
    finally:
        con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db", required=True,
        help="SQLite 文件路径（例如 .\\out\\data.sqlite）"
    )
    ap.add_argument(
        "--base", required=True,
        help="claims 明细表名（通常是 claims）"
    )
    ap.add_argument(
        "--ids", required=True,
        help="只含交集 ID 的表名（例如 p3p4p5p6p7p8_claims）"
    )
    ap.add_argument(
        "--out", required=True,
        help="输出表名（例如 p3p4p5p6p7p8_claims_in_claims）"
    )
    args = ap.parse_args()
    build_join(Path(args.db), args.base, args.ids, args.out)
