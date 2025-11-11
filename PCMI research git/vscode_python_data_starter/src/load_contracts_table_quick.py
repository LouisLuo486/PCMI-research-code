#!/usr/bin/env python3
import pandas as pd, sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB   = BASE/'out'/'data.sqlite'
CSV  = next(p for p in [BASE/'data'/'contracts.csv', BASE/'data'/'contracts (1).csv'] if p.exists())

df = pd.read_csv(CSV, dtype=str, low_memory=False)

def pick(df, cands):
    cols = {c.lower(): c for c in df.columns}
    for k in cands:
        if k.lower() in cols: return cols[k.lower()]
    for k in cands:
        for lc, orig in cols.items():
            if k.lower() in lc: return orig
    return None

cid = pick(df, ['iId','iid','contract_id','id','icontractid'])
cov = pick(df, ['iCoverageId','coverage_id','icoverageid','icoverageplanid','coverage_plan_id','iCoveragePlanId'])

if not cid or not cov:
    raise SystemExit(f"[contracts.csv] 找不到必要列：contract_id候选={['iId','iid','contract_id','id','icontractid']}，"
                     f"coverage_id候选={['iCoverageId','coverage_id','icoverageid','icoverageplanid','coverage_plan_id','iCoveragePlanId']}\n"
                     f"实际列：{list(df.columns)}")

out = (df[[cid, cov]]
       .rename(columns={cid:'contract_id', cov:'coverage_id'})
       .astype(str).apply(lambda s: s.str.strip())
       .dropna().drop_duplicates())

print("> contracts 预览：")
print(out.head())

DB.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB) as conn:
    conn.execute("DROP TABLE IF EXISTS contracts;")
    out.to_sql('contracts', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contracts_id  ON contracts(contract_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contracts_cov ON contracts(coverage_id);")
    n = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    print(f"> 已写入表 contracts，行数={n}")
