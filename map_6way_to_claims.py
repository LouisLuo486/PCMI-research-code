# map_6way_to_claims.py  (robust key detection)
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # 项目根
CLAIMS_CSV = ROOT / "data" / "claims.csv"
SRC_CSV    = ROOT / "out"  / "p3p4p5p6p7p8_claims.csv"
OUT_DIR    = ROOT / "out"  / "mapped_claims"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / "p3p4p5p6p7p8_claims_in_claims.csv"

def find_key(df, candidates=("iid","iId","iclaimid","claim_id","id")):
    """在 df 里按候选键（大小写不敏感）找第一列命中者；若没有则用第一列但避开 rowid。"""
    lowers = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in lowers:
            return lowers[k.lower()]
    # 避开 rowid 之类
    for c in df.columns:
        if c.lower() != "rowid":
            return c
    return df.columns[0]

def to_id_set(series):
    # 去空格、转数字、转 int64
    return set(pd.to_numeric(series.astype(str).str.strip(), errors='coerce').dropna().astype('int64').tolist())

print("[PATH] claims.csv  =", CLAIMS_CSV)
print("[PATH] 6way csv    =", SRC_CSV)

# 1) 读 claims.csv（带 BOM 时用 utf-8-sig 更稳）
claims = pd.read_csv(CLAIMS_CSV, low_memory=False, encoding="utf-8-sig")
claim_key = find_key(claims, candidates=("iid","iId","iclaimid","claim_id","id"))
print(f"[INFO] detected claims key column = '{claim_key}'")
claims_ids = to_id_set(claims[claim_key])
# 统一成 claim_id 字段名
claims = claims.rename(columns={claim_key: "claim_id"})

# 2) 读 6 重交集 CSV，定位 iid 列（避开 ROWID）
src = pd.read_csv(SRC_CSV, low_memory=False, encoding="utf-8-sig")
src_key = find_key(src, candidates=("iid","iId","iclaimid","claim_id","id"))
print(f"[INFO] detected 6way key column   = '{src_key}'")
hit_ids = to_id_set(src[src_key])

print(f"[CHECK] ids_in_6way = {len(hit_ids)} ; ids_in_claims = {len(claims_ids)} ; overlap = {len(hit_ids & claims_ids)}")

# 3) 过滤 + 排列列顺序
front = ['claim_id','icontractid','sclaimnumber','sclaimstatus','dtservicedate','iservicerid']
for c in front:
    if c not in claims.columns:
        claims[c] = pd.NA

subset = claims[claims['claim_id'].isin(hit_ids)].copy()
subset = subset[front + [c for c in subset.columns if c not in front]]

subset.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] matched rows = {len(subset)} -> {OUT_CSV}")
