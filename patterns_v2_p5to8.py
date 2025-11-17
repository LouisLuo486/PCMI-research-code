#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patterns v2 (start from P5; legacy P1–P4 not used):
- P5: VIN Burst (short-window multi-claims per VIN)
- P6: Cross-shop Repairs (short-window multi-servicers per VIN)
- P7: Coordinated Clusters (servicer×seller high-density pairs)
- P8: Invoice Cloning (exact part-list signature reuse)

Outputs: CSVs under out/, summary_counts_v2.csv, and optional write-back to SQLite tables.
"""

import os, sqlite3, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

# ================== Tunables ==================
# P5
P5_WINDOW_DAYS       = 14
P5_MIN_CLAIMS        = 3

# P6
P6_WINDOW_DAYS       = 30
P6_MIN_SERVICERS     = 2

# P7 (pair density thresholds)
P7_MIN_DISTINCT_VINS = 30
P7_MIN_DISTINCT_PH   = 10  # 若无 policyholder 列会自动降级

# P8（发票克隆）
P8_MIN_DUP_HASH      = 2
PRICE_COL_CANDS      = ['unit_price','price','unitprice','nunitprice','nprice','amount','line_price','nlabouramount']
QTY_COL_CANDS        = ['qty','quantity','nqty','nquantity','count','num']
PART_COL_CANDS       = ['part_code','partnumber','part_no','spartno','spartcode','part','part_code_id','partid','sparepart','partdesc','part_description']

# ================== Paths ==================
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH  = BASE_DIR / 'out' / 'data.sqlite'
OUT_DIR  = DB_PATH.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"> DB: {DB_PATH}  exists={DB_PATH.exists()}")

# ================== Helpers ==================
def table_exists(conn, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)"
    try:
        return pd.read_sql(q, conn, params=[name]).shape[0] > 0
    except Exception:
        return False

def pick_col(df: pd.DataFrame, candidates, required=True, tag="", exclude=None):
    if df is None or df.empty:
        if required: raise KeyError(f"[{tag}] empty df")
        return None
    exclude = [x.lower() for x in (exclude or [])]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in cols_lower:
            orig = cols_lower[lc]
            if any(e in orig.lower() for e in exclude): continue
            print(f"  - [{tag}] use column: {orig}")
            return orig
    for c in candidates:
        lc = c.lower()
        for cl, orig in cols_lower.items():
            if lc in cl and not any(e in cl for e in exclude):
                print(f"  - [{tag}] use (substring) column: {orig}")
                return orig
    if required:
        print(f"!!! [{tag}] candidates not found: {candidates}\ncols={list(df.columns)}")
        raise KeyError(f"[{tag}] column not found")
    return None

def as_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce')

def save_csv(df: pd.DataFrame, name: str) -> Path:
    p = OUT_DIR / name
    df.to_csv(p, index=False)
    print(f"  -> saved: {p}")
    return p

def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8','ignore')).hexdigest()

# ================== Load tables ==================
with sqlite3.connect(DB_PATH) as conn:
    tabs = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].str.lower().tolist()
    print("> tables:", tabs)

    def read_if(tn):
        return pd.read_sql(f"SELECT * FROM {tn}", conn) if (tn and table_exists(conn, tn)) else pd.DataFrame()

    t_claims   = 'claims' if 'claims' in tabs else next((t for t in tabs if 'claims' in t or t=='claim'), None)
    t_cdet     = 'claim_details' if 'claim_details' in tabs else next((t for t in tabs if 'claim' in t and 'detail' in t), None)
    t_contracts= 'contracts' if 'contracts' in tabs else next((t for t in tabs if 'contract' in t and 'vehicle' not in t), None)
    t_cveh     = 'contract_vehicle' if 'contract_vehicle' in tabs else next((t for t in tabs if 'vehicle' in t), None)
    t_serv     = 'entity_servicers' if 'entity_servicers' in tabs else next((t for t in tabs if 'servicer' in t), None)
    t_seller   = 'entity_sellers' if 'entity_sellers' in tabs else next((t for t in tabs if 'seller' in t), None)
    t_cpsi     = 'coverage_plans_seller_inclusion' if 'coverage_plans_seller_inclusion' in tabs else next((t for t in tabs if 'coverage' in t and 'seller' in t), None)

    claims     = read_if(t_claims)
    cdet       = read_if(t_cdet)
    contracts  = read_if(t_contracts)
    cveh       = read_if(t_cveh)
    serv       = read_if(t_serv)
    seller     = read_if(t_seller)
    cpsi       = read_if(t_cpsi)

if claims.empty:
    raise RuntimeError("claims table not found or empty")

# ===== pick columns =====
col_claim_id   = pick_col(claims, ['iid','iId','claim_id','id'], tag='claims.id')
col_claim_date = pick_col(claims, ['service_date','servicedate','dservicedate','claim_date','dclaimdate','created_at','dentrydate'], required=False, tag='claims.date')
col_contract_id= pick_col(claims, ['icontractid','iContractId','contract_id'], required=False, tag='claims.contract')
col_servicer   = pick_col(claims, ['iservicerid','iServicerId','servicer_id'], required=False, tag='claims.servicer')

if col_claim_date:
    claims[col_claim_date] = to_dt(claims[col_claim_date])

# VIN map（优先 contract_vehicle）
vin_map = pd.DataFrame()
if not cveh.empty:
    col_v_contract = pick_col(cveh, ['icontractid','contract_id','iContractId'], tag='cveh.contract')
    col_vin        = pick_col(cveh, ['vin','svin','vehicle_vin'], tag='cveh.vin')
    vin_map = cveh[[col_v_contract, col_vin]].dropna().copy()
    vin_map[col_v_contract] = as_str(vin_map[col_v_contract]); vin_map[col_vin] = as_str(vin_map[col_vin])
elif not contracts.empty:
    col_v_contract = pick_col(contracts, ['iid','iId','contract_id','id'], tag='contracts.id')
    col_vin        = pick_col(contracts, ['vin','svin','vehicle_vin'], required=False, tag='contracts.vin')
    if col_vin:
        vin_map = contracts[[col_v_contract, col_vin]].dropna().copy()
        vin_map[col_v_contract] = as_str(vin_map[col_v_contract]); vin_map[col_vin] = as_str(vin_map[col_vin])

# policyholder（可选）
ph_map = pd.DataFrame()
if not contracts.empty:
    col_c_id = pick_col(contracts, ['iid','iId','contract_id','id'], tag='contracts.id(reuse)')
    col_ph   = pick_col(contracts, ['policyholder_id','icustomerid','customer_id','ipolicyholderid'], required=False, tag='contracts.ph')
    if col_ph:
        ph_map = contracts[[col_c_id, col_ph]].dropna().copy()
        ph_map[col_c_id] = as_str(ph_map[col_c_id]); ph_map[col_ph] = as_str(ph_map[col_ph])

# coverage→seller（供 P7 用）
cc = pd.DataFrame()
if (not cpsi.empty) and (col_contract_id is not None):
    col_cov_on_contracts = pick_col(contracts, ['icoverageid','coverage_id','coverage_plan_id','iCoverageId','iCoveragePlanId'], required=False, tag='contracts.coverage')
    col_cpsi_cov = pick_col(cpsi, ['coverage_id','icoverageid','coverage_plan_id','icoverageplanid','iCoverageId'], tag='cpsi.cov')
    col_cpsi_sel = pick_col(cpsi, ['seller_id','isellerid','iSellerId'], tag='cpsi.seller')
    if col_cov_on_contracts:
        _c = claims[[col_claim_id, col_contract_id]].dropna().copy()
        _c[col_claim_id] = as_str(_c[col_claim_id]); _c[col_contract_id] = as_str(_c[col_contract_id])
        _k = contracts[[col_c_id, col_cov_on_contracts]].dropna().copy()
        _k[col_c_id] = as_str(_k[col_c_id]); _k[col_cov_on_contracts] = as_str(_k[col_cov_on_contracts])
        cc = _c.merge(_k, left_on=col_contract_id, right_on=col_c_id, how='left')
        if not cc.empty:
            cc = cc.merge(
                cpsi[[col_cpsi_cov, col_cpsi_sel]].assign(
                    **{col_cpsi_cov: as_str(cpsi[col_cpsi_cov]), col_cpsi_sel: as_str(cpsi[col_cpsi_sel])}
                ),
                left_on=col_cov_on_contracts, right_on=col_cpsi_cov, how='left'
            )

# attach VIN/policyholder/servicer/date to claims
claims_min = claims[[col_claim_id]].copy()
claims_min[col_claim_id] = as_str(claims_min[col_claim_id])
if col_claim_date: claims_min[col_claim_date] = claims[col_claim_date]
if col_contract_id: claims_min[col_contract_id] = as_str(claims[col_contract_id])
if col_servicer:    claims_min[col_servicer]    = as_str(claims[col_servicer])

if not vin_map.empty and col_contract_id:
    claims_min = claims_min.merge(vin_map, left_on=col_contract_id, right_on=col_v_contract, how='left').rename(columns={col_vin:'vin'})
if not ph_map.empty and col_contract_id:
    claims_min = claims_min.merge(ph_map, left_on=col_contract_id, right_on=col_c_id, how='left').rename(columns={ph_map.columns[1]:'policyholder_id'})

# =========================================================
# P5: VIN Burst
p5_bursts = pd.DataFrame()
if 'vin' in claims_min.columns and col_claim_date:
    df = claims_min.dropna(subset=['vin', col_claim_date]).copy()
    df['vin'] = as_str(df['vin'])
    df = df.sort_values(['vin', col_claim_date])

    rows = []
    for vin, g in df.groupby('vin'):
        dates = g[col_claim_date].values
        l = 0
        for r in range(len(g)):
            while (dates[r] - dates[l]).astype('timedelta64[D]').astype(int) > P5_WINDOW_DAYS:
                l += 1
            cnt = r - l + 1
            if cnt >= P5_MIN_CLAIMS:
                sub = g.iloc[l:r+1]
                rows.append({
                    'vin': vin,
                    'window_start': sub[col_claim_date].min(),
                    'window_end':   sub[col_claim_date].max(),
                    'num_claims':   int(cnt),
                    'distinct_servicers': int(sub[col_servicer].nunique()) if col_servicer else np.nan,
                    'claim_ids': "|".join(as_str(sub[col_claim_id]).tolist())
                })
    p5_bursts = pd.DataFrame(rows).drop_duplicates().sort_values(['num_claims','distinct_servicers'], ascending=False)

p5_path = save_csv(p5_bursts, 'p5_vin_bursts.csv')

# =========================================================
# P6: Cross-shop Repairs
p6_cross = pd.DataFrame()
if 'vin' in claims_min.columns and col_claim_date and col_servicer:
    df = claims_min.dropna(subset=['vin', col_claim_date, col_servicer]).copy()
    df['vin'] = as_str(df['vin']); df[col_servicer] = as_str(df[col_servicer])
    df = df.sort_values(['vin', col_claim_date])

    rows = []
    for vin, g in df.groupby('vin'):
        dates = g[col_claim_date].values
        l = 0
        for r in range(len(g)):
            while (dates[r] - dates[l]).astype('timedelta64[D]').astype(int) > P6_WINDOW_DAYS:
                l += 1
            sub = g.iloc[l:r+1]
            s_cnt = sub[col_servicer].nunique()
            if s_cnt >= P6_MIN_SERVICERS:
                rows.append({
                    'vin': vin,
                    'window_start': sub[col_claim_date].min(),
                    'window_end':   sub[col_claim_date].max(),
                    'distinct_servicers': int(s_cnt),
                    'num_claims':   int(len(sub)),
                    'servicer_ids': "|".join(sorted(as_str(sub[col_servicer]).unique().tolist())),
                    'claim_ids':    "|".join(as_str(sub[col_claim_id]).tolist())
                })
    p6_cross = pd.DataFrame(rows).drop_duplicates().sort_values(['distinct_servicers','num_claims'], ascending=False)

p6_path = save_csv(p6_cross, 'p6_cross_shop.csv')

# =========================================================
# P7: Coordinated Clusters (servicer×seller high-density pairs)
# 这里同时算两样：
# 1) pair 级的统计表 p7_pairs（写到 p7_cluster_pairs.csv）
# 2) 被这些高密度 pair 覆盖到的 claim 集合 p7_pair_claims_set（后面 P3–P8 交集要用）
# =========================================================

p7_pairs = pd.DataFrame()
p7_pair_claims_set = set()

if not cc.empty and 'vin' in claims_min.columns and col_servicer:

    # ---------- 1) 先构造 claim 级明细 tmp：claim_id, servicer_id, seller_id, vin, policyholder(optional) ----------
    # 从 cc 里取 claim_id 和 seller
    tmp = cc[[col_claim_id, col_cpsi_sel]].dropna().copy()

    # 把 id 列转成整数类型（不用字符串 + strip，省内存）
    tmp[col_claim_id] = pd.to_numeric(tmp[col_claim_id], errors='coerce')
    tmp[col_cpsi_sel] = pd.to_numeric(tmp[col_cpsi_sel], errors='coerce')
    tmp = tmp.dropna(subset=[col_claim_id, col_cpsi_sel]).astype(
        {col_claim_id: 'Int64', col_cpsi_sel: 'Int64'}
    )

    # 从 claims_min 附上 vin / servicer / policyholder
    attach_cols = [col_claim_id, 'vin', col_servicer]
    if 'policyholder_id' in claims_min.columns:
        attach_cols.append('policyholder_id')

    attach = claims_min[attach_cols].dropna(subset=['vin', col_servicer]).copy()
    attach['vin'] = attach['vin'].astype(str)

    attach[col_claim_id] = pd.to_numeric(attach[col_claim_id], errors='coerce')
    attach[col_servicer] = pd.to_numeric(attach[col_servicer], errors='coerce')
    attach = attach.dropna(subset=[col_claim_id, col_servicer]).astype(
        {col_claim_id: 'Int64', col_servicer: 'Int64'}
    )

    # inner join：只要既在 cc 里又在 claims_min 里的 claim
    tmp = tmp.merge(attach, on=col_claim_id, how='inner')

    # ---------- 2) pair 级聚合（得到 p7_pairs） ----------
    grp = tmp.groupby([col_servicer, col_cpsi_sel]).agg(
        distinct_vins=('vin', 'nunique'),
        distinct_policyholders=(
            ('policyholder_id', 'nunique')
            if 'policyholder_id' in tmp.columns
            else ('vin', 'size')
        ),
        claims=('vin', 'size')
    ).reset_index().rename(
        columns={col_servicer: 'servicer_id', col_cpsi_sel: 'seller_id'}
    )

    if 'policyholder_id' in tmp.columns:
        p7_pairs = grp[
            (grp['distinct_vins'] >= P7_MIN_DISTINCT_VINS)
            & (grp['distinct_policyholders'] >= P7_MIN_DISTINCT_PH)
        ].copy()
    else:
        p7_pairs = grp[grp['distinct_vins'] >= P7_MIN_DISTINCT_VINS].copy()

    p7_pairs = p7_pairs.sort_values(['distinct_vins', 'claims'], ascending=False)
    save_csv(p7_pairs, 'p7_cluster_pairs.csv')

    # ---------- 3) claim 级覆盖集合 p7_pair_claims_set ----------
    if not p7_pairs.empty:
        # 先拿到所有 claim_id + (servicer_id, seller_id)
        claim_pairs = tmp[[col_claim_id, col_servicer, col_cpsi_sel]].copy()
        claim_pairs = claim_pairs.rename(
            columns={col_servicer: 'servicer_id', col_cpsi_sel: 'seller_id'}
        )

        # 只保留落在高密度 pair 列表里的那些 (servicer, seller)
        key = ['servicer_id', 'seller_id']
        pairs_key = p7_pairs[key].drop_duplicates()

        hit = claim_pairs.merge(pairs_key, on=key, how='inner')

        # 转回字符串，后面做集合交集统一用 str
        p7_pair_claims_set = set(hit[col_claim_id].astype(str))

else:
    # 没有 cc 或没有 vin/servicer，就输出空表避免后面读文件报错
    save_csv(pd.DataFrame(), 'p7_cluster_pairs.csv')



# =========================================================
# P8: Invoice Cloning
p8_hash = pd.DataFrame(); p8_claims = pd.DataFrame()
if not cdet.empty:
    col_cd_claim = pick_col(cdet, ['iclaimid','claim_id'], tag='cd.claim')
    col_part = pick_col(cdet, PART_COL_CANDS, required=False, tag='cd.part')
    col_qty  = pick_col(cdet, QTY_COL_CANDS, required=False, tag='cd.qty')
    col_price= pick_col(cdet, PRICE_COL_CANDS, required=False, tag='cd.price', exclude=['qty','quantity','hours','rate'])
    cols = [c for c in [col_part, col_qty, col_price] if c]

    df = cdet[[col_cd_claim] + cols].copy()
    for c in df.columns: df[c] = df[c].astype(str).str.strip().str.lower()

    def norm_line(row):
        p = (row.get(col_part) or '').replace(' ', '')
        q = row.get(col_qty) or ''
        pr= row.get(col_price) or ''
        return f"{p}:{q}:{pr}"

    df['_line'] = df.apply(norm_line, axis=1)
    lines = (df.groupby(col_cd_claim)['_line']
               .apply(lambda s: "|".join(sorted([x for x in s if x]))).reset_index(name='invoice_signature'))
    lines['invoice_hash'] = lines['invoice_signature'].apply(md5)

    meta_cols = [col_claim_id]
    if 'vin' in claims_min.columns: meta_cols.append('vin')
    if col_servicer: meta_cols.append(col_servicer)
    meta = claims_min[meta_cols].copy()
    meta[col_claim_id] = as_str(meta[col_claim_id])

    p8 = lines.rename(columns={col_cd_claim: col_claim_id}).merge(meta, on=col_claim_id, how='left')

    hstat = p8.groupby('invoice_hash').agg(
        num_claims   =(col_claim_id,'nunique'),
        distinct_vins=('vin','nunique') if 'vin' in p8.columns else (col_claim_id,'size'),
        distinct_servicers=(col_servicer,'nunique') if col_servicer in p8.columns else (col_claim_id,'size'),
        first_seen=('invoice_signature','first')
    ).reset_index().sort_values('num_claims', ascending=False)

    p8_hash = hstat[hstat['num_claims'] >= P8_MIN_DUP_HASH].copy()
    flagged = p8[p8['invoice_hash'].isin(p8_hash['invoice_hash'])]
    keep = [col_claim_id,'invoice_hash']
    if 'vin' in flagged.columns: keep.append('vin')
    if col_servicer in flagged.columns: keep.append(col_servicer)
    p8_claims = flagged[keep].drop_duplicates()

p8h_path = save_csv(p8_hash,   'p8_invoice_hashes.csv')
p8c_path = save_csv(p8_claims, 'p8_invoice_flagged_claims.csv')



# ================== (optional) write back simple tables ==================
try:
    with sqlite3.connect(DB_PATH) as conn:
        for t in ["p8_invoice_hashes","p8_invoice_claims"]:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        p8_hash.to_sql('p8_invoice_hashes', conn, if_exists='replace', index=False)
        p8_claims.to_sql('p8_invoice_claims', conn, if_exists='replace', index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_p8_hash ON p8_invoice_claims(invoice_hash)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_p8_claim ON p8_invoice_claims({col_claim_id})")
        print("  -> wrote tables: p8_invoice_hashes / p8_invoice_claims")
except Exception as e:
    print("(warn) write-back skipped:", e)


# ================== Intersections (P5–P8) ==================
# 依赖：claims_min（含 claim_id, vin, servicer/date/contract）、cc（claim→seller 映射，可为空）
# 以及已算好的 p5_bursts, p6_cross, p7_pairs, p8_hash, p8_claims

# def _to_set(sr):
#     return set(sr.dropna().astype(str)) if isinstance(sr, pd.Series) else set()

# ---------- 1) P5 ∩ P6：VIN 级 ----------
# p5_vins = _to_set(p5_bursts['vin']) if not p5_bursts.empty and 'vin' in p5_bursts.columns else set()
# p6_vins = _to_set(p6_cross['vin'])   if not p6_cross.empty   and 'vin' in p6_cross.columns   else set()
# vin_p5_p6 = p5_vins & p6_vins

# p5p6_events = pd.DataFrame()
# if vin_p5_p6:
#     # 聚合 VIN 的窗口与统计
#     def _agg_window(df, date_col):
#         return pd.Series({
#             'first_date': df[date_col].min(),
#             'last_date' : df[date_col].max(),
#             'num_claims': df.shape[0],
#             'distinct_servicers': df[col_servicer].nunique() if col_servicer in df.columns else np.nan,
#             'claim_ids': "|".join(df[col_claim_id].astype(str).tolist())
#         })

#     # 取交集 VIN 的所有理赔
#     base = claims_min.dropna(subset=['vin']) if 'vin' in claims_min.columns else pd.DataFrame()
#     if not base.empty:
#         base = base.assign(**{
#             col_claim_id: base[col_claim_id].astype(str),
#             'vin': base['vin'].astype(str)
#         })
#         p5p6_events = (base[base['vin'].isin(vin_p5_p6)]
#                        .groupby('vin', as_index=False)
#                        .apply(lambda g: _agg_window(g, col_claim_date) if col_claim_date in g.columns else pd.Series({'num_claims':len(g),'claim_ids':"|".join(g[col_claim_id])}))
#                        .reset_index(drop=True))
# save_csv(p5p6_events, 'p5p6_vin_intersection.csv')

# # 同时给一张 claim 级明细（便于抽样）
# p5p6_claims = pd.DataFrame()
# if vin_p5_p6 and not claims_min.empty and 'vin' in claims_min.columns:
#     p5p6_claims = (claims_min[claims_min['vin'].astype(str).isin(vin_p5_p6)]
#                    [[col_claim_id,'vin'] + ([col_claim_date] if col_claim_date in claims_min.columns else []) +
#                     ([col_servicer] if col_servicer in claims_min.columns else [])])
#     save_csv(p5p6_claims, 'p5p6_flagged_claims.csv')

# ---------- 2) P7 pairs 与 (P5 或 P6) 的交集 ----------
# p7_pairs_hits = pd.DataFrame()
# if ( 'vin' in claims_min.columns and
#      not claims_min.empty and
#      not cc.empty and
#      col_servicer and
#      'p7_pairs' in globals() and
#      not p7_pairs.empty ):

#     # 构造 claim → (servicer, seller, vin) 明细
#     tmp = cc[[col_claim_id, col_cpsi_sel]].dropna().copy()
#     tmp[col_claim_id] = as_str(tmp[col_claim_id])
#     tmp[col_cpsi_sel] = as_str(tmp[col_cpsi_sel])

#     attach = claims_min[[col_claim_id, 'vin', col_servicer]].dropna().copy()
#     attach[col_claim_id] = as_str(attach[col_claim_id])
#     attach['vin'] = as_str(attach['vin'])
#     attach[col_servicer] = as_str(attach[col_servicer])

#     tmp = tmp.merge(attach, on=col_claim_id, how='left') \
#              .dropna(subset=['vin', col_servicer])
#     tmp = tmp.rename(columns={col_cpsi_sel: 'seller_id',
#                               col_servicer: 'servicer_id'})

#     # 标记这个 claim 所在 VIN 是否命中 P5 或 P6
#     p5_or_p6_vins = p5_vins | p6_vins
#     tmp['in_p5_or_p6_vin'] = tmp['vin'].isin(p5_or_p6_vins)

#     # 只保留已进入 P7 的 pair
#     key = ['servicer_id', 'seller_id']
#     pairs = p7_pairs[key].copy()
#     pairs['servicer_id'] = as_str(pairs['servicer_id'])
#     pairs['seller_id']   = as_str(pairs['seller_id'])

#     tmp = tmp.merge(pairs, on=key, how='inner')

#     # 统计每个 pair 在 P5/P6 VIN 上的覆盖情况
#     p7_pairs_hits = (
#         tmp.groupby(key)
#            .agg(
#                distinct_vins_total=('vin', 'nunique'),
#                distinct_vins_in_p5p6=('in_p5_or_p6_vin', 'sum'),
#                claims=('vin', 'size')
#            )
#            .reset_index()
#     )

#     # 只保留真的命中 P5/P6 的 pair
#     p7_pairs_hits = p7_pairs_hits[
#         p7_pairs_hits['distinct_vins_in_p5p6'] > 0
#     ].sort_values(
#         ['distinct_vins_in_p5p6', 'distinct_vins_total', 'claims'],
#         ascending=False
#     )

# save_csv(p7_pairs_hits, 'p7_pairs_intersect_p5_or_p6.csv')
 # ---------- 3) P8 与 (P5 或 P6) 的交集（hash & claim 两张） ----------
# p8_hash_hits = pd.DataFrame()
# p8_claims_hits = pd.DataFrame()
# if not p8_claims.empty:
#     p5_or_p6_vins = p5_vins | p6_vins
#     # 先确保有 VIN；若 p8_claims 里没有 VIN，就左连 claims_min 拿 VIN
#     p8c = p8_claims.copy()
#     if 'vin' not in p8c.columns and 'vin' in claims_min.columns:
#         p8c = p8c.merge(claims_min[[col_claim_id, 'vin']].astype({col_claim_id:str}), on=col_claim_id, how='left')
#     if 'vin' in p8c.columns:
#         p8_claims_hits = p8c[p8c['vin'].astype(str).isin(p5_or_p6_vins)].drop_duplicates()
#         save_csv(p8_claims_hits, 'p8_claims_intersect_p5_or_p6.csv')

#         # hash 层汇总：命中 p5/p6 的覆盖度
#         p8_hash_hits = (p8_claims_hits.groupby('invoice_hash')
#                         .agg(num_claims=('invoice_hash','size'),
#                              distinct_vins=('vin','nunique'))
#                         .reset_index()
#                         .sort_values(['distinct_vins','num_claims'], ascending=False))
#         save_csv(p8_hash_hits, 'p8_hashes_intersect_p5_or_p6.csv')

# # ---------- 4) 三重交集：P5 ∩ P6 ∩ P8（claim 级） ----------
# p5p6p8_claims = pd.DataFrame()
# if not p8_claims.empty and vin_p5_p6:
#     p8c2 = p8_claims.copy()
#     if 'vin' not in p8c2.columns and 'vin' in claims_min.columns:
#         p8c2 = p8c2.merge(claims_min[[col_claim_id, 'vin']].astype({col_claim_id:str}), on=col_claim_id, how='left')
#     if 'vin' in p8c2.columns:
#         p5p6p8_claims = p8c2[p8c2['vin'].astype(str).isin(vin_p5_p6)].drop_duplicates()
#         save_csv(p5p6p8_claims, 'p5p6p8_claims.csv')

# print("[Intersections] Saved:",
#       "p5p6_vin_intersection.csv,",
#       "p5p6_flagged_claims.csv,",
#       "p7_pairs_intersect_p5_or_p6.csv,",
#       "p8_claims_intersect_p5_or_p6.csv,",
#       "p8_hashes_intersect_p5_or_p6.csv,",
#       "p5p6p8_claims.csv")

# ================= P3∩P4∩P5∩P6∩P7∩P8 (claim-level) =================
# 依赖：OUT_DIR, col_claim_id, claims_min（含 vin/servicer/日期更好）
# 尽量复用已在内存里的集合；若没有，则从 out/*.csv 回读做兜底

from pathlib import Path

def _read_claim_ids_csv(path, col='claim_id', reason_filter=None):
    p = Path(path)
    if not p.exists(): return set()
    try:
        df = pd.read_csv(p, dtype=str, low_memory=False)
        if reason_filter is not None and 'reason' in df.columns:
            df = df[df['reason'].isin(reason_filter)]
        if col in df.columns:
            return set(df[col].dropna().astype(str).str.strip())
    except Exception as e:
        print(f"(warn) fail reading {p.name}: {e}")
    return set()

# ---------- P3：高频 servicer/seller 的 claim 集 ----------
# 先用内存变量（若你之前已算过 p3_*_claims），否则从 out 回读
p3_claims = set()
if 'p3_servicer_claims' in globals(): p3_claims |= set(map(str, p3_servicer_claims))
if 'p3_seller_claims'   in globals(): p3_claims |= set(map(str, p3_seller_claims))
if not p3_claims:
    p3_claims |= _read_claim_ids_csv(OUT_DIR / 'p3_servicer_flagged_claims.csv', col='claim_id')
    p3_claims |= _read_claim_ids_csv(OUT_DIR / 'p3_seller_flagged_claims.csv',   col='claim_id')

# ---------- P4：重复 loss code 的 claim 集 ----------
p4_claims_set = set()
# 内存变量优先
if 'p4_claims' in globals():
    p4_claims_set |= set(map(str, p4_claims))
# 兜底：从旧口径合并文件里抽 reason = P4
if not p4_claims_set:
    p4_claims_set |= _read_claim_ids_csv(OUT_DIR / 'flagged_claims_p1_p4.csv',
                                         col=str(col_claim_id), reason_filter=['P4_repeat_loss_code'])
# 兜底2：如果有 p4_suspicious_loss_codes.csv + claim_details 可做回查（此处略）

# ---------- P5 / P6：由 VIN 回查 claim 集 ----------
def _claims_by_vins(vin_set):
    if vin_set and not claims_min.empty and 'vin' in claims_min.columns:
        vin_set = set(map(str, vin_set))
        tmp = claims_min[[col_claim_id,'vin']].dropna().copy()
        tmp[col_claim_id] = tmp[col_claim_id].astype(str)
        tmp['vin'] = tmp['vin'].astype(str)
        return set(tmp.loc[tmp['vin'].isin(vin_set), col_claim_id])
    return set()

# 若你前面已算过 p5_claims_set/p6_claims_set，就直接用；否则用 p5/p6 的 VIN 反查
p5_claims_set = globals().get('p5_claims_set', set())
p6_claims_set = globals().get('p6_claims_set', set())
if not p5_claims_set:
    p5_vins = set(pd.read_csv(OUT_DIR/'p5_vin_bursts.csv', dtype=str)['vin'].dropna().astype(str)) \
              if (OUT_DIR/'p5_vin_bursts.csv').exists() else set()
    p5_claims_set = _claims_by_vins(p5_vins)
if not p6_claims_set:
    p6_vins = set(pd.read_csv(OUT_DIR/'p6_cross_shop.csv', dtype=str)['vin'].dropna().astype(str)) \
              if (OUT_DIR/'p6_cross_shop.csv').exists() else set()
    p6_claims_set = _claims_by_vins(p6_vins)

# ---------- P7：高密度 pair 覆盖的 claim 集 ----------
p7_pair_claims_set = globals().get('p7_pair_claims_set', set())


# ---------- P8：克隆发票的 claim 集 ----------
p8_claims_set = set()
if 'p8_claims' in globals() and not p8_claims.empty and col_claim_id in p8_claims.columns:
    p8_claims_set = set(p8_claims[col_claim_id].dropna().astype(str))
elif (OUT_DIR/'p8_invoice_flagged_claims.csv').exists():
    p8_claims_set = _read_claim_ids_csv(OUT_DIR/'p8_invoice_flagged_claims.csv', col='claim_id')

# ---------- 六重交集 ----------
claims_p3to8 = sorted(list(
    (p3_claims or set()) &
    (p4_claims_set or set()) &
    (p5_claims_set or set()) &
    (p6_claims_set or set()) &
    (p7_pair_claims_set or set()) &
    (p8_claims_set or set())
))

# 保存成 CSV（附带常用维度）
p3to8_df = pd.DataFrame({col_claim_id: claims_p3to8})

if not claims_min.empty:
    keep = [col_claim_id]
    if 'vin' in claims_min.columns:
        keep.append('vin')
    if col_servicer in claims_min.columns:
        keep.append(col_servicer)
    if col_claim_date in claims_min.columns:
        keep.append(col_claim_date)

    p3to8_df = p3to8_df.merge(
        claims_min[keep].drop_duplicates(subset=[col_claim_id]),
        on=col_claim_id,
        how='left'
    )

# ===== 把 seller 标到结果表上 =====
# cc 里已经有 claim_id -> seller_id 的映射（col_cpsi_sel）
if 'cc' in globals() and not cc.empty and (col_claim_id in cc.columns):
    try:
        # 只取 claim_id 和 seller 列
        seller_map = cc[[col_claim_id, col_cpsi_sel]].dropna().copy()
        seller_map[col_claim_id] = as_str(seller_map[col_claim_id])
        seller_map[col_cpsi_sel] = as_str(seller_map[col_cpsi_sel])

        # 一个 claim 只留一个 seller（去重）
        seller_map = seller_map.drop_duplicates(subset=[col_claim_id])

        # merge 到 p3to8_df
        p3to8_df[col_claim_id] = as_str(p3to8_df[col_claim_id])
        p3to8_df = p3to8_df.merge(
            seller_map.rename(columns={col_cpsi_sel: 'seller_id'}),
            on=col_claim_id,
            how='left'
        )

        # （可选）再从 entity_sellers 表里把 seller 名字带上
        if not seller.empty:
            col_seller_id_main = pick_col(seller,
                                          ['iid', 'iId', 'seller_id', 'id'],
                                          tag='seller.id')
            col_seller_name = pick_col(seller,
                                       ['sname', 'seller_name', 'name'],
                                       required=False,
                                       tag='seller.name')
            if col_seller_name:
                s_info = seller[[col_seller_id_main, col_seller_name]].dropna().copy()
                s_info[col_seller_id_main] = as_str(s_info[col_seller_id_main])

                p3to8_df['seller_id'] = as_str(p3to8_df['seller_id'])
                p3to8_df = p3to8_df.merge(
                    s_info,
                    left_on='seller_id',
                    right_on=col_seller_id_main,
                    how='left'
                )
                # 统一列名：seller_name
                p3to8_df = p3to8_df.rename(columns={col_seller_name: 'seller_name'})
                # 把内部使用的 seller 主键列删掉，避免重复
                p3to8_df.drop(columns=[col_seller_id_main], inplace=True, errors='ignore')

    except Exception as e:
        print("(warn) attach seller to p3p4p5p6p7p8_claims failed:", e)

# 最后再保存
save_csv(p3to8_df, 'p3p4p5p6p7p8_claims.csv')
print(f"[P3–P8 six-way intersection] claims = {len(p3to8_df)}  -> p3p4p5p6p7p8_claims.csv")