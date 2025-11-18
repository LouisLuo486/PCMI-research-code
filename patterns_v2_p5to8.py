#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patterns v2 (MSSQL version; start from P5, legacy P1–P4 not used):
- P5: VIN Burst (short-window multi-claims per VIN)
- P6: Cross-shop Repairs (short-window multi-servicers per VIN)
- P7: Coordinated Clusters (servicer×seller high-density pairs)
- P8: Invoice Cloning (exact part-list signature reuse)

Data source: MSSQL Server (tables: claims, claim_details, contracts, contract_vehicle,
            entity_servicers, entity_sellers, coverage_plans_seller_inclusion, ...)

Outputs:
  - CSVs under OUT_DIR (from config.py / .env)
  - MSSQL tables for some results (e.g., p8_invoice_hashes, p8_invoice_claims,
    p3p4p5p6p7p8_claims)
"""

import os
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
import sqlalchemy as sa

from config import Config

cfg = Config()

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

# ================== Paths & MSSQL Engine ==================
BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR  = Path(cfg.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MSSQL_CONN_STR = cfg.mssql_conn_str
if not MSSQL_CONN_STR:
    raise RuntimeError("MSSQL_CONN_STR not set. 请在 .env 里配置 MSSQL_CONN_STR")

engine = sa.create_engine(MSSQL_CONN_STR, fast_executemany=True)
print(f"> MSSQL engine created for: {MSSQL_CONN_STR}")

# ================== Helpers ==================
def pick_col(df: pd.DataFrame, candidates, required=True, tag="", exclude=None):
    if df is None or df.empty:
        if required:
            raise KeyError(f"[{tag}] empty df")
        return None
    exclude = [x.lower() for x in (exclude or [])]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in cols_lower:
            orig = cols_lower[lc]
            if any(e in orig.lower() for e in exclude):
                continue
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

# ================== Load tables from MSSQL ==================
with engine.begin() as conn:
    # 打印现有表名（方便 sanity check）
    try:
        tabs = pd.read_sql(
            "SELECT TABLE_NAME AS name FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'",
            conn
        )['name'].str.lower().tolist()
    except Exception as e:
        print("(warn) cannot list tables from MSSQL:", e)
        tabs = []
    print("> MSSQL tables:", tabs)

    def read_if(tn: str) -> pd.DataFrame:
        """如果表存在就读，不存在返回空 DataFrame。"""
        if not tn:
            return pd.DataFrame()
        try:
            return pd.read_sql(f"SELECT * FROM {tn}", conn)
        except Exception as e:
            print(f"(warn) cannot read table {tn}: {e}")
            return pd.DataFrame()

    # 假设 MSSQL 中的表名如下（如有 schema，可写成 'dbo.claims' 等）
    t_claims    = 'claims'
    t_cdet      = 'claim_details'
    t_contracts = 'contracts'
    t_cveh      = 'contract_vehicle'
    t_serv      = 'entity_servicers'
    t_seller    = 'entity_sellers'
    t_cpsi      = 'coverage_plans_seller_inclusion'

    claims    = read_if(t_claims)
    cdet      = read_if(t_cdet)
    contracts = read_if(t_contracts)
    cveh      = read_if(t_cveh)
    serv      = read_if(t_serv)
    seller    = read_if(t_seller)
    cpsi      = read_if(t_cpsi)

if claims.empty:
    raise RuntimeError("claims table not found or empty in MSSQL")

# ===== pick columns =====
col_claim_id   = pick_col(claims, ['iid','iId','claim_id','id'], tag='claims.id')
col_claim_date = pick_col(
    claims,
    ['service_date','servicedate','dservicedate','claim_date','dclaimdate','created_at','dentrydate'],
    required=False,
    tag='claims.date'
)
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
    vin_map[col_v_contract] = as_str(vin_map[col_v_contract])
    vin_map[col_vin]        = as_str(vin_map[col_vin])
elif not contracts.empty:
    col_v_contract = pick_col(contracts, ['iid','iId','contract_id','id'], tag='contracts.id')
    col_vin        = pick_col(contracts, ['vin','svin','vehicle_vin'], required=False, tag='contracts.vin')
    if col_vin:
        vin_map = contracts[[col_v_contract, col_vin]].dropna().copy()
        vin_map[col_v_contract] = as_str(vin_map[col_v_contract])
        vin_map[col_vin]        = as_str(vin_map[col_vin])

# policyholder（可选）
ph_map = pd.DataFrame()
col_c_id = None
if not contracts.empty:
    col_c_id = pick_col(contracts, ['iid','iId','contract_id','id'], tag='contracts.id(reuse)')
    col_ph   = pick_col(
        contracts,
        ['policyholder_id','icustomerid','customer_id','ipolicyholderid'],
        required=False,
        tag='contracts.ph'
    )
    if col_ph:
        ph_map = contracts[[col_c_id, col_ph]].dropna().copy()
        ph_map[col_c_id] = as_str(ph_map[col_c_id])
        ph_map[col_ph]   = as_str(ph_map[col_ph])

# coverage→seller（供 P7 用）
cc = pd.DataFrame()
if (not cpsi.empty) and (col_contract_id is not None) and (not contracts.empty) and (col_c_id is not None):
    col_cov_on_contracts = pick_col(
        contracts,
        ['icoverageid','coverage_id','coverage_plan_id','iCoverageId','iCoveragePlanId'],
        required=False,
        tag='contracts.coverage'
    )
    col_cpsi_cov = pick_col(
        cpsi,
        ['coverage_id','icoverageid','coverage_plan_id','icoverageplanid','iCoverageId'],
        tag='cpsi.cov'
    )
    col_cpsi_sel = pick_col(
        cpsi,
        ['seller_id','isellerid','iSellerId'],
        tag='cpsi.seller'
    )
    if col_cov_on_contracts:
        _c = claims[[col_claim_id, col_contract_id]].dropna().copy()
        _c[col_claim_id]   = as_str(_c[col_claim_id])
        _c[col_contract_id]= as_str(_c[col_contract_id])
        _k = contracts[[col_c_id, col_cov_on_contracts]].dropna().copy()
        _k[col_c_id]           = as_str(_k[col_c_id])
        _k[col_cov_on_contracts]= as_str(_k[col_cov_on_contracts])
        cc = _c.merge(_k, left_on=col_contract_id, right_on=col_c_id, how='left')
        if not cc.empty:
            cps = cpsi[[col_cpsi_cov, col_cpsi_sel]].copy()
            cps[col_cpsi_cov] = as_str(cps[col_cpsi_cov])
            cps[col_cpsi_sel] = as_str(cps[col_cpsi_sel])
            cc = cc.merge(
                cps,
                left_on=col_cov_on_contracts,
                right_on=col_cpsi_cov,
                how='left'
            )

# attach VIN/policyholder/servicer/date to claims
claims_min = claims[[col_claim_id]].copy()
claims_min[col_claim_id] = as_str(claims_min[col_claim_id])
if col_claim_date:
    claims_min[col_claim_date] = claims[col_claim_date]
if col_contract_id:
    claims_min[col_contract_id] = as_str(claims[col_contract_id])
if col_servicer:
    claims_min[col_servicer] = as_str(claims[col_servicer])

if not vin_map.empty and col_contract_id:
    claims_min = claims_min.merge(
        vin_map,
        left_on=col_contract_id,
        right_on=col_v_contract,
        how='left'
    ).rename(columns={col_vin: 'vin'})
if not ph_map.empty and col_contract_id:
    claims_min = claims_min.merge(
        ph_map,
        left_on=col_contract_id,
        right_on=col_c_id,
        how='left'
    ).rename(columns={ph_map.columns[1]: 'policyholder_id'})

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
    p5_bursts = pd.DataFrame(rows).drop_duplicates().sort_values(
        ['num_claims','distinct_servicers'],
        ascending=False
    )

p5_path = save_csv(p5_bursts, 'p5_vin_bursts.csv')

# =========================================================
# P6: Cross-shop Repairs
p6_cross = pd.DataFrame()
if 'vin' in claims_min.columns and col_claim_date and col_servicer:
    df = claims_min.dropna(subset=['vin', col_claim_date, col_servicer]).copy()
    df['vin'] = as_str(df['vin'])
    df[col_servicer] = as_str(df[col_servicer])
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
    p6_cross = pd.DataFrame(rows).drop_duplicates().sort_values(
        ['distinct_servicers','num_claims'],
        ascending=False
    )

p6_path = save_csv(p6_cross, 'p6_cross_shop.csv')

# =========================================================
# P7: Coordinated Clusters (servicer×seller high-density pairs)
# 1) pair 级统计 p7_pairs → p7_cluster_pairs.csv
# 2) 被这些 pair 覆盖的 claim 集合 p7_pair_claims_set
# =========================================================

p7_pairs = pd.DataFrame()
p7_pair_claims_set = set()

if not cc.empty and 'vin' in claims_min.columns and col_servicer:

    # ---------- 1) 构造 claim 明细 tmp：claim_id, servicer_id, seller_id, vin, policyholder(optional) ----------
    tmp = cc[[col_claim_id, col_cpsi_sel]].dropna().copy()

    tmp[col_claim_id] = pd.to_numeric(tmp[col_claim_id], errors='coerce')
    tmp[col_cpsi_sel] = pd.to_numeric(tmp[col_cpsi_sel], errors='coerce')
    tmp = tmp.dropna(subset=[col_claim_id, col_cpsi_sel]).astype(
        {col_claim_id: 'Int64', col_cpsi_sel: 'Int64'}
    )

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

    tmp = tmp.merge(attach, on=col_claim_id, how='inner')

    # ---------- 2) pair 级聚合（得到 p7_pairs） ----------
    if 'policyholder_id' in tmp.columns:
        grp = tmp.groupby([col_servicer, col_cpsi_sel]).agg(
            distinct_vins=('vin', 'nunique'),
            distinct_policyholders=('policyholder_id', 'nunique'),
            claims=('vin', 'size')
        ).reset_index()
    else:
        grp = tmp.groupby([col_servicer, col_cpsi_sel]).agg(
            distinct_vins=('vin', 'nunique'),
            distinct_policyholders=('vin', 'size'),
            claims=('vin', 'size')
        ).reset_index()

    grp = grp.rename(columns={col_servicer: 'servicer_id', col_cpsi_sel: 'seller_id'})

    if 'policyholder_id' in tmp.columns:
        p7_pairs = grp[
            (grp['distinct_vins'] >= P7_MIN_DISTINCT_VINS) &
            (grp['distinct_policyholders'] >= P7_MIN_DISTINCT_PH)
        ].copy()
    else:
        p7_pairs = grp[grp['distinct_vins'] >= P7_MIN_DISTINCT_VINS].copy()

    p7_pairs = p7_pairs.sort_values(['distinct_vins', 'claims'], ascending=False)
    save_csv(p7_pairs, 'p7_cluster_pairs.csv')

    # ---------- 3) claim 级覆盖集合 p7_pair_claims_set ----------
    if not p7_pairs.empty:
        claim_pairs = tmp[[col_claim_id, col_servicer, col_cpsi_sel]].copy()
        claim_pairs = claim_pairs.rename(
            columns={col_servicer: 'servicer_id', col_cpsi_sel: 'seller_id'}
        )

        key = ['servicer_id', 'seller_id']
        pairs_key = p7_pairs[key].drop_duplicates()

        hit = claim_pairs.merge(pairs_key, on=key, how='inner')
        p7_pair_claims_set = set(hit[col_claim_id].astype(str))

else:
    save_csv(pd.DataFrame(), 'p7_cluster_pairs.csv')

# =========================================================
# P8: Invoice Cloning
# =========================================================
p8_hash = pd.DataFrame()
p8_claims = pd.DataFrame()

if not cdet.empty:
    col_cd_claim = pick_col(cdet, ['iclaimid','claim_id'], tag='cd.claim')
    col_part = pick_col(cdet, PART_COL_CANDS, required=False, tag='cd.part')
    col_qty  = pick_col(cdet, QTY_COL_CANDS, required=False, tag='cd.qty')
    col_price= pick_col(cdet, PRICE_COL_CANDS, required=False, tag='cd.price', exclude=['qty','quantity','hours','rate'])
    cols = [c for c in [col_part, col_qty, col_price] if c]

    df = cdet[[col_cd_claim] + cols].copy()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower()

    def norm_line(row):
        p = (row.get(col_part) or '').replace(' ', '')
        q = row.get(col_qty) or ''
        pr = row.get(col_price) or ''
        return f"{p}:{q}:{pr}"

    df['_line'] = df.apply(norm_line, axis=1)
    lines = (
        df.groupby(col_cd_claim)['_line']
          .apply(lambda s: "|".join(sorted([x for x in s if x])))
          .reset_index(name='invoice_signature')
    )
    lines['invoice_hash'] = lines['invoice_signature'].apply(md5)

    meta_cols = [col_claim_id]
    if 'vin' in claims_min.columns:
        meta_cols.append('vin')
    if col_servicer:
        meta_cols.append(col_servicer)
    meta = claims_min[meta_cols].copy()
    meta[col_claim_id] = as_str(meta[col_claim_id])

    p8 = lines.rename(columns={col_cd_claim: col_claim_id}).merge(
        meta,
        on=col_claim_id,
        how='left'
    )

    hstat = p8.groupby('invoice_hash').agg(
        num_claims=(col_claim_id,'nunique'),
        distinct_vins=('vin','nunique') if 'vin' in p8.columns else (col_claim_id,'size'),
        distinct_servicers=(col_servicer,'nunique') if col_servicer in p8.columns else (col_claim_id,'size'),
        first_seen=('invoice_signature','first')
    ).reset_index().sort_values('num_claims', ascending=False)

    p8_hash = hstat[hstat['num_claims'] >= P8_MIN_DUP_HASH].copy()
    flagged = p8[p8['invoice_hash'].isin(p8_hash['invoice_hash'])]

    keep = [col_claim_id, 'invoice_hash']
    if 'vin' in flagged.columns:
        keep.append('vin')
    if col_servicer in flagged.columns:
        keep.append(col_servicer)
    p8_claims = flagged[keep].drop_duplicates()

p8h_path = save_csv(p8_hash,   'p8_invoice_hashes.csv')
p8c_path = save_csv(p8_claims, 'p8_invoice_flagged_claims.csv')

# ================== write back P8 to MSSQL ==================
try:
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            IF OBJECT_ID('dbo.p8_invoice_hashes', 'U') IS NOT NULL
                DROP TABLE dbo.p8_invoice_hashes;
        """)
        conn.exec_driver_sql("""
            IF OBJECT_ID('dbo.p8_invoice_claims', 'U') IS NOT NULL
                DROP TABLE dbo.p8_invoice_claims;
        """)

        p8_hash.to_sql('p8_invoice_hashes', con=conn, if_exists='replace', index=False)
        p8_claims.to_sql('p8_invoice_claims', con=conn, if_exists='replace', index=False)

        conn.exec_driver_sql(
            "CREATE INDEX IX_p8_invoice_claims_hash ON dbo.p8_invoice_claims(invoice_hash);"
        )
        conn.exec_driver_sql(
            f"CREATE INDEX IX_p8_invoice_claims_id ON dbo.p8_invoice_claims({col_claim_id});"
        )

        print("  -> wrote MSSQL tables: p8_invoice_hashes / p8_invoice_claims")
except Exception as e:
    print("(warn) MSSQL write-back for P8 skipped:", e)

# ================= P3∩P4∩P5∩P6∩P7∩P8 (claim-level) =================
# 依赖：OUT_DIR, col_claim_id, claims_min（含 vin/servicer/date 更好）
# 尽量复用已在内存里的集合；若没有，则从 out/*.csv 回读兜底

def _read_claim_ids_csv(path, col='claim_id', reason_filter=None):
    """
    从一个 CSV 文件里读取 claim_id 集合。
    - col: 哪一列是 claim_id
    - reason_filter: 如果提供，就只保留 reason 在这个列表里的行
    """
    p = Path(path)
    if not p.exists():
        return set()
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
p3_claims = set()
if 'p3_servicer_claims' in globals():
    p3_claims |= set(map(str, p3_servicer_claims))
if 'p3_seller_claims' in globals():
    p3_claims |= set(map(str, p3_seller_claims))

if not p3_claims:
    p3_claims |= _read_claim_ids_csv(OUT_DIR / 'p3_servicer_flagged_claims.csv', col='claim_id')
    p3_claims |= _read_claim_ids_csv(OUT_DIR / 'p3_seller_flagged_claims.csv',   col='claim_id')

# ---------- P4：重复 loss code 的 claim 集 ----------
p4_claims_set = set()
if 'p4_claims' in globals():
    p4_claims_set |= set(map(str, p4_claims))

if not p4_claims_set:
    p4_claims_set |= _read_claim_ids_csv(
        OUT_DIR / 'flagged_claims_p1_p4.csv',
        col=str(col_claim_id),
        reason_filter=['P4_repeat_loss_code']
    )

# ---------- P5 / P6：由 VIN 反查 claim 集 ----------
def _claims_by_vins(vin_set):
    if vin_set and not claims_min.empty and 'vin' in claims_min.columns:
        vin_set = set(map(str, vin_set))
        tmp = claims_min[[col_claim_id, 'vin']].dropna().copy()
        tmp[col_claim_id] = tmp[col_claim_id].astype(str)
        tmp['vin']        = tmp['vin'].astype(str)
        return set(tmp.loc[tmp['vin'].isin(vin_set), col_claim_id])
    return set()

p5_claims_set = globals().get('p5_claims_set', set())
p6_claims_set = globals().get('p6_claims_set', set())

if not p5_claims_set:
    p5_vins = set()
    p5_csv = OUT_DIR / 'p5_vin_bursts.csv'
    if p5_csv.exists():
        df_p5 = pd.read_csv(p5_csv, dtype=str, low_memory=False)
        if 'vin' in df_p5.columns:
            p5_vins = set(df_p5['vin'].dropna().astype(str))
    p5_claims_set = _claims_by_vins(p5_vins)

if not p6_claims_set:
    p6_vins = set()
    p6_csv = OUT_DIR / 'p6_cross_shop.csv'
    if p6_csv.exists():
        df_p6 = pd.read_csv(p6_csv, dtype=str, low_memory=False)
        if 'vin' in df_p6.columns:
            p6_vins = set(df_p6['vin'].dropna().astype(str))
    p6_claims_set = _claims_by_vins(p6_vins)

# ---------- P7：高密度 pair 覆盖的 claim 集 ----------
p7_pair_claims_set = globals().get('p7_pair_claims_set', set())

# ---------- P8：克隆发票的 claim 集 ----------
p8_claims_set = set()
if 'p8_claims' in globals() and not p8_claims.empty and col_claim_id in p8_claims.columns:
    p8_claims_set = set(p8_claims[col_claim_id].dropna().astype(str))
else:
    p8_csv = OUT_DIR / 'p8_invoice_flagged_claims.csv'
    if p8_csv.exists():
        p8_claims_set = _read_claim_ids_csv(p8_csv, col='claim_id')

# ---------- 六重交集（P3∩P4∩P5∩P6∩P7∩P8） ----------
claims_p3to8 = sorted(list(
    (p3_claims or set()) &
    (p4_claims_set or set()) &
    (p5_claims_set or set()) &
    (p6_claims_set or set()) &
    (p7_pair_claims_set or set()) &
    (p8_claims_set or set())
))

print(f"[P3–P8 six-way intersection] |intersection| = {len(claims_p3to8)}")

# ---------- 生成明细表 p3p4p5p6p7p8_claims ----------
p3to8_df = pd.DataFrame({col_claim_id: claims_p3to8})

# 挂上 vin / servicer / 日期
if not claims_min.empty:
    keep = [col_claim_id]
    if 'vin' in claims_min.columns:
        keep.append('vin')
    if col_servicer in claims_min.columns:
        keep.append(col_servicer)
    if col_claim_date in claims_min.columns:
        keep.append(col_claim_date)

    base = claims_min[keep].drop_duplicates(subset=[col_claim_id]).copy()
    base[col_claim_id] = base[col_claim_id].astype(str).str.strip()
    p3to8_df[col_claim_id] = p3to8_df[col_claim_id].astype(str).str.strip()

    p3to8_df = p3to8_df.merge(base, on=col_claim_id, how='left')

# ---------- 把 seller_id / seller_name 挂上 ----------
if 'cc' in globals() and not cc.empty and (col_claim_id in cc.columns):
    try:
        seller_map = cc[[col_claim_id, col_cpsi_sel]].dropna().copy()
        seller_map[col_claim_id] = as_str(seller_map[col_claim_id])
        seller_map[col_cpsi_sel] = as_str(seller_map[col_cpsi_sel])

        seller_map = seller_map.drop_duplicates(subset=[col_claim_id])

        p3to8_df[col_claim_id] = as_str(p3to8_df[col_claim_id])
        p3to8_df = p3to8_df.merge(
            seller_map.rename(columns={col_cpsi_sel: 'seller_id'}),
            on=col_claim_id,
            how='left'
        )

        if not seller.empty:
            col_seller_id_main = pick_col(
                seller,
                ['iid', 'iId', 'seller_id', 'id'],
                tag='seller.id'
            )
            col_seller_name = pick_col(
                seller,
                ['sname', 'seller_name', 'name', 'legal_name', 'display_name'],
                required=False,
                tag='seller.name'
            )
            if col_seller_id_main:
                s_info_cols = [col_seller_id_main]
                if col_seller_name:
                    s_info_cols.append(col_seller_name)

                s_info = seller[s_info_cols].dropna().copy()
                s_info[col_seller_id_main] = as_str(s_info[col_seller_id_main])

                p3to8_df['seller_id'] = as_str(p3to8_df['seller_id'])
                p3to8_df = p3to8_df.merge(
                    s_info,
                    left_on='seller_id',
                    right_on=col_seller_id_main,
                    how='left'
                )

                if col_seller_name:
                    p3to8_df = p3to8_df.rename(columns={col_seller_name: 'seller_name'})

                p3to8_df.drop(columns=[col_seller_id_main], inplace=True, errors='ignore')

    except Exception as e:
        print("(warn) attach seller to p3p4p5p6p7p8_claims failed:", e)

# ---------- 保存到 CSV ----------
csv_path = save_csv(p3to8_df, 'p3p4p5p6p7p8_claims.csv')

# ---------- 同步写回 MSSQL ----------
try:
    with engine.begin() as conn:
        conn.exec_driver_sql("""
            IF OBJECT_ID('dbo.p3p4p5p6p7p8_claims', 'U') IS NOT NULL
                DROP TABLE dbo.p3p4p5p6p7p8_claims;
        """)
        p3to8_df.to_sql('p3p4p5p6p7p8_claims', con=conn, if_exists='replace', index=False)

        conn.exec_driver_sql(
            f"CREATE INDEX IX_p3p4p5p6p7p8_claims_{col_claim_id} "
            f"ON dbo.p3p4p5p6p7p8_claims({col_claim_id});"
        )
        print("  -> wrote MSSQL table: p3p4p5p6p7p8_claims")
except Exception as e:
    print("(warn) MSSQL write-back for p3p4p5p6p7p8_claims failed:", e)

print(f"[P3–P8 six-way intersection] final rows = {len(p3to8_df)}  -> {csv_path.name}")
