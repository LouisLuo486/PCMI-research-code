#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fraud Pattern Mining (P3–P4 on SQLite) — tightened thresholds
- P3: High-frequency Seller/Servicer (tighter + diversity/min-count filters)
- P4: Repeat loss codes (tighter + min-count + diversity by servicers + secondary tightening)
- P3 ∩ P4 overlap
- Summary & CSVs
"""

import os, sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# ============== Tunable thresholds (TIGHTENED) ==============
P3_SELLER_Q         = 0.9999      # top 0.5% sellers by weighted_claims
P3_SELLER_MIN_W     = 100.0       # absolute min weighted claims
P3_SELLER_MIN_COV   = 8           # min distinct coverage plans

P3_SERVICER_Q       = 0.9995      # top 0.5% servicers by claim count
P3_SERVICER_MIN_N   = 5           # absolute min claim count

P4_LOSS_Q           = 0.998       # top loss codes by claim count
P4_LOSS_MIN_N       = 500         # absolute min claim count
P4_LOSS_MIN_SVC_DV  = 5           # min distinct servicers involved

# —— P4 收紧开关（占比 + TopK）——
P4_MAX_PROP   = 0.02   # 单码命中理赔数 / 全库 ≤ 2%
P4_TOPK       = 50     # 仅保留命中量/占比最高的前 K 个码
P4_FILTER_MODE = "intersection"  # 选项: "intersection" | "union" | "prop_only" | "topk_only"

# ============== Paths ==============
BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH  = BASE_DIR / 'out' / 'data.sqlite'
OUT_DIR  = DB_PATH.parent

print(f"> CWD        : {os.getcwd()}")
print(f"> DB_PATH    : {DB_PATH}")
print(f"> DB exists? : {DB_PATH.exists()}")

# ============== Helpers ==============
def table_exists(conn, name):
    q = "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)"
    return pd.read_sql(q, conn, params=[name]).shape[0] > 0

def pick_col(df, candidates, required=True, tag="", exclude_substrings=None):
    if df.empty:
        if required: raise KeyError(f"[{tag}] DataFrame empty")
        return None
    exclude_substrings = [s.lower() for s in (exclude_substrings or [])]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in cols_lower:
            orig = cols_lower[lc]
            if any(ex in orig.lower() for ex in exclude_substrings): 
                continue
            print(f"  - [{tag}] using column: {orig}")
            return orig
    for cand in candidates:
        lcand = cand.lower()
        for lc, orig in cols_lower.items():
            if lcand in lc and not any(ex in lc for ex in exclude_substrings):
                print(f"  - [{tag}] using (substring) column: {orig}")
                return orig
    if required:
        print(f"!!! [{tag}] candidates not found. Available columns:\n{list(df.columns)}")
        raise KeyError(f"[{tag}] None of {candidates} found.")
    return None

def parse_dates_inplace(df, cols):
    for c in cols:
        if c and c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

def to_number(series):
    if series is None:
        return pd.Series(dtype='float64')
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce')

def as_str(sr):
    return sr.astype(str).str.strip()

def save_csv(df, name):
    out = OUT_DIR / name
    df.to_csv(out, index=False)
    print(f"  -> saved: {out}")
    return out

def _p4_tighten_by_prop_and_topk(p4_df, claims_df, col_claim_id, key_col,
                                 mode="intersection", max_prop=0.02, topk=50):
    """
    在现有 p4_df（含：key_col=loss_code_id，'claim_count'）基础上做二次收紧：
    1) 计算 claim_prop = claim_count / 全库去重理赔数
    2) 按模式过滤（交集/并集/只占比/只TopK）
    返回 (过滤后的 DataFrame, 调试信息字典)
    """
    if p4_df.empty:
        return p4_df, {"total_claims": 0, "before_codes": 0, "after_codes": 0, "mode": mode}

    total_claims = claims_df[col_claim_id].astype(str).str.strip().nunique()
    df = p4_df.copy()
    if 'claim_prop' not in df.columns:
        df['claim_prop'] = df['claim_count'] / max(total_claims, 1)

    # 统一排序口径（先占比、再命中数）
    df = df.sort_values(['claim_prop', 'claim_count'], ascending=[False, False])

    keep_by_prop = df[df['claim_prop'] <= max_prop]
    keep_by_topk = df.head(topk)

    if mode == "intersection":
        kept_keys = set(keep_by_prop[key_col]).intersection(set(keep_by_topk[key_col]))
        out = df[df[key_col].isin(kept_keys)]
    elif mode == "union":
        kept_keys = pd.concat([keep_by_prop[key_col], keep_by_topk[key_col]]).drop_duplicates()
        out = df[df[key_col].isin(kept_keys)]
    elif mode == "prop_only":
        out = keep_by_prop
    elif mode == "topk_only":
        out = keep_by_topk
    else:
        out = df  # 未知模式则不变

    info = {
        "total_claims": total_claims,
        "before_codes": int(len(df)),
        "after_codes":  int(len(out)),
        "mode": mode, "max_prop": max_prop, "topk": topk
    }
    # 维持与上游一致的排序
    out = out.sort_values(['claim_prop', 'claim_count'], ascending=[False, False])
    return out, info

# ============== Load tables from SQLite ==============
with sqlite3.connect(DB_PATH) as conn:
    names = {
        'claims': None, 'claim_details': None, 'contracts': None,
        'coverage_plans': None, 'product_types': None,
        'entity_servicers': None, 'entity_sellers': None, 'product_loss_codes': None,
        'coverage_plans_seller_inclusion': None
    }
    existing = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )['name'].str.lower().tolist()
    print("> tables:", existing)

    preferred_exact = {
        'claims':'claims','claim_details':'claim_details','contracts':'contracts',
        'coverage_plans':'coverage_plans','product_types':'product_types',
        'entity_servicers':'entity_servicers','entity_sellers':'entity_sellers',
        'product_loss_codes':'product_loss_codes',
        'coverage_plans_seller_inclusion':'coverage_plans_seller_inclusion'
    }
    existing_set = set(existing)
    for key in list(names.keys()):
        cand = preferred_exact.get(key)
        if cand and cand in existing_set:
            names[key] = cand
            continue
        for tbl in existing:
            if key == 'contracts' and 'contract_vehicle' in tbl:
                continue
            if key.rstrip('s') in tbl or key in tbl:
                names[key] = tbl
                break

    tbl = {}
    for k, v in names.items():
        tbl[k] = pd.read_sql(f"SELECT * FROM {v}", conn) if (v and table_exists(conn, v)) else pd.DataFrame()

claims, claim_details = tbl['claims'], tbl['claim_details']
contracts, coverage   = tbl['contracts'], tbl['coverage_plans']
product_types         = tbl['product_types']
servicers, sellers    = tbl['entity_servicers'], tbl['entity_sellers']
loss_codes            = tbl['product_loss_codes']
cpsi                  = tbl['coverage_plans_seller_inclusion']

# ============== Column selection ==============
if claims.empty:
    raise RuntimeError("Claims table not found or empty.")

col_claim_id        = pick_col(claims, ['iId','iid','claim_id','id'], tag='claims.id')
col_claim_contract  = pick_col(claims, ['iContractId','icontractid','contract_id'], required=False, tag='claims.contract')
col_claim_servicer  = pick_col(claims, ['iServicerId','iservicerid','servicer_id'], required=False, tag='claims.servicer')
col_claim_coverage  = pick_col(claims, ['iCoverageId','icoverageid','coverage_id'], required=False, tag='claims.coverage')

col_claim_date      = pick_col(
    claims,
    ['dservicedate','dclaimdate','dentrydate','claim_date','created_at','service_date','servicedate','claimdate'],
    required=False, tag='claims.date'
)
col_total_paid      = pick_col(
    claims,
    ['total_paid','nTotalPaid','paid_amount','namountcovered','namountauthorized','amount'],
    required=False, tag='claims.total'
)

if not contracts.empty:
    col_contract_id       = pick_col(contracts, ['iId','iid','contract_id','id'], tag='contracts.id')
    col_contract_coverage = pick_col(
        contracts, ['iCoverageId','icoverageid','coverage_id','coverage_plan_id','iCoveragePlanId'],
        required=False, tag='contracts.coverage'
    )
    col_contract_start    = pick_col(
        contracts,
        ['dttwarrantyinservice','dtwarrantyinservice','contract_start_date',
         'dstartdate','start_date','effective_date','contractstartdate'],
        required=False, tag='contracts.start'
    )
else:
    col_contract_id = col_contract_coverage = col_contract_start = None

if not coverage.empty:
    col_cov_id       = pick_col(coverage, ['iId','id','coverage_id'], tag='coverage.id')
    col_cov_prodtype = pick_col(coverage, ['iProductTypeId','product_type_id','iproducttypeid'],
                                required=False, tag='coverage.prodtype')
else:
    col_cov_id = col_cov_prodtype = None

if not product_types.empty:
    col_pt_id   = pick_col(product_types, ['iId','id','product_type_id'], tag='ptype.id')
    col_pt_name = pick_col(
        product_types,
        ['sproductdesc','sproductcode','stextcode','product_type_name','name','type_name'],
        required=False, tag='ptype.name'
    )
else:
    col_pt_id = col_pt_name = None

if not claim_details.empty:
    col_cd_claim = pick_col(claim_details, ['iClaimId','claim_id'], tag='cd.claim')
    col_cd_part  = pick_col(
        claim_details,
        ['part_cost','parts_cost','part_amount','parts_amount','npartamount','npartsamount','npartcost','npartscost',
         'nparttotal','npartstotal','dparttotal','partstotal'],
        required=False, tag='cd.part',
        exclude_substrings=['qty','quantity','count','num','rate','hours','hrs','desc','code']
    )
    col_cd_labor = pick_col(
        claim_details,
        ['labor_cost','labor_amount','nlabortotal','dlabortotal','nlaboramount','nlabouramount','nlabourtotal','labortotal'],
        required=False, tag='cd.labor',
        exclude_substrings=['qty','quantity','hours','hrs','rate','desc','code']
    )
    col_cd_loss  = pick_col(claim_details, ['iLossCodeId','loss_code_id'], required=False, tag='cd.loss')
else:
    col_cd_claim = col_cd_part = col_cd_labor = col_cd_loss = None

if not servicers.empty:
    col_sv_id = pick_col(servicers, ['iId','id','servicer_id'], tag='servicer.id')
    name_sv   = next((c for c in ['servicer_name','name','legal_name','display_name'] if c in servicers.columns), None)
else:
    col_sv_id = name_sv = None

if not sellers.empty:
    col_se_id = pick_col(sellers, ['iId','id','seller_id'], tag='seller.id')
    name_se   = next((c for c in ['seller_name','name','legal_name','display_name'] if c in sellers.columns), None)
else:
    col_se_id = name_se = None

if not loss_codes.empty and col_cd_loss:
    col_lc_id   = pick_col(loss_codes, ['iId','id','loss_code_id'], tag='loss.id')
    col_lc_desc = next((c for c in ['loss_code_description','description','desc','name'] if c in loss_codes.columns), None)
else:
    col_lc_id = col_lc_desc = None

if not cpsi.empty:
    col_cpsi_cov = next(
        (c for c in ['coverage_id','icoverageid','coverage_plan_id','icoverageplanid','iCoverageId'] if c in cpsi.columns),
        None
    )
    col_cpsi_sel = next(
        (c for c in ['seller_id','isellerid','iseller','iSellerId'] if c in cpsi.columns),
        None
    )
else:
    col_cpsi_cov = col_cpsi_sel = None

# ============== Dates ==============
if col_claim_date:
    parse_dates_inplace(claims, [col_claim_date])
if col_contract_start:
    parse_dates_inplace(contracts, [col_contract_start])

# ============== P1 (disabled) ==============
p1_out = pd.DataFrame()

# ============== P2 (disabled) ==============
p2_out = pd.DataFrame()

# ============== P3 Seller via CPSI (tight) ==============
p3_seller = pd.DataFrame()
cc = pd.DataFrame()
can_use_cpsi = (not cpsi.empty) and (col_cpsi_cov is not None) and (col_cpsi_sel is not None)

print("\n[P3 seller via CPSI] sanity:")
print(f"  - can_use_cpsi           : {can_use_cpsi}")
print(f"  - col_claim_contract     : {col_claim_contract}")
print(f"  - col_contract_id        : {col_contract_id}")
print(f"  - col_contract_coverage  : {col_contract_coverage}")
print(f"  - col_claim_coverage     : {col_claim_coverage}")

if can_use_cpsi:
    pathA_ready = (col_claim_contract and col_contract_id and col_contract_coverage and not contracts.empty)
    pathB_ready = (col_claim_coverage is not None)

    if pathA_ready:
        print("  - using PATH A (claims.contract_id -> contracts.coverage_id)")
        _Ac = claims[[col_claim_id, col_claim_contract]].copy()
        _Ac[col_claim_id]       = as_str(_Ac[col_claim_id])
        _Ac[col_claim_contract] = as_str(_Ac[col_claim_contract])
        _Ak = contracts[[col_contract_id, col_contract_coverage]].copy()
        _Ak[col_contract_id]       = as_str(_Ak[col_contract_id])
        _Ak[col_contract_coverage] = as_str(_Ak[col_contract_coverage])
        cc = _Ac.merge(
            _Ak, left_on=col_claim_contract, right_on=col_contract_id, how='left'
        )[[col_claim_id, col_contract_coverage]].dropna()
        cov_on_cc = col_contract_coverage
    elif pathB_ready:
        print("  - using PATH B (claims.coverage_id direct)")
        cc = claims[[col_claim_id, col_claim_coverage]].dropna().copy()
        cc[col_claim_id]       = as_str(cc[col_claim_id])
        cc[col_claim_coverage] = as_str(cc[col_claim_coverage])
        cov_on_cc = col_claim_coverage
    else:
        print("  - neither PATH A nor PATH B ready.")
        cov_on_cc = None

    if not cc.empty and cov_on_cc:
        _CPSI = cpsi[[col_cpsi_cov, col_cpsi_sel]].copy()
        _CPSI[col_cpsi_cov] = as_str(_CPSI[col_cpsi_cov])
        _CPSI[col_cpsi_sel] = as_str(_CPSI[col_cpsi_sel])
        cc[cov_on_cc]       = as_str(cc[cov_on_cc])
        cc = cc.merge(_CPSI, left_on=cov_on_cc, right_on=col_cpsi_cov, how='left') \
               .dropna(subset=[col_cpsi_sel])

        # claim × seller（加 coverage_id 保留做多样性）
        grp_sizes = cc.groupby(col_claim_id)[col_cpsi_sel].transform('nunique').astype(float)
        cc['weight'] = 1.0 / grp_sizes

        # seller 级别：加权计数 + 覆盖多样性
        seller_agg = (cc.groupby(col_cpsi_sel)
                        .agg(weighted_claims=('weight','sum'),
                             cov_diversity=(cov_on_cc,'nunique'))
                        .reset_index()
                        .rename(columns={col_cpsi_sel:'seller_id'}))
        if not seller_agg.empty:
            n = len(seller_agg)
            thr = float(seller_agg['weighted_claims'].quantile(P3_SELLER_Q)) if n >= 10 else P3_SELLER_MIN_W
            p3_seller = seller_agg[
                (seller_agg['weighted_claims'] >= max(thr, P3_SELLER_MIN_W)) &
                (seller_agg['cov_diversity']    >= P3_SELLER_MIN_COV)
            ].copy().sort_values('weighted_claims', ascending=False)

            if not sellers.empty:
                id_col = next((c for c in ['iId','id','seller_id'] if c in sellers.columns), None)
                name_col = name_se or next((c for c in ['seller_name','name','legal_name','display_name'] if c in sellers.columns), None)
                if id_col and name_col:
                    tmp_se = sellers[[id_col, name_col]].copy()
                    tmp_se[id_col]     = as_str(tmp_se[id_col])
                    p3_seller['seller_id'] = as_str(p3_seller['seller_id'])
                    p3_seller = p3_seller.merge(tmp_se, left_on='seller_id', right_on=id_col, how='left')

            print(
                f"  - sellers_total={n}  thr_q={P3_SELLER_Q} thr_val≈{thr:.2f}  "
                f"min_w={P3_SELLER_MIN_W} min_cov={P3_SELLER_MIN_COV}  selected={len(p3_seller)}"
            )
        else:
            print("  - seller_agg empty after join to CPSI.")

# ============== P3 Servicer (tight) ==============
p3_servicer = pd.DataFrame()
if col_claim_servicer:
    servicer_freq = claims.groupby(col_claim_servicer).size().reset_index(name='num_claims')
    if not servicer_freq.empty:
        n = len(servicer_freq)
        thr_serv = float(servicer_freq['num_claims'].quantile(P3_SERVICER_Q)) if n >= 10 else P3_SERVICER_MIN_N
        p3_servicer = servicer_freq[
            servicer_freq['num_claims'] >= max(thr_serv, P3_SERVICER_MIN_N)
        ].copy().sort_values('num_claims', ascending=False)

        if not servicers.empty and col_sv_id and name_sv:
            p3_servicer = p3_servicer.merge(
                servicers[[col_sv_id, name_sv]],
                left_on=col_claim_servicer, right_on=col_sv_id, how='left'
            )

        print(
            f"[P3 servicer] servicers_total={n}  thr_q={P3_SERVICER_Q} thr_val≈{thr_serv:.0f}  "
            f"min_n={P3_SERVICER_MIN_N}  selected={len(p3_servicer)}"
        )
    else:
        print("[P3 servicer] freq is empty.")
else:
    print("[P3 servicer] skipped: no servicer column on claims.")

# ============== P4 Loss Codes (tight + secondary tighten) ==============
p4_out = pd.DataFrame()
if not claim_details.empty and col_cd_loss and col_cd_claim:
    lc_claims = claim_details[[col_cd_loss, col_cd_claim]].drop_duplicates()
    lc_claims[col_cd_loss]  = as_str(lc_claims[col_cd_loss])
    lc_claims[col_cd_claim] = as_str(lc_claims[col_cd_claim])

    # 附带 servicer 以计算多样性
    svc_col = col_claim_servicer if col_claim_servicer in claims.columns else None
    if svc_col:
        svc_map = claims[[col_claim_id, svc_col]].copy()
        svc_map[col_claim_id] = as_str(svc_map[col_claim_id])
        svc_map[svc_col]      = as_str(svc_map[svc_col])
        lc_claims = lc_claims.merge(
            svc_map, left_on=col_cd_claim, right_on=col_claim_id, how='left'
        )

    lc_freq = lc_claims.groupby(col_cd_loss)[col_cd_claim].nunique().reset_index(name='claim_count')
    if svc_col:
        svc_div = lc_claims.groupby(col_cd_loss)[svc_col].nunique().reset_index(name='servicer_diversity')
        lc_freq = lc_freq.merge(svc_div, on=col_cd_loss, how='left')
    else:
        lc_freq['servicer_diversity'] = 0

    if not lc_freq.empty:
        n = len(lc_freq)
        thr_lc = float(lc_freq['claim_count'].quantile(P4_LOSS_Q)) if n >= 10 else P4_LOSS_MIN_N
        p4_out = lc_freq[
            (lc_freq['claim_count']        >= max(thr_lc, P4_LOSS_MIN_N)) &
            (lc_freq['servicer_diversity'] >= P4_LOSS_MIN_SVC_DV)
        ].copy().sort_values('claim_count', ascending=False)

        if not loss_codes.empty and col_lc_id:
            desc_cols = [col_lc_id]
            if col_lc_desc:
                desc_cols.append(col_lc_desc)
            loss_map = loss_codes[desc_cols].copy()
            loss_map[col_lc_id] = as_str(loss_map[col_lc_id])
            p4_out = p4_out.merge(loss_map, left_on=col_cd_loss, right_on=col_lc_id, how='left')

        print(
            f"[P4 loss code] codes_total={n}  thr_q={P4_LOSS_Q} thr_val≈{thr_lc:.0f}  "
            f"min_n={P4_LOSS_MIN_N} min_svc_div={P4_LOSS_MIN_SVC_DV}  selected={len(p4_out)}"
        )

        # 二次收紧：按照 claim 占比 + TopK
        if not p4_out.empty:
            KEY_COL = col_cd_loss
            p4_out, p4_info = _p4_tighten_by_prop_and_topk(
                p4_out, claims, col_claim_id, KEY_COL,
                mode=P4_FILTER_MODE, max_prop=P4_MAX_PROP, topk=P4_TOPK
            )
            print(
                f"[P4 tighten] mode={p4_info['mode']} total_claims={p4_info['total_claims']} "
                f"codes_before={p4_info['before_codes']} codes_after={p4_info['after_codes']} "
                f"max_prop={p4_info['max_prop']} topk={p4_info['topk']}"
            )
else:
    p4_out = pd.DataFrame(columns=[col_cd_loss, 'claim_count'])

# ============== Expand P3 to claim level ==============
CLAIM_COL_P3 = 'claim_id'

# Servicer -> claims
p3_servicer_claims = set()
p3_servicer_claims_df = pd.DataFrame()
if col_claim_servicer and not p3_servicer.empty:
    top_servicers = set(p3_servicer.iloc[:, 0].astype(str))
    tmp = claims[[col_claim_id, col_claim_servicer]].dropna().copy()
    tmp[col_claim_id]       = as_str(tmp[col_claim_id])
    tmp[col_claim_servicer] = as_str(tmp[col_claim_servicer])
    tmp.rename(columns={col_claim_id: CLAIM_COL_P3}, inplace=True)
    p3_servicer_claims_df = tmp.loc[tmp[col_claim_servicer].isin(top_servicers)].drop_duplicates()
    if not servicers.empty and col_sv_id:
        name_col = name_sv or next(
            (c for c in ['servicer_name','name','legal_name','display_name'] if c in servicers.columns),
            None
        )
        cols = [col_sv_id] + ([name_col] if name_col else [])
        sv = servicers[cols].copy()
        sv[col_sv_id] = as_str(sv[col_sv_id])
        p3_servicer_claims_df = p3_servicer_claims_df.merge(
            sv, left_on=col_claim_servicer, right_on=col_sv_id, how='left'
        )
    p3_servicer_claims = set(p3_servicer_claims_df[CLAIM_COL_P3].astype(str))

# Seller -> claims
p3_seller_claims = set()
p3_seller_claims_df = pd.DataFrame()
if can_use_cpsi and not p3_seller.empty and not cc.empty:
    top_sellers = set(p3_seller['seller_id'].astype(str))
    cc_exp = cc.copy()
    cc_exp[col_claim_id] = as_str(cc_exp[col_claim_id])
    cc_exp[col_cpsi_sel] = as_str(cc_exp[col_cpsi_sel])
    cc_hit = cc_exp[cc_exp[col_cpsi_sel].isin(top_sellers)].copy()
    cc_hit = cc_hit.rename(columns={col_cpsi_sel: 'seller_id', col_claim_id: 'claim_id'})
    p3_seller_claims = set(cc_hit['claim_id'])
    if not sellers.empty:
        id_col = next((c for c in ['iId','id','seller_id'] if c in sellers.columns), None)
        name_col = name_se or next(
            (c for c in ['seller_name','name','legal_name','display_name'] if c in sellers.columns),
            None
        )
        if id_col and name_col:
            tmp_se = sellers[[id_col, name_col]].copy()
            tmp_se[id_col] = as_str(tmp_se[id_col])
            cc_hit = cc_hit.merge(tmp_se, left_on='seller_id', right_on=id_col, how='left')
    keep_cols = ['claim_id', 'seller_id', 'weight']
    if 'name_col' in locals() and name_col in cc_hit.columns:
        keep_cols.append(name_col)
    p3_seller_claims_df = cc_hit[keep_cols].drop_duplicates()

# ============== P3 ∩ P4 overlap ==============
p3_claims_union = (p3_servicer_claims | p3_seller_claims)

p4_claims = set()
if not claim_details.empty and not p4_out.empty and col_cd_loss and col_cd_claim:
    top_lc = set(p4_out.iloc[:, 0].astype(str))
    _tmp = claim_details[[col_cd_claim, col_cd_loss]].drop_duplicates().copy()
    _tmp[col_cd_claim] = as_str(_tmp[col_cd_claim])
    _tmp[col_cd_loss]  = as_str(_tmp[col_cd_loss])
    p4_claims = set(_tmp.loc[_tmp[col_cd_loss].isin(top_lc), col_cd_claim])

p3_p4_overlap = p3_claims_union & p4_claims

overlap_df = pd.DataFrame({'claim_id': list(p3_p4_overlap)}).sort_values('claim_id')

if not p3_servicer_claims_df.empty:
    svc_cols_keep = ['claim_id', col_claim_servicer]
    if col_sv_id in p3_servicer_claims_df.columns:
        svc_cols_keep.append(col_sv_id)
    if name_sv and name_sv in p3_servicer_claims_df.columns:
        svc_cols_keep.append(name_sv)
    svc_side = p3_servicer_claims_df[svc_cols_keep].drop_duplicates()
    overlap_df = overlap_df.merge(svc_side, on='claim_id', how='left')
    overlap_df['hit_p3_servicer'] = overlap_df[col_claim_servicer].notna()

if not p3_seller_claims_df.empty:
    sel_cols_keep = ['claim_id', 'seller_id', 'weight']
    if col_se_id and col_se_id in p3_seller_claims_df.columns:
        sel_cols_keep.append(col_se_id)
    if name_se and name_se in p3_seller_claims_df.columns:
        sel_cols_keep.append(name_se)
    sel_side = p3_seller_claims_df[sel_cols_keep].drop_duplicates()
    overlap_df = overlap_df.merge(sel_side, on='claim_id', how='left')
    overlap_df['hit_p3_seller'] = overlap_df['seller_id'].notna()

# 附上 P4 loss code 描述
if not claim_details.empty and not p4_out.empty and col_cd_loss and col_cd_claim:
    top_lc = set(p4_out.iloc[:, 0].astype(str))
    lc_side = claim_details[[col_cd_claim, col_cd_loss]].drop_duplicates().copy()
    lc_side[col_cd_claim] = as_str(lc_side[col_cd_claim])
    lc_side[col_cd_loss]  = as_str(lc_side[col_cd_loss])
    lc_side = lc_side[
        lc_side[col_cd_claim].isin(p3_p4_overlap) & lc_side[col_cd_loss].isin(top_lc)
    ]
    if not loss_codes.empty and col_lc_id:
        desc_cols = [col_lc_id]
        if col_lc_desc:
            desc_cols.append(col_lc_desc)
        loss_map = loss_codes[desc_cols].copy()
        loss_map[col_lc_id] = as_str(loss_map[col_lc_id])
        lc_side = lc_side.merge(loss_map, left_on=col_cd_loss, right_on=col_lc_id, how='left')
    lc_side = lc_side.rename(columns={col_cd_claim: 'claim_id'})
    overlap_df = overlap_df.merge(lc_side, on='claim_id', how='left')

# ============== Save main outputs (no P1/P2) ==============
print("\n> Saving outputs...")
save_csv(p3_seller, 'p3_seller_suspicious.csv')
save_csv(p3_servicer, 'p3_servicer_suspicious.csv')
save_csv(p4_out, 'p4_suspicious_loss_codes.csv')
save_csv(p3_servicer_claims_df.rename(columns={CLAIM_COL_P3: 'claim_id'}), 'p3_servicer_flagged_claims.csv')
save_csv(p3_seller_claims_df, 'p3_seller_flagged_claims.csv')
save_csv(overlap_df, 'p3_p4_overlap_claims.csv')

# ============== Combined flags (only P3/P4) ==============
flags_old = []
# P1/P2 已关闭，仅保留 P4 的 repeat loss code
if not claim_details.empty and not p4_out.empty and col_cd_loss and col_cd_claim:
    top_lc = set(p4_out.iloc[:, 0].astype(str))
    tmp = claim_details[[col_cd_claim, col_cd_loss]].drop_duplicates().copy()
    tmp[col_cd_claim] = as_str(tmp[col_cd_claim])
    tmp[col_cd_loss]  = as_str(tmp[col_cd_loss])
    f = tmp[tmp[col_cd_loss].isin(top_lc)][[col_cd_claim]].rename(columns={col_cd_claim: col_claim_id})
    f[col_claim_id] = as_str(f[col_claim_id])
    f['reason'] = 'P4_repeat_loss_code'
    flags_old.append(f)

combined_old = pd.concat(flags_old, ignore_index=True) if flags_old else pd.DataFrame(columns=[col_claim_id, 'reason'])
combined_old[col_claim_id] = combined_old[col_claim_id].astype(str)
save_csv(combined_old.drop_duplicates(), 'flagged_claims_p1_p4.csv')  # 文件名沿用，内容仅 P4

flags_all = []
if not combined_old.empty:
    flags_all.append(combined_old[[col_claim_id, 'reason']])
if len(p3_servicer_claims) > 0:
    flags_all.append(pd.DataFrame({col_claim_id: list(p3_servicer_claims), 'reason': 'P3_high_freq_servicer'}))
if len(p3_seller_claims) > 0:
    flags_all.append(pd.DataFrame({col_claim_id: list(p3_seller_claims), 'reason': 'P3_high_freq_seller'}))

combined_all = pd.concat(flags_all, ignore_index=True) if flags_all else pd.DataFrame(columns=[col_claim_id, 'reason'])
combined_all[col_claim_id] = combined_all[col_claim_id].astype(str)
save_csv(combined_all.drop_duplicates(), 'flagged_claims_all.csv')

# === P3-seller ∩ P3-servicer ∩ P4 (claim-level triple overlap) =================

def _agg_uniq(series):
    vals = [str(x) for x in series.dropna().astype(str).unique().tolist()]
    return "|".join(sorted(vals))

# 1) 三者交集（claim_id 集合）
triple_overlap_claims = sorted(list(
    (p3_seller_claims or set()) &
    (p3_servicer_claims or set()) &
    (p4_claims or set())
))
triple_df = pd.DataFrame({'claim_id': triple_overlap_claims})

# 2) 取 P3-seller 侧的信息（多卖家 → 聚合）
seller_side = pd.DataFrame()
if 'p3_seller_claims_df' in globals() and not p3_seller_claims_df.empty:
    seller_side = p3_seller_claims_df.copy()
    _seller_name_col = next(
        (c for c in ['seller_name','name','legal_name','display_name'] if c in seller_side.columns),
        None
    )
    keep = ['claim_id', 'seller_id'] + ([_seller_name_col] if _seller_name_col else []) + (['weight'] if 'weight' in seller_side.columns else [])
    seller_side = seller_side[keep].drop_duplicates()

    agg_dict = {'seller_id': _agg_uniq}
    if _seller_name_col:
        agg_dict[_seller_name_col] = _agg_uniq
    if 'weight' in seller_side.columns:
        agg_dict['weight'] = 'sum'
    seller_side = seller_side.groupby('claim_id', as_index=False).agg(agg_dict)
    if 'weight' in seller_side.columns:
        seller_side.rename(columns={'weight': 'seller_weight_sum'}, inplace=True)
        seller_side['seller_count'] = seller_side['seller_id'].str.count(r'\|').fillna(0).astype(int) + 1

# 3) 取 P3-servicer 侧的信息（多服务商 → 聚合）
servicer_side = pd.DataFrame()
if 'p3_servicer_claims_df' in globals() and not p3_servicer_claims_df.empty:
    servicer_side = p3_servicer_claims_df.copy()
    _sv_name_col = name_sv if ('name_sv' in globals() and name_sv and name_sv in servicer_side.columns) else \
                   next((c for c in ['servicer_name','name','legal_name','display_name'] if c in servicer_side.columns), None)
    keep = ['claim_id', col_claim_servicer] + ([_sv_name_col] if _sv_name_col else [])
    keep = [k for k in keep if k in servicer_side.columns]
    servicer_side = servicer_side[keep].drop_duplicates()

    rename_map = {}
    if col_claim_servicer in servicer_side.columns:
        rename_map[col_claim_servicer] = 'servicer_id'
    if _sv_name_col:
        rename_map[_sv_name_col] = 'servicer_name'
    if rename_map:
        servicer_side = servicer_side.rename(columns=rename_map)

    agg_dict_sv = {'servicer_id': _agg_uniq}
    if 'servicer_name' in servicer_side.columns:
        agg_dict_sv['servicer_name'] = _agg_uniq
    servicer_side = servicer_side.groupby('claim_id', as_index=False).agg(agg_dict_sv)
    servicer_side['servicer_count'] = servicer_side['servicer_id'].str.count(r'\|').fillna(0).astype(int) + 1

# 4) 取 P4 侧的 loss codes（只保留 top loss codes；多码 → 聚合）
loss_side = pd.DataFrame()
if not claim_details.empty and not p4_out.empty and col_cd_loss and col_cd_claim:
    top_lc = set(p4_out.iloc[:, 0].astype(str))
    loss_side = claim_details[[col_cd_claim, col_cd_loss]].drop_duplicates().copy()
    loss_side[col_cd_claim] = as_str(loss_side[col_cd_claim])
    loss_side[col_cd_loss]  = as_str(loss_side[col_cd_loss])
    loss_side = loss_side[
        loss_side[col_cd_claim].isin(triple_overlap_claims) & loss_side[col_cd_loss].isin(top_lc)
    ]
    loss_side = loss_side.rename(columns={col_cd_claim: 'claim_id', col_cd_loss: 'loss_code_id'})

    _loss_desc_col = None
    if not loss_codes.empty:
        _loss_id_col = next(
            (c for c in [col_lc_id,'iId','id','loss_code_id'] if c and c in loss_codes.columns),
            None
        )
        _loss_desc_col = next(
            (c for c in [col_lc_desc,'loss_code_description','description','desc','name'] if c and c in loss_codes.columns),
            None
        )
        if _loss_id_col:
            lm = loss_codes[[_loss_id_col] + ([_loss_desc_col] if _loss_desc_col else [])].copy()
            lm[_loss_id_col] = as_str(lm[_loss_id_col])
            loss_side = loss_side.merge(lm, left_on='loss_code_id', right_on=_loss_id_col, how='left')
            if _loss_id_col != 'loss_code_id':
                loss_side.drop(columns=[_loss_id_col], inplace=True, errors='ignore')

    agg_dict_lc = {'loss_code_id': _agg_uniq}
    if _loss_desc_col:
        loss_side = loss_side.rename(columns={_loss_desc_col: 'loss_desc'})
        agg_dict_lc['loss_desc'] = _agg_uniq
    loss_side = loss_side.groupby('claim_id', as_index=False).agg(agg_dict_lc)

# 5) 合并三侧信息到一张表
triple_full = triple_df.copy()
for side in (seller_side, servicer_side, loss_side):
    if not side.empty:
        triple_full = triple_full.merge(side, on='claim_id', how='left')

# 6) 保存 CSV
out_path = save_csv(triple_full, 'p3seller_p3servicer_p4_overlap_claims.csv')
print(f"[P3-seller ∩ P3-servicer ∩ P4] claims: {len(triple_full)}")

# 7) （可选）写回 SQLite，便于直接 SQL
try:
    with sqlite3.connect(DB_PATH) as _conn:
        _conn.execute("DROP TABLE IF EXISTS p3seller_p3servicer_p4_overlap;")
        triple_full.to_sql('p3seller_p3servicer_p4_overlap', _conn, if_exists='replace', index=False)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_p3s_p3v_p4_claim ON p3seller_p3servicer_p4_overlap(claim_id);")
        print("  -> wrote table: p3seller_p3servicer_p4_overlap")
except Exception as e:
    print(f"  (warn) write-back to sqlite failed: {e}")

# ===================== Summary (only P3/P4) =====================
def _to_set(x):
    return set(x.dropna().astype(str)) if isinstance(x, pd.Series) else (x or set())

# P4（由 p4_out 里的 top loss codes 回查 claim）
p4_claims = set()
if not claim_details.empty and not p4_out.empty and col_cd_loss and col_cd_claim:
    _top_lc = set(p4_out.iloc[:, 0].astype(str))
    _tmp = claim_details[[col_cd_claim, col_cd_loss]].drop_duplicates().copy()
    _tmp[col_cd_claim] = _tmp[col_cd_claim].astype(str).str.strip()
    _tmp[col_cd_loss]  = _tmp[col_cd_loss].astype(str).str.strip()
    p4_claims = set(_tmp.loc[_tmp[col_cd_loss].isin(_top_lc), col_cd_claim])

# P3（来自上文的集合变量）
p3_seller_claims   = p3_seller_claims   if 'p3_seller_claims'   in globals() else set()
p3_servicer_claims = p3_servicer_claims if 'p3_servicer_claims' in globals() else set()

# 交集
p3_both_overlap = (p3_seller_claims or set()) & (p3_servicer_claims or set())
p3_p4_overlap   = ((p3_seller_claims or set()) | (p3_servicer_claims or set())) & (p4_claims or set())
triple_overlap  = (p3_seller_claims or set()) & (p3_servicer_claims or set()) & (p4_claims or set())

# Combined（不再包含 P1/P2）
combined_claims_all = (p4_claims or set()) | (p3_seller_claims or set()) | (p3_servicer_claims or set())

# 写 Summary
summary_rows = [
    ["P3_high_freq_sellers (entities)",         len(p3_seller)],
    ["P3_high_freq_servicers (entities)",       len(p3_servicer)],
    ["P3_seller_flagged_claims (claims)",       len(p3_seller_claims or set())],
    ["P3_servicer_flagged_claims (claims)",     len(p3_servicer_claims or set())],
    ["P4_repeat_loss_codes (codes)",            len(p4_out)],
    ["P4_flagged_claims (claims)",              len(p4_claims)],
    ["P3_seller∩P3_servicer (claims)",          len(p3_both_overlap)],
    ["P3∩P4_overlap_claims (claims)",           len(p3_p4_overlap)],
    ["P3_seller∩P3_servicer∩P4 (claims)",       len(triple_overlap)],
    ["Combined_flagged_claims_all (claims)",    len(combined_claims_all)],
]

summary_df = pd.DataFrame(summary_rows, columns=["metric", "count"])
summary_path = OUT_DIR / "summary_counts.csv"
summary_df.to_csv(summary_path, index=False)

print("\n================ SUMMARY ================")
for m, c in summary_rows:
    print(f"{m:40s} : {c}")
print(f"Summary CSV saved to: {summary_path}")
print("========================================\n")
