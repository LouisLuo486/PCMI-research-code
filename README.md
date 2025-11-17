# PCMI Research — Fraud Pattern Mining (P3–P8, MSSQL)

This repository builds a **unified Microsoft SQL Server (MSSQL) database** from raw PCMI CSVs, runs fraud-suspect patterns **P3–P8**, and materializes their unions/intersections back to the **claim** level.  
Main outputs are:

- MSSQL tables (claims-level flags & intersections)
- Optional CSV exports under `./out/`

> Note: Legacy P1/P2 (amount outliers, early claims) are no longer used in the main pipeline. The core focus is **P3–P8**.

---

## Quick Start (TL;DR)

```bash
# 1) Create environment & install deps
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
. .venv/bin/activate

pip install -r requirements.txt
```

2. **Configure connection & paths**

Create a `.env` file in the project root (or edit `src/config.py`):

```env
MSSQL_CONN_STR=mssql+pyodbc://USER:PASSWORD@SERVER/DATABASE?driver=ODBC+Driver+17+for+SQL+Server
DATA_DIR=./data
OUT_DIR=./out
```

3. **Import base CSVs into MSSQL**

```bash
# Example: import claims
python src/import_csv_to_mssql.py --csv data/claims.csv --table claims

# Example: import claim_details
python src/import_csv_to_mssql.py --csv data/claim_details.csv --table claim_details

# Other core tables (names may vary slightly in your folder)
python src/import_csv_to_mssql.py --csv data/contracts.csv                        --table contracts
python src/import_csv_to_mssql.py --csv data/contract_vehicle.csv                 --table contract_vehicle
python src/import_csv_to_mssql.py --csv data/coverage_plans_seller_inclusion.csv  --table coverage_plans_seller_inclusion
python src/import_csv_to_mssql.py --csv data/entity_sellers.csv                   --table entity_sellers
python src/import_csv_to_mssql.py --csv data/entity_servicers.csv                 --table entity_servicers
python src/import_csv_to_mssql.py --csv data/claim_details.csv                    --table claim_details
```

4. **Run pattern miners and mapping (MSSQL-based)**

```bash
# P5–P8 patterns + P3–P8 intersections
python src/patterns_v2_p5to8.py

# Map P3–P8 intersection back to full claims table
python src/make_in_claims_tables.py   --base claims   --ids  p3p4p5p6p7p8_claims   --out  p3p4p5p6p7p8_claims_in_claims

# Enrich intersection with seller_id / seller_name (via contracts & coverage)
python src/add_seller_to_intersections.py
```

5. **Explore results in MSSQL**

Use SQL Server Management Studio / Azure Data Studio / any MSSQL client:

```sql
SELECT TOP 50 *
FROM p3p4p5p6p7p8_claims_in_claims_with_seller
ORDER BY service_date DESC;
```

CSV exports will be under `./out/`.

---

## Repository Layout

```text
project_root/
  data/
    claims.csv
    claim_details.csv
    contracts.csv
    contract_vehicle.csv
    coverage_plans_seller_inclusion.csv
    entity_sellers.csv
    entity_servicers.csv
    product_loss_codes.csv
    ...

  out/
    p5_vin_bursts.csv
    p6_cross_shop.csv
    p7_cluster_pairs.csv
    p8_invoice_hashes.csv
    p8_invoice_flagged_claims.csv
    p3p4p5p6p7p8_claims.csv
    p3p4p5p6p7p8_claims_in_claims.csv
    p3p4p5p6p7p8_claims_in_claims_with_seller.csv
    ...

  src/
    config.py                     # Reads .env, exposes DATA_DIR, OUT_DIR, MSSQL_CONN_STR
    import_csv_to_mssql.py        # Generic CSV → MSSQL table importer (type inference + indexes)
    patterns_v2_p5to8.py          # P5–P8 pattern mining + P3–P8 intersections
    make_in_claims_tables.py      # Join intersection IDs back to claims (MSSQL, SELECT INTO)
    add_seller_to_intersections.py# Attach seller_id / seller_name via contracts & coverage
    ...
```

> Some filenames may differ slightly; adjust commands to your actual file names.

---

## Configuration

### `src/config.py` / `.env`

Key settings:

- `MSSQL_CONN_STR`  
  MSSQL SQLAlchemy connection string (using `pyodbc` driver).

- `DATA_DIR`  
  Folder containing original CSVs (e.g. `./data`).

- `OUT_DIR`  
  Output folder for CSV exports (e.g. `./out`).

Thresholds (e.g. windows / minimum counts) are defined in `patterns_v2_p5to8.py`, for example:

```python
# P5
P5_WINDOW_DAYS   = 14
P5_MIN_CLAIMS    = 3

# P6
P6_WINDOW_DAYS   = 30
P6_MIN_SERVICERS = 2

# P7
P7_MIN_DISTINCT_VINS = 30
P7_MIN_DISTINCT_PH   = 10

# P8
P8_MIN_DUP_HASH  = 2
```

Tune these to control sensitivity vs. precision.

---

## Patterns (P3–P8)

> P1 / P2 are legacy and not used in this pipeline. The core focus is P3–P8.

### P3 — High-Frequency Sellers / Servicers

- **Servicer side**: counts claims per servicer (`iservicerid` / `servicer_id`) and flags high-volume servicers based on quantile threshold + absolute minimum.
- **Seller side**: uses `coverage_plans_seller_inclusion` + `contracts` to count **weighted claims per seller** and enforce a **coverage diversity** requirement.

These produce sets of **high-frequency entities**; their claim_ids are used inside the P3–P8 intersection logic.

---

### P4 — High-Frequency Loss Codes

- Uses `claim_details` + `product_loss_codes`.
- For each loss code:
  - Count distinct claims,
  - Count distinct servicers.
- Apply min-count and min-diversity thresholds.  
  Loss codes above both thresholds are labeled as **P4 “hot” codes**, and all related claims are collected.

---

### P5 — VIN Bursts (Short-Window Multi-Claims)

Detect VINs with many claims in a short period.

- For each VIN, sort by service date and slide a `P5_WINDOW_DAYS` window.
- If the window contains ≥ `P5_MIN_CLAIMS` claims, record a burst.
- Output includes:
  - `vin`
  - `window_start`, `window_end`
  - `num_claims`
  - `distinct_servicers`
  - `claim_ids` (pipe-separated)

CSV: `out/p5_vin_bursts.csv`.

---

### P6 — Cross-Shop (Multi-Servicer per VIN Window)

Detect VINs that visit multiple servicers in a short period.

- For each VIN, sort by date and slide a `P6_WINDOW_DAYS` window.
- Count distinct servicers per window.
- If distinct servicers ≥ `P6_MIN_SERVICERS`, record a cross-shop event.

Output columns include:

- `vin`
- `window_start`, `window_end`
- `distinct_servicers`
- `num_claims`
- `servicer_ids`
- `claim_ids`

CSV: `out/p6_cross_shop.csv`.

---

### P7 — Coordinated Clusters (Servicer × Seller Pairs)

Detect dense **servicer–seller** pairs that appear across many VINs and policyholders.

Steps (done inside `patterns_v2_p5to8.py`):

1. Build claim-level detail combining:
   - `claims` (servicer_id, contract_id, claim_id)
   - `contracts` (coverage, policyholder_id)
   - `coverage_plans_seller_inclusion` (coverage → seller_id)
   - `contract_vehicle` / `contracts` (VIN)

2. Group by `(servicer_id, seller_id)` and compute:
   - `distinct_vins`
   - `distinct_policyholders` (if available)
   - `claims`

3. Apply thresholds (e.g. `P7_MIN_DISTINCT_VINS`, `P7_MIN_DISTINCT_PH`).

Outputs:

- `out/p7_cluster_pairs.csv`  
  A list of dense `(servicer_id, seller_id)` pairs with counts.

Internally, a set `p7_pair_claims_set` tracks all claim_ids belonging to these pairs for intersection logic.

---

### P8 — Invoice Cloning (Duplicate Part/Qty/Price)

Use `claim_details` to find **identical invoice structures** across claims:

1. Detect part / qty / price columns and normalize each detail row into:
   - `"<part>:<qty>:<price>"`.
2. For each claim_id:
   - Sort and join all line signatures into an `invoice_signature`.
   - Hash to `invoice_hash` (MD5).
3. Group by `invoice_hash` and retain hashes where:
   - `num_claims ≥ P8_MIN_DUP_HASH`.

Outputs:

- `out/p8_invoice_hashes.csv` — hash-level stats.
- `out/p8_invoice_flagged_claims.csv` — claim-level mapping to `invoice_hash`.

---

## Intersections & Mapping

### P3–P8 Intersection (Claim-Level)

`patterns_v2_p5to8.py` gathers claim sets from:

- P3 high-frequency entities,
- P4 high-frequency loss codes,
- P5 VIN bursts,
- P6 cross-shop,
- P7 dense pairs,
- P8 cloned invoices,

and computes:

```text
P3 ∩ P4 ∩ P5 ∩ P6 ∩ P7 ∩ P8
```

The result is written as:

- MSSQL table: `p3p4p5p6p7p8_claims`
- CSV: `out/p3p4p5p6p7p8_claims.csv`

This is the **core suspicious-claim set**.

---

### Mapping Back to Full Claims

`src/make_in_claims_tables.py` is an MSSQL-based helper that:

- Automatically detects primary key columns in:
  - `claims` table (e.g. `iid`, `claim_id`, etc.),
  - `p3p4p5p6p7p8_claims` table.
- Performs:

```sql
SELECT INTO p3p4p5p6p7p8_claims_in_claims
FROM claims AS c
JOIN p3p4p5p6p7p8_claims AS x
  ON c.claim_key = x.claim_key
LEFT JOIN contract_vehicle AS cv   -- attach VIN (if available)
LEFT JOIN p8_invoice_claims AS p8  -- attach invoice_hash (if available)
```

Run:

```bash
python src/make_in_claims_tables.py   --base claims   --ids  p3p4p5p6p7p8_claims   --out  p3p4p5p6p7p8_claims_in_claims
```

This also exports a CSV:

- `out/p3p4p5p6p7p8_claims_in_claims.csv`

---

### Adding Seller Context

`src/add_seller_to_intersections.py` (name may vary) enriches the intersection with seller info:

- Reads `out/p3p4p5p6p7p8_claims_in_claims.csv`.
- Joins:
  - `contracts` (icontractid → coverage),
  - `coverage_plans_seller_inclusion` (coverage → seller_id),
  - `entity_sellers` (seller_id → seller_name).

Outputs:

- MSSQL table: `p3p4p5p6p7p8_claims_in_claims_with_seller`
- CSV: `out/p3p4p5p6p7p8_claims_in_claims_with_seller.csv`

This final table is intended for **manual review and case studies**.

---

## Example MSSQL Queries

```sql
-- Check how many claims are in the final P3–P8 intersection
SELECT COUNT(*) AS suspicious_claims
FROM p3p4p5p6p7p8_claims_in_claims_with_seller;

-- Inspect a random sample of suspicious claims
SELECT TOP 50
    claim_id,
    vin,
    servicer_id,
    seller_id,
    seller_name,
    service_date,
    invoice_hash
FROM p3p4p5p6p7p8_claims_in_claims_with_seller
ORDER BY NEWID();

-- See which servicer–seller pairs dominate in the final set
SELECT
    servicer_id,
    seller_id,
    seller_name,
    COUNT(*) AS num_claims
FROM p3p4p5p6p7p8_claims_in_claims_with_seller
GROUP BY servicer_id, seller_id, seller_name
ORDER BY num_claims DESC;
```

---

## Troubleshooting

**No rows in P3–P8 intersection**

- Confirm all base tables exist in MSSQL:
  - `claims`, `claim_details`, `contracts`, `coverage_plans_seller_inclusion`, `entity_sellers`, `entity_servicers`, etc.
- Check that key columns are detected correctly (IDs, VINs, dates).  
  The scripts use flexible column-name matching, but very unusual schemas may require adjusting candidate lists.
- Thresholds may be too strict for small datasets. Try lowering:
  - `P7_MIN_DISTINCT_VINS`, `P7_MIN_DISTINCT_PH`
  - `P5_MIN_CLAIMS`, `P6_MIN_SERVICERS`
  - `P8_MIN_DUP_HASH`

**Import script fails on types**

- Make sure CSV encoding is compatible (UTF-8 with or without BOM).
- Large numeric IDs may need to be kept as TEXT; type inference in `import_csv_to_mssql.py` can be tuned.

**“Table not found”**

- Verify that `DATA_DIR` and `MSSQL_CONN_STR` are correct.
- In MSSQL client, run `SELECT * FROM INFORMATION_SCHEMA.TABLES` to confirm.

---

## Development Notes

- Indexing is important for performance; the scripts typically create indexes on:
  - `claims` key columns (e.g., `claim_id` / `iid`),
  - `contract` keys,
  - `claim_details` claim key,
  - join keys in intersection tables.
- Pattern thresholds and window sizes are at the top of `patterns_v2_p5to8.py` — centralize tuning there.
- For new patterns (P9, P10, …), follow the same design:
  1. Build a clean claim-level set of IDs.
  2. Add that set into the intersection logic.
  3. Extend downstream mapping scripts if needed.

---

## License

For internal research / academic use.  
If you use this codebase in publications, please acknowledge the PCMI dataset and this repository.
