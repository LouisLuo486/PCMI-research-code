# PCMI Research — Fraud Pattern Mining (P1–P8)

This repository builds a unified SQLite database from raw PCMI CSVs, runs fraud-suspect patterns (P1–P8), and materializes their unions/intersections back to the **claims** level. Outputs are written to `./out/data.sqlite` and optional CSVs under `./out/csv`.

## Quick Start (TL;DR)

```bash
# 1) Create environment & install deps
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
. .venv/bin/activate
pip install -r requirements.txt

# 2) Configure paths in src/config.py (DATA_DIR, DB_PATH, etc.)
#    DATA_DIR → folder containing raw CSVs (claims.csv, contracts.csv, ...)

# 3) Build the SQLite DB & import base tables
bash scripts/run_ingest.sh
# (On Windows without bash, open the script and run its python lines manually.)

# 4) Run pattern miners and mapping
python -m src.fraud_p1_p4_sqlite
python -m src.load_p3to8_to_sqlite
python -m src.patterns_v2_p5to8
python -m src.map_6way_to_claims
python -m src.make_in_claims_tables
# (Optional) If you already computed overlap CSVs:
python -m src.import_overlap_csv_to_sqlite

# 5) Explore results
sqlite3 ./out/data.sqlite
```

---

## Repository Layout

```
scripts/
  ├─ peek_contracts.py          # Small helper to quickly inspect contracts/data
  └─ run_ingest.sh              # One-shot pipeline to build SQLite + import base data

src/
  ├─ __pycache__/               # (auto-generated)
  ├─ config.py                  # Project config: paths, thresholds, chunk sizes
  ├─ ingest.py                  # CSV → canonical columns → SQLite (+ indexes)
  ├─ load_contracts_table_quick.py
  ├─ load_overlap_to_sqlite.py  # Normalize & import external overlap CSVs
  ├─ load_p3to8_to_sqlite.py    # Prepare/compute P3–P8 intermediate tables + indexes
  ├─ fraud_p1_p4_sqlite.py      # Compute P1–P4 (amount outliers, early claims, etc.)
  ├─ patterns_v2_p5to8.py       # Compute P5–P8 (VIN bursts, cross-shop, pairs, hashes)
  ├─ map_6way_to_claims.py      # Map multi-route flags back to claim-level rows
  ├─ make_in_claims_tables.py   # Create convenient claim-centric result tables/views
  ├─ import_overlap_csv_to_sqlite.py
  └─ main.py                    # Optional unified entry point (if you want to orchestrate)
```

---

## Configuration

Edit `src/config.py`:

```python
# Paths
DATA_DIR = "./data/raw"           # Folder containing original CSVs
OUT_DIR  = "./out"                # Output folder (auto-created)
DB_PATH  = "./out/data.sqlite"    # Unified SQLite DB

# Performance
CHUNK_SIZE = 200000               # CSV chunk size for imports

# Pattern thresholds (examples — tune as needed)
P3_MIN_COUNT   = 30               # High-frequency entity min count
P4_MIN_REPEAT  = 3                # Loss code min repeat
# Additional P5–P8 window sizes / rules live in their respective scripts
```

**Windows path tip:** prefer forward slashes or raw strings (`r"...""`) to avoid escaping issues.

---

## Pipeline

### 1) Base Import & Cleaning
- `ingest.py`: reads raw CSVs, normalizes column names/dtypes, writes to `DB_PATH` and creates indexes (`claims`, `contracts`, `claim_details`, `entity_sellers`, `entity_servicers`, etc.).
- `load_contracts_table_quick.py`: optional quick load/view for contracts.

### 2) Pattern Mining
- `fraud_p1_p4_sqlite.py`
  - **P1**: amount outliers
  - **P2**: early claims
  - **P3**: high-frequency sellers/servicers (with min-count & diversity constraints)
  - **P4**: repeated loss codes (frequency threshold)
  - Writes `flags_p1…p4` and a `flagged_claims` rollup.
- `patterns_v2_p5to8.py`
  - **P5**: VIN bursts (temporal spikes)
  - **P6**: cross-shop (same VIN, multiple repairers)
  - **P7**: cluster pairs (suspicious pair/cluster relations)
  - **P8**: invoice hashes (identical/near-duplicate invoices)
  - Writes `flags_p5…p8`.

### 3) Intersections / Mapping Back to Claims
- `map_6way_to_claims.py`: consolidates P3–P8 signals and maps them to `claims` (`iId` level).
- `make_in_claims_tables.py`: creates claim-centric tables/views (`in_claims_*`) for easy querying.
- `load_overlap_to_sqlite.py` & `import_overlap_csv_to_sqlite.py`: optional import of externally computed overlaps into the same DB.

---

## Outputs

- **SQLite**: `./out/data.sqlite`
  - Base: `claims`, `contracts`, `claim_details`, `entity_*`, …
  - Pattern flags: `flags_p1` … `flags_p8`, `flagged_claims`
  - Intersections/views: `in_claims_p3_p4`, `in_claims_p5_p6`, `in_claims_p3to8_all`, `overlaps_*` (names may vary by script)
- **CSV**: `./out/csv/*` (when export is enabled in scripts)
- **Logs**: `./out/logs/*` (if implemented)

Open the DB:

```bash
sqlite3 ./out/data.sqlite
.tables
.schema flagged_claims
SELECT * FROM flagged_claims LIMIT 20;
```

---

## Example SQL

```sql
-- VIN-level intersection of P5 ∩ P6
SELECT DISTINCT vin
FROM flags_p5_vin_bursts AS p5
JOIN flags_p6_cross_shop AS p6 USING (vin);

-- Full claims-level intersection: P3 ∩ P4 ∩ P5 ∩ P6 ∩ P7 ∩ P8
SELECT c.iId AS claim_id
FROM in_claims_p3to8_all AS c
WHERE c.p3=1 AND c.p4=1 AND c.p5=1 AND c.p6=1 AND c.p7=1 AND c.p8=1;

-- Inspect flagged claims with basic context
SELECT f.claim_id, cl.dServiceDate, cl.iContractId, cl.iServicerId
FROM flagged_claims f
JOIN claims cl ON cl.iId = f.claim_id
ORDER BY cl.dServiceDate DESC
LIMIT 50;
```

---

## Troubleshooting

**No rows produced / empty tables**
- Verify `DATA_DIR` points to the actual CSV folder.
- Confirm key columns match script expectations (`iId`, `iContractId`, `iServicerId`, `VIN`, etc.).
- Thresholds (e.g., `P3_MIN_COUNT`, `P4_MIN_REPEAT`) may be too strict for small samples—lower and rerun.
- Ensure base import finished successfully; check `.tables` in `data.sqlite`.

**“I don’t see any SQL tables”**
- All scripts write to `./out/data.sqlite`. Open it with `sqlite3` or a VS Code SQLite extension.

**Export intersections to CSV**
```sql
.headers on
.mode csv
.output ./out/csv/p3_p4_intersection.csv
SELECT * FROM in_claims_p3_p4;
.output stdout
```

**Windows without bash**
- Open `scripts/run_ingest.sh` and run each `python -m src.<module>` line in PowerShell.

---

## Development Notes

- Index large tables (scripts already create common indexes on `claims(iId)`, `claims(iContractId)`, `contracts(iId)`, `claim_details(iClaimId)`, etc.).
- Use chunked CSV imports (`CHUNK_SIZE`) for memory efficiency.
- Keep tunable thresholds/windows at the top of each pattern script or centralized in `config.py`.
- If you prefer one entrypoint, orchestrate the order in `src/main.py`:
  1) `ingest.py`
  2) `fraud_p1_p4_sqlite.py`
  3) `load_p3to8_to_sqlite.py`
  4) `patterns_v2_p5to8.py`
  5) `map_6way_to_claims.py`
  6) `make_in_claims_tables.py`

---

## License

For research use. Please acknowledge this repository and the PCMI dataset in publications.
