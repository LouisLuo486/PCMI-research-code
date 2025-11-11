# vscode_python_data_starter

A minimal, VSCode-friendly Python project to **ingest your datasets** (CSV/Excel) and export to **Parquet** and **SQLite** with automatic schema summaries.

## Quick start
1. (Optional) Create a virtual environment
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

3. Put your files into `data/` (CSV: `.csv`, Excel: `.xlsx`/`.xls`).

4. Run the ingester (choose your data dir):
   ```bash
   python -m src.main --data-dir ./data --out-dir ./out --db ./out/data.sqlite
   ```

5. Results:
   - Cleaned Parquet files in `out/parquet/`
   - A single SQLite DB at `out/data.sqlite` (one table per file)
   - Schema & stats summaries printed to console and saved to `out/summaries.json`

## Tips
- If your CSVs have non‑UTF8 encodings, try `--encoding latin1` or `--encoding cp1252`.
- Use `--sep` to set a delimiter (default `,`). For TSVs: `--sep "\t"`.
- You can re-run safely: tables are replaced on each run.

