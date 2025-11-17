from __future__ import annotations
import argparse
from .config import Config
from .ingest import process_all

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ingest CSV/Excel to Parquet + SQLite with summaries")
    ap.add_argument("--data-dir", type=str, default=None, help="Directory containing input files")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory for outputs")
    ap.add_argument("--db", type=str, default=None, help="SQLite path (e.g., ./out/data.sqlite)")
    ap.add_argument("--sep", type=str, default=None, help="CSV separator (default ,)")
    ap.add_argument("--encoding", type=str, default=None, help="CSV encoding (default utf-8)")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = Config()
    data_dir = args.data_dir or cfg.data_dir
    out_dir = args.out_dir or cfg.out_dir
    sqlite_path = args.db or cfg.sqlite_path
    sep = args.sep or cfg.csv_sep
    enc = args.encoding or cfg.csv_encoding

    result = process_all(data_dir, out_dir, sqlite_path, sep=sep, encoding=enc)
    # Print a short recap
    print(f"\n=== Recap ===")
    for s in result.get("summaries", []):
        print(f"- {s['table']}: rows={s['rows']} cols={s['cols']} (from {s['source_path']})")

if __name__ == "__main__":
    main()
