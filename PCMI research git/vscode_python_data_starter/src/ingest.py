from __future__ import annotations
import os, re, json, pathlib
from typing import Dict, Any, List, Tuple
import pandas as pd
from sqlalchemy import create_engine

SUPPORTED_EXTS = {".csv", ".xlsx", ".xls"}

def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "table"

def discover_files(data_dir: str) -> List[pathlib.Path]:
    p = pathlib.Path(data_dir)
    files = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTS and f.is_file()]
    files.sort()
    return files

def read_any(path: pathlib.Path, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
    elif ext in (".xlsx", ".xls"):
        # read first sheet by default
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def summarize_df(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    # Basic schema & sample stats
    summary: Dict[str, Any] = {
        "table": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": [],
    }
    for col in df.columns:
        s = df[col]
        item: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
        }
        # Add small sample stats for numeric columns
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            # ensure JSON serializable
            item["stats"] = {k: (None if pd.isna(v) else float(v)) for k, v in desc.items()}
        else:
            # top frequent categories (up to 5)
            top = s.astype("string").value_counts(dropna=True).head(5).to_dict()
            item["top_values"] = {k: int(v) for k, v in top.items()}
        summary["columns"].append(item)
    return summary

def to_sqlite(df: pd.DataFrame, table_name: str, sqlite_path: str) -> None:
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    engine = create_engine(f"sqlite:///{sqlite_path}")
    # Replace existing table on each run
    df.to_sql(table_name, engine, if_exists="replace", index=False)

def to_parquet(df: pd.DataFrame, out_dir: str, table_name: str) -> str:
    pq_dir = os.path.join(out_dir, "parquet")
    os.makedirs(pq_dir, exist_ok=True)
    out_path = os.path.join(pq_dir, f"{table_name}.parquet")
    df.to_parquet(out_path, index=False)
    return out_path

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Clean column names
    clean = []
    for c in df.columns:
        c2 = re.sub(r"\s+", "_", str(c)).strip("_")
        c2 = re.sub(r"[^0-9a-zA-Z_]", "_", c2)
        c2 = re.sub(r"_+", "_", c2).lower()
        clean.append(c2 or "col")
    df = df.copy()
    df.columns = clean
    return df

def process_all(data_dir: str, out_dir: str, sqlite_path: str, sep: str = ",", encoding: str = "utf-8") -> Dict[str, Any]:
    files = discover_files(data_dir)
    if not files:
        print(f"[WARN] No data files found under {data_dir}. Supported: {', '.join(SUPPORTED_EXTS)}")
        return {"files": [], "summaries": []}

    os.makedirs(out_dir, exist_ok=True)
    all_summaries = []
    for f in files:
        print(f"[INFO] Reading: {f}")
        df = read_any(f, sep=sep, encoding=encoding)
        df = sanitize_columns(df)
        table = _slugify(f.stem)
        # Export Parquet & SQLite
        pq_path = to_parquet(df, out_dir, table)
        to_sqlite(df, table, sqlite_path)
        # Summarize
        summary = summarize_df(df, table)
        summary["source_path"] = str(f)
        summary["parquet_path"] = str(pq_path)
        summary["sqlite_table"] = table
        all_summaries.append(summary)
        print(f"[OK]  -> table='{table}', rows={df.shape[0]}, cols={df.shape[1]}")

    # Save a combined summary
    sum_path = os.path.join(out_dir, "summaries.json")
    with open(sum_path, "w", encoding="utf-8") as w:
        json.dump({"files": [str(f) for f in files], "summaries": all_summaries}, w, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote summary: {sum_path}")
    return {"files": [str(f) for f in files], "summaries": all_summaries}
