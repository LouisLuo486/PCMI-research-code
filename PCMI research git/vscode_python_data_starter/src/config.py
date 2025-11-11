from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    data_dir: str = os.getenv("DATA_DIR", "./data")
    out_dir: str = os.getenv("OUT_DIR", "./out")
    sqlite_path: str = os.getenv("SQLITE_PATH", "./out/data.sqlite")
    csv_sep: str = os.getenv("CSV_SEP", ",")
    csv_encoding: str = os.getenv("CSV_ENCODING", "utf-8")
