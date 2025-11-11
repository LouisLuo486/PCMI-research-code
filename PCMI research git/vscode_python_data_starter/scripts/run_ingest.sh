#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --data-dir ./data --out-dir ./out --db ./out/data.sqlite
