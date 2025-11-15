#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
if [ ! -d .venv ]; then
  echo "Python environment (.venv) not found. Please create it before running run.sh" >&2
  exit 1
fi
source .venv/bin/activate
python pipeline.py --config config/best_pipeline.yaml
python evaluate.py --config config/best_pipeline.yaml --out-dir artifacts/eval
