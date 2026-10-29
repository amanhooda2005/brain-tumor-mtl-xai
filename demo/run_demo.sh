#!/usr/bin/env bash
# Start the live demo (macOS / Linux). Extra options are passed through, e.g.:
#   bash demo/run_demo.sh --data_dir /path/to/data/raw
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
python3 demo/app.py "$@"
