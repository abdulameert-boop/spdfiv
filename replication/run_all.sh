#!/usr/bin/env bash
# Reproduce every number in the paper and rebuild the PDF.
# Usage: bash replication/run_all.sh path/to/v113i06.dta path/to/W.csv
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/replication"
python -m unittest discover -s tests
python -W ignore replication/monte_carlo.py --reps 1000 --workers "${WORKERS:-2}"
python replication/make_tables.py
python replication/illustration.py
python -W ignore replication/bank_replication.py "$1" "$2"
cd paper
pdflatex spdfiv && bibtex spdfiv && pdflatex spdfiv && pdflatex spdfiv
