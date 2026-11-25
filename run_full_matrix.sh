#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
JOBS="${JOBS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
python research_runner.py --seeds 10 --jobs "$JOBS" 2>&1 | tee runs_full.log
python analysis/analyze.py --units runs/units --out runs/analysis
